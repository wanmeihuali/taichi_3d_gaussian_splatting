# %%
import numpy as np
import math
import tqdm
import taichi as ti
from taichi.lang import impl
import torch
from torch import Tensor
from taichi._lib import core as _ti_core

# %%
arch = ti.vulkan
ti.init(arch=arch)

# %%
SCAN_BLOCK_SIZE = 1024
WARP_SIZE_LOG2 = 5
WARP_SIZE = 1 << WARP_SIZE_LOG2
HALF_WARP_SIZE = 1 << (WARP_SIZE_LOG2 - 1)
WARP_NUM = SCAN_BLOCK_SIZE // WARP_SIZE
SHM_SIZE = WARP_NUM * (WARP_SIZE + HALF_WARP_SIZE)


@ti.func
def scan_warp(
    value,
    lane_idx,
):
    for i in ti.static(range(WARP_SIZE_LOG2)):
        new_value = ti.simt.warp.shfl_up_i32(ti.u32(0xFFFFFFFF), value, 1 << i)
        if lane_idx >= (1 << i):
            value += new_value
    return value


@ti.func
def reduce_warp(
    value,
    lane_idx,
):
    for i in ti.static(range(WARP_SIZE_LOG2)):
        value += ti.simt.warp.shfl_down_i32(ti.u32(0xFFFFFFFF),
                                            value, 1 << (WARP_SIZE_LOG2 - i - 1))
    return value


@ti.func
def reduce_block_by_cuda(
    value,
    thread_idx,
):
    lane_idx: ti.i32 = thread_idx & (WARP_SIZE - 1)
    warp_idx: ti.i32 = thread_idx >> WARP_SIZE_LOG2
    warp_reduced_value = ti.simt.block.SharedArray(
        REDUCE_WARP_NUM, dtype=ti.i32)
    value = reduce_warp(value, lane_idx)
    ti.simt.block.sync()
    if lane_idx == 0:
        warp_reduced_value[warp_idx] = value
    ti.simt.block.sync()
    assert REDUCE_WARP_NUM == WARP_SIZE
    if warp_idx == 0:
        value = reduce_warp(warp_reduced_value[lane_idx], lane_idx)
    return value


@ti.func
def scan_block_by_cuda(
    value,
    thread_idx,
):
    warp_idx: ti.i32 = thread_idx >> WARP_SIZE_LOG2
    lane_idx: ti.i32 = thread_idx & (WARP_SIZE - 1)
    warp_sum = ti.simt.block.SharedArray(
        shape=(WARP_SIZE, ), dtype=ti.i32)
    value = scan_warp(value, lane_idx)
    ti.simt.block.sync()
    if lane_idx == (WARP_SIZE - 1):
        warp_sum[warp_idx] = value
    ti.simt.block.sync()
    assert WARP_NUM == WARP_SIZE
    if warp_idx == 0:
        warp_sum[lane_idx] = scan_warp(warp_sum[lane_idx], lane_idx)
    ti.simt.block.sync()
    if warp_idx > 0:
        value += warp_sum[warp_idx - 1]
    ti.simt.block.sync()
    return value


# %%

if arch == ti.cuda:
    def test_reduce_warp():
        a = ti.field(dtype=ti.i32, shape=WARP_SIZE)

        @ti.kernel
        def foo():
            ti.loop_config(block_dim=WARP_SIZE)
            for i in range(WARP_SIZE):
                lane_idx = i & (WARP_SIZE - 1)
                a[i] = reduce_warp(a[i], lane_idx)

        for i in range(32):
            a[i] = i

        foo()
        assert a[0] == 32 * (31 + 0) // 2

    test_reduce_warp()

    def test_scan_warp():
        a = ti.field(dtype=ti.i32, shape=WARP_SIZE)

        @ti.kernel
        def foo():
            ti.loop_config(block_dim=WARP_SIZE)
            for i in range(WARP_SIZE):
                lane_idx = i & (WARP_SIZE - 1)
                a[i] = scan_warp(a[i], lane_idx)

        for i in range(32):
            a[i] = i

        foo()
        for i in range(32):
            assert a[i] == i * (i + 1) // 2

    test_scan_warp()
# %%

if arch == ti.metal or arch == ti.vulkan:
    @ti.kernel
    def detect_subgroup_size() -> ti.i32:
        return ti.simt.subgroup.group_size()

    SUBGROUP_SIZE = detect_subgroup_size()
    SUBGROUP_SIZE_LOG2 = int(math.log2(SUBGROUP_SIZE))

    REDUCE_BLOCK_SIZE = SUBGROUP_SIZE * SUBGROUP_SIZE
    SCAN_BLOCK_SIZE = SUBGROUP_SIZE * SUBGROUP_SIZE
    REDUCE_WARP_NUM = REDUCE_BLOCK_SIZE // SUBGROUP_SIZE


@ti.func
def reduce_block_by_subgroup(
    value,
    thread_idx,
):
    warp_reduced_value = ti.simt.block.SharedArray(
        REDUCE_BLOCK_SIZE // REDUCE_WARP_NUM, dtype=ti.i32)
    value = ti.simt.subgroup.reduce_add(value)
    subgroup_id = thread_idx // SUBGROUP_SIZE
    ti.simt.block.sync()
    if ti.simt.subgroup.elect():
        warp_reduced_value[subgroup_id] = value
    ti.simt.block.sync()
    if subgroup_id == 0:
        value = ti.simt.subgroup.reduce_add(
            warp_reduced_value[ti.simt.subgroup.invocation_id()])
    ti.simt.block.sync()
    return value
# %%


@ti.func
def reduce_block(
    value,
    thread_idx,
):
    arch = impl.get_runtime().prog.config().arch
    if ti.static(arch == _ti_core.cuda):
        return reduce_block_by_cuda(value, thread_idx)
    elif ti.static(arch == _ti_core.vulkan or arch == _ti_core.metal):
        return reduce_block_by_subgroup(value, thread_idx)
    else:
        raise ValueError(f"reduce_block is not supported for arch {arch}")


def test_reduce_block():
    a = ti.field(dtype=ti.i32, shape=REDUCE_BLOCK_SIZE)
    b = ti.field(dtype=ti.i32, shape=REDUCE_BLOCK_SIZE)

    @ti.kernel
    def foo():
        ti.loop_config(block_dim=REDUCE_BLOCK_SIZE)
        for i in range(REDUCE_BLOCK_SIZE):
            thread_idx = i & (REDUCE_BLOCK_SIZE - 1)
            b[i] = reduce_block(a[i], thread_idx)

    for i in range(REDUCE_BLOCK_SIZE):
        a[i] = i

    foo()

    assert b[0] == REDUCE_BLOCK_SIZE * (REDUCE_BLOCK_SIZE - 1) // 2


test_reduce_block()

# %%


@ti.func
def scan_block_by_subgroup(
    value,
    thread_idx,
):
    warp_idx: ti.i32 = thread_idx >> SUBGROUP_SIZE_LOG2
    lane_idx: ti.i32 = thread_idx & (SUBGROUP_SIZE - 1)
    warp_sum = ti.simt.block.SharedArray(
        shape=(SUBGROUP_SIZE, ), dtype=ti.i32)
    value = ti.simt.subgroup.inclusive_add(value)
    ti.simt.block.sync()
    if lane_idx == (SUBGROUP_SIZE - 1):
        warp_sum[warp_idx] = value
    ti.simt.block.sync()
    if warp_idx == 0:
        warp_sum[lane_idx] = ti.simt.subgroup.inclusive_add(warp_sum[lane_idx])
    ti.simt.block.sync()
    if warp_idx > 0:
        value += warp_sum[warp_idx - 1]
    ti.simt.block.sync()
    return value


@ti.func
def scan_block(
    value,
    thread_idx,
):
    arch = impl.get_runtime().prog.config().arch
    if ti.static(arch == _ti_core.cuda):
        return scan_block_by_cuda(value, thread_idx)
    elif ti.static(arch == _ti_core.vulkan or arch == _ti_core.metal):
        return scan_block_by_subgroup(value, thread_idx)
    else:
        raise ValueError(f"reduce_block is not supported for arch {arch}")


def test_scan_block():
    a = ti.field(dtype=ti.i32, shape=SCAN_BLOCK_SIZE)

    @ti.kernel
    def foo():
        ti.loop_config(block_dim=SCAN_BLOCK_SIZE)
        for i in range(SCAN_BLOCK_SIZE):
            thread_idx = i & (SCAN_BLOCK_SIZE - 1)
            a[i] = scan_block_by_subgroup(a[i], thread_idx)

    for i in range(SCAN_BLOCK_SIZE):
        a[i] = i

    foo()
    for i in range(SCAN_BLOCK_SIZE):
        assert a[i] == i * \
            (i + 1) // 2, f"a[{i}] = {a[i]}, expected {i * (i + 1) // 2}"


test_scan_block()
# %%

# %%
SCAN_PART_NUM = 1024


@ti.kernel
def reduce_then_scan(
    num_elements: ti.i32,
    inputs: ti.template(),
    outputs: ti.template(),
    global_base_sum: ti.template(),
):
    part_size = (num_elements + SCAN_PART_NUM - 1) // SCAN_PART_NUM
    ti.loop_config(block_dim=SCAN_BLOCK_SIZE)
    for idx in range(SCAN_PART_NUM * SCAN_BLOCK_SIZE):
        thread_idx = idx & (SCAN_BLOCK_SIZE - 1)
        block_idx = idx // SCAN_BLOCK_SIZE
        part_begin = block_idx * part_size
        part_end = ti.min(part_begin + part_size, num_elements)
        part_sum: ti.i32 = 0
        offset = part_begin + thread_idx
        while offset < part_end:
            part_sum += inputs[offset]
            offset += SCAN_BLOCK_SIZE
        part_sum = reduce_block(part_sum, thread_idx)
        if thread_idx == 0:
            global_base_sum[block_idx] = part_sum

    ti.loop_config(block_dim=SCAN_BLOCK_SIZE)
    for idx in range(SCAN_BLOCK_SIZE):
        thread_idx = idx & (SCAN_BLOCK_SIZE - 1)
        global_base_sum[thread_idx] = scan_block(
            value=global_base_sum[thread_idx],
            thread_idx=thread_idx)
    ti.loop_config(block_dim=SCAN_BLOCK_SIZE)
    for idx in range(SCAN_PART_NUM * SCAN_BLOCK_SIZE):
        thread_idx = idx & (SCAN_BLOCK_SIZE - 1)
        block_idx = idx // SCAN_BLOCK_SIZE

        base_sum = ti.simt.block.SharedArray(1, dtype=ti.i32)
        if thread_idx == 0:
            base_sum[0] = global_base_sum[block_idx -
                                          1] if block_idx > 0 else 0

        part_begin = block_idx * part_size
        part_end = part_begin + part_size
        offset = part_begin + thread_idx
        while offset < part_end:
            value: ti.i32 = inputs[offset] if offset < num_elements else 0
            value = scan_block(value, thread_idx)
            if offset < num_elements:
                outputs[offset] = value + base_sum[0]
            ti.simt.block.sync()
            if thread_idx == (SCAN_BLOCK_SIZE - 1):
                base_sum[0] += value
            ti.simt.block.sync()
            offset += SCAN_BLOCK_SIZE


# %%
N = 100000
# %%

inputs = ti.field(ti.i32, shape=N)


@ti.kernel
def random_write(
    inputs: ti.template(),
):
    for i in range(inputs.shape[0]):
        inputs[i] = ti.random(ti.i32) % 100
        # inputs[i] = i % 100


random_write(inputs)
outputs = ti.field(ti.i32, shape=N)
g_base_sum = ti.field(ti.i32, shape=SCAN_PART_NUM)

# %%
reduce_then_scan(N, inputs, outputs, g_base_sum)
assert np.allclose(outputs.to_numpy(), inputs.to_numpy().cumsum())
# %%

for _ in tqdm.trange(1000):
    reduce_then_scan(N, inputs, outputs, g_base_sum)

ti.sync()
for _ in tqdm.trange(10000):
    reduce_then_scan(N, inputs, outputs, g_base_sum)
ti.sync()

# %%
# decoupled look back scan requires state and prefix sum to be int64, which is not supported by metal
SCAN_X_STATE = 0
SCAN_A_STATE = 1
SCAN_P_STATE = 2
SCAN_PART_SIZE = 4096
MAX_LOCAL_INPUTS = (SCAN_PART_SIZE + SCAN_BLOCK_SIZE -
                    1) // SCAN_BLOCK_SIZE


@ti.kernel
def decoupled_look_back_scan(
    num_elements: ti.i32,
    inputs: ti.types.ndarray(dtype=ti.i32, ndim=1),
    outputs: ti.types.ndarray(dtype=ti.i32, ndim=1),
    global_counter: ti.types.ndarray(dtype=ti.i32, ndim=1),
    state_and_prefix_sum: ti.types.ndarray(dtype=ti.i64, ndim=1),
):
    scan_part_num = (num_elements + SCAN_PART_SIZE - 1) // SCAN_PART_SIZE
    global_counter[0] = 0
    for idx in state_and_prefix_sum:
        state_and_prefix_sum[idx] = 0
    ti.loop_config(block_dim=SCAN_BLOCK_SIZE)
    for idx in range(scan_part_num * SCAN_BLOCK_SIZE):
        thread_idx = idx & (SCAN_BLOCK_SIZE - 1)
        current_global_counter = ti.simt.block.SharedArray(1, dtype=ti.i32)
        if thread_idx == 0:
            current_global_counter[0] = ti.atomic_add(global_counter[0], 1)
        ti.simt.block.sync()  # ensure all threads in block have the same global counter
        # use the global counter to determine the block index, so that we can ensure the block order
        block_idx = current_global_counter[0]
        base_sum = ti.simt.block.SharedArray(1, dtype=ti.i32)
        base_sum[0] = 0
        local_inputs = ti.Vector(
            [0 for _ in ti.static(range(MAX_LOCAL_INPUTS))], dt=ti.i32)
        part_sum: ti.i32 = 0
        offset = block_idx * SCAN_PART_SIZE + thread_idx
        for local_inputs_idx in ti.static(range(SCAN_PART_SIZE // SCAN_BLOCK_SIZE)):
            local_inputs[local_inputs_idx] = inputs[offset] if offset < num_elements else 0
            part_sum += local_inputs[local_inputs_idx]
            offset += SCAN_BLOCK_SIZE

        if block_idx > 0:
            part_sum = reduce_block(part_sum, thread_idx)
            if thread_idx == 0:
                part_state_and_prefix_sum: ti.i64 = (
                    ti.cast(SCAN_A_STATE, ti.i64) << 32) | part_sum
                ti.atomic_max(state_and_prefix_sum[block_idx],
                              part_state_and_prefix_sum)
                predecessor_idx = block_idx - 1
                exclusive_prefix: ti.i32 = 0
                while True:
                    predecessor_state_and_prefix_sum: ti.i64 = ti.atomic_or(state_and_prefix_sum[
                        predecessor_idx], ti.i64(0))
                    if (predecessor_state_and_prefix_sum >> 32) == SCAN_A_STATE:
                        exclusive_prefix += ti.cast(
                            predecessor_state_and_prefix_sum & ti.i64(0xFFFFFFFF), ti.i32)
                        predecessor_idx -= 1
                    elif (predecessor_state_and_prefix_sum >> 32) == SCAN_P_STATE:
                        exclusive_prefix += ti.cast(
                            predecessor_state_and_prefix_sum & ti.i64(0xFFFFFFFF), ti.i32)
                        break
                    # else: SCAN_X_STATE, wait for the predecessor to finish
                inclusive_prefix = exclusive_prefix + part_sum
                part_state_and_prefix_sum: ti.i64 = (
                    ti.cast(SCAN_P_STATE, ti.i64) << 32) | inclusive_prefix
                ti.atomic_max(state_and_prefix_sum[block_idx],
                              part_state_and_prefix_sum)
                base_sum[0] = exclusive_prefix
            ti.simt.block.sync()

        part_begin = block_idx * SCAN_PART_SIZE
        part_end = part_begin + SCAN_PART_SIZE
        offset = part_begin + thread_idx
        lastest_value = 0
        assert SCAN_PART_SIZE % SCAN_BLOCK_SIZE == 0
        for local_inputs_idx in ti.static(range(SCAN_PART_SIZE // SCAN_BLOCK_SIZE)):
            value: ti.i32 = local_inputs[local_inputs_idx] if offset < num_elements else 0
            value = scan_block(value, thread_idx)
            if offset < num_elements:
                lastest_value = value + base_sum[0]
                outputs[offset] = lastest_value
            ti.simt.block.sync()
            if thread_idx == (SCAN_BLOCK_SIZE - 1):
                base_sum[0] += value
            ti.simt.block.sync()
            offset += SCAN_BLOCK_SIZE

        if block_idx == 0 and offset - SCAN_BLOCK_SIZE == ti.min(part_end, num_elements) - 1:
            first_part_state_and_prefix_sum = (
                ti.cast(SCAN_P_STATE, ti.i64) << 32) | lastest_value
            ti.atomic_max(
                state_and_prefix_sum[0], first_part_state_and_prefix_sum)

# %%

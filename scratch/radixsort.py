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
def reduce_vector_block_by_subgroup(
    value_vector,
    thread_idx,
):
    warp_reduced_value = ti.simt.block.SharedArray(
        (ti.static(value_vector.get_shape()[0]), REDUCE_BLOCK_SIZE // REDUCE_WARP_NUM), dtype=ti.i32)
    for i in ti.static(range(value_vector.get_shape()[0])):
        value_vector[i] = ti.simt.subgroup.reduce_add(value_vector[i])
    subgroup_id = thread_idx // SUBGROUP_SIZE
    ti.simt.block.sync()
    if ti.simt.subgroup.elect():
        for i in ti.static(range(value_vector.get_shape()[0])):
            warp_reduced_value[i, subgroup_id] = value_vector[i]
    ti.simt.block.sync()
    if subgroup_id == 0:
        for i in ti.static(range(value_vector.get_shape()[0])):
            value_vector[i] = ti.simt.subgroup.reduce_add(
                warp_reduced_value[i, ti.simt.subgroup.invocation_id()])
    ti.simt.block.sync()
    return value_vector
# %%


@ti.func
def reduce_vector_block_by_cuda(
    value_vector,
    thread_idx,
):
    warp_idx: ti.i32 = thread_idx >> WARP_SIZE_LOG2
    lane_idx: ti.i32 = thread_idx & (WARP_SIZE - 1)
    warp_reduced_value = ti.simt.block.SharedArray(
        (ti.static(value_vector.get_shape()[0]), REDUCE_BLOCK_SIZE // REDUCE_WARP_NUM), dtype=ti.i32)
    for i in ti.static(range(value_vector.get_shape()[0])):
        value_vector[i] = reduce_warp(value_vector[i], thread_idx)
    ti.simt.block.sync()
    if lane_idx == 0:
        # warp_reduced_value[warp_idx] = value
        for i in ti.static(range(value_vector.get_shape()[0])):
            warp_reduced_value[i, warp_idx] = value_vector[i]
    ti.simt.block.sync()
    assert REDUCE_WARP_NUM == WARP_SIZE
    if warp_idx == 0:
        # value = reduce_warp(warp_reduced_value[lane_idx], lane_idx)
        for i in ti.static(range(value_vector.get_shape()[0])):
            value_vector[i] = reduce_warp(
                warp_reduced_value[i, lane_idx], lane_idx)
    return value_vector


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


@ti.func
def reduce_vector_block(
    value_vector,
    thread_idx,
):
    arch = impl.get_runtime().prog.config().arch
    if ti.static(arch == _ti_core.cuda):
        return reduce_vector_block_by_cuda(value_vector, thread_idx)
    elif ti.static(arch == _ti_core.vulkan or arch == _ti_core.metal):
        return reduce_vector_block_by_subgroup(value_vector, thread_idx)
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


def test_reduce_block_vector():
    a = ti.Vector.field(10, dtype=ti.i32, shape=REDUCE_BLOCK_SIZE)
    b = ti.Vector.field(10, dtype=ti.i32, shape=REDUCE_BLOCK_SIZE)

    @ti.kernel
    def foo():
        ti.loop_config(block_dim=REDUCE_BLOCK_SIZE)
        for i in range(REDUCE_BLOCK_SIZE):
            thread_idx = i & (REDUCE_BLOCK_SIZE - 1)
            b[i] = reduce_vector_block_by_subgroup(a[i], thread_idx)

    for i in range(REDUCE_BLOCK_SIZE):
        for j in range(10):
            a[i][j] = i + j

    foo()

    for i in range(10):
        for j in range(REDUCE_BLOCK_SIZE):
            assert b[0][i] == REDUCE_BLOCK_SIZE * (REDUCE_BLOCK_SIZE - 1) // 2 + \
                i * REDUCE_BLOCK_SIZE


test_reduce_block()
test_reduce_block_vector()
# %%
ITEM_PER_THREAD = 8
TILE_SIZE = ITEM_PER_THREAD * REDUCE_BLOCK_SIZE
REDUCE_BLOCK_SIZE_LOG2 = int(math.log2(REDUCE_BLOCK_SIZE))


@ti.kernel
def upfront_histogram_kernel(
    num_elements: ti.i32,  # total number of elements
    high_bits: ti.template(),  # highest bit of the histogram
    digit_bits: ti.template(),  # number of bits in each digit
    keys: ti.template(),  # keys to be sorted, (num_elements, )
    # histogram, (dight_range, (high_bits + 1) // digit_bits,), (high_bits + 1) // digit_bits is the number of digit place
    histogram: ti.template(),
    # histogram for each tile, (dight_range, (high_bits + 1) // digit_bits, num_tiles, )
    tile_histogram: ti.template(),
):
    """ taichi Vulkan backend seems not working well with atomics, so we 
    implement the kernel by block reduce.
    """
    num_tiles = (num_elements + TILE_SIZE - 1) // TILE_SIZE
    num_digits = (high_bits + digit_bits - 1) // digit_bits
    dight_range = 1 << digit_bits
    assert num_tiles < TILE_SIZE
    ti.block_dim(REDUCE_BLOCK_SIZE)
    for global_idx in range(num_tiles * REDUCE_BLOCK_SIZE):
        thread_idx = global_idx & (REDUCE_BLOCK_SIZE - 1)
        tile_idx = global_idx >> REDUCE_BLOCK_SIZE_LOG2
        tile_start = tile_idx * TILE_SIZE
        tile_end = ti.min(tile_start + TILE_SIZE, num_elements)
        thread_counter = ti.Vector([0] * num_digits * dight_range, dt=ti.i32)
        for block_in_tile_idx in ti.static(range(ITEM_PER_THREAD)):
            element_idx = tile_start + block_in_tile_idx * REDUCE_BLOCK_SIZE + thread_idx
            if element_idx < tile_end:
                element = keys[element_idx]
                for digit_idx in range(num_digits):
                    digit = (element_idx >> (digit_idx * digit_bits)) & (
                        dight_range - 1)
                    thread_counter[digit_idx * dight_range + digit] += 1
        thread_counter = reduce_vector_block(thread_counter, thread_idx)
        if thread_idx == 0:
            for digit_idx in range(num_digits):
                for digit in range(dight_range):
                    tile_histogram[digit, digit_idx, tile_idx] = thread_counter[
                        digit_idx * dight_range + digit]

    ti.block_dim(REDUCE_BLOCK_SIZE)
    for global_idx in range(REDUCE_BLOCK_SIZE):
        thread_idx = global_idx & (REDUCE_BLOCK_SIZE - 1)
        tile_start = 0
        tile_end = ti.min(TILE_SIZE, num_tiles)
        thread_counter = ti.Vector([0] * num_digits * dight_range, dt=ti.i32)
        for block_in_tile_idx in range(ITEM_PER_THREAD):
            tile_idx = block_in_tile_idx * REDUCE_BLOCK_SIZE + thread_idx
            if tile_idx < tile_end:
                for digit_idx in range(num_digits):
                    for digit in range(dight_range):
                        thread_counter[digit_idx * dight_range + digit] += tile_histogram[
                            digit, digit_idx, tile_idx]

        thread_counter = reduce_vector_block(thread_counter, thread_idx)
        if thread_idx == 0:
            for digit_idx in range(num_digits):
                for digit in range(dight_range):
                    histogram[digit, digit_idx] = thread_counter[
                        digit_idx * dight_range + digit]


# %%

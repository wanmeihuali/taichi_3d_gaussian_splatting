import taichi as ti
import torch
from torch import Tensor

arch = ti.cuda
ti.init(arch=arch)

# Each row uses `block_dim_x` threads to parallel.
block_dim_x = 16  # col dim
# And `block_dim_y` threads are used to parallelize the rows.
block_dim_y = 32  # row dim
# So a block is consist of `block_dim_x * block_dim_y` threads.
block_dim = block_dim_x * block_dim_y


@ti.func
def sum(a, b):
    return a + b


@ti.func
def prod(a, b):
    return a * b


@ti.kernel
def scan_impl(
    tgt_: ti.types.ndarray(ndim=1),
    src_: ti.types.ndarray(ndim=1),
    num_rows: int,
    chunk_starts: ti.types.ndarray(ndim=1),
    chunk_cnts: ti.types.ndarray(ndim=1),
    init: float,
    binary_op: ti.template(),
    exclusive: bool,
):
    num_blocks = (num_rows + block_dim_y - 1) // block_dim_y
    ti.loop_config(block_dim=block_dim)
    for block_id, thread_id in ti.ndrange(num_blocks, block_dim):
        block_buf = ti.simt.block.SharedArray(
            (block_dim_y, 2 * block_dim_x), ti.f32
        )

        thread_id_y = thread_id // block_dim_x
        thread_id_x = thread_id % block_dim_x
        row = block_id * block_dim_y + thread_id_y

        block_total = init
        if row >= num_rows:
            continue

        row_start = chunk_starts[row]
        row_size = chunk_cnts[row]
        if row_size == 0:
            continue

        if exclusive:
            tgt_[row_start] = init

        # `range()` does not support 3 arguments in Taichi, so we use `while` instead.
        # https://github.com/taichi-dev/taichi/issues/4903
        # for block_col in range(0, row_size, 2 * block_dim_x):
        block_col = 0
        while block_col < row_size:
            col1 = block_col + thread_id_x
            col2 = block_col + block_dim_x + thread_id_x
            if row < num_rows:
                if col1 < row_size:
                    block_buf[thread_id_y, thread_id_x] = src_[row_start + col1]
                else:
                    block_buf[thread_id_y, thread_id_x] = init

                if col2 < row_size:
                    block_buf[thread_id_y, block_dim_x + thread_id_x] = src_[
                        row_start + col2
                    ]
                else:
                    block_buf[thread_id_y, block_dim_x + thread_id_x] = init

                # Add the total value of all previous blocks to the first value of this block.
                if thread_id_x == 0:
                    block_buf[thread_id_y, 0] = binary_op(
                        block_buf[thread_id_y, 0], block_total
                    )
            ti.simt.block.sync()

            # Parallel reduction (up-sweep).
            s, d = block_dim_x, 1
            while s >= 1:
                if row < num_rows and thread_id_x < s:
                    offset = (2 * thread_id_x + 1) * d - 1
                    block_buf[thread_id_y, offset + d] = binary_op(
                        block_buf[thread_id_y, offset],
                        block_buf[thread_id_y, offset + d],
                    )
                ti.simt.block.sync()
                s >>= 1
                d <<= 1

            # Down-sweep.
            s, d = 2, block_dim_x // 2
            while d >= 1:
                if row < num_rows and thread_id_x < s - 1:
                    offset = 2 * (thread_id_x + 1) * d - 1
                    block_buf[thread_id_y, offset + d] = binary_op(
                        block_buf[thread_id_y, offset],
                        block_buf[thread_id_y, offset + d],
                    )
                ti.simt.block.sync()
                s <<= 1
                d >>= 1

            # Write back to output.
            shift = 1 if exclusive else 0
            if row < num_rows:
                if col1 < (row_size - shift):
                    tgt_[row_start + col1 + shift] = block_buf[
                        thread_id_y, thread_id_x
                    ]
                if col2 < (row_size - shift):
                    tgt_[row_start + col2 + shift] = block_buf[
                        thread_id_y, block_dim_x + thread_id_x
                    ]
            block_total = block_buf[thread_id_y, 2 * block_dim_x - 1]
            ti.simt.block.sync()

            block_col += 2 * block_dim_x


def inclusive_scan_impl(
    tgt_: ti.types.ndarray(ndim=1),
    src_: ti.types.ndarray(ndim=1),
    num_rows: int,
    chunk_starts: ti.types.ndarray(ndim=1),
    chunk_cnts: ti.types.ndarray(ndim=1),
    init: float,
    binary_op: ti.template(),
):
    scan_impl(
        tgt_, src_, num_rows, chunk_starts, chunk_cnts, init, binary_op, False
    )


def exclusive_scan_impl(
    tgt_: ti.types.ndarray(ndim=1),
    src_: ti.types.ndarray(ndim=1),
    num_rows: int,
    chunk_starts: ti.types.ndarray(ndim=1),
    chunk_cnts: ti.types.ndarray(ndim=1),
    init: float,
    binary_op: ti.template(),
):
    scan_impl(
        tgt_, src_, num_rows, chunk_starts, chunk_cnts, init, binary_op, True
    )


def inclusive_sum_fwd(
    chunk_starts: Tensor, chunk_cnts: Tensor, inputs: Tensor
) -> Tensor:
    assert chunk_starts.dim() == chunk_starts.dim() == inputs.dim() == 1
    assert chunk_starts.shape == chunk_cnts.shape
    n_rays = chunk_starts.numel()
    n_edges = inputs.size(0)
    outputs = torch.empty_like(inputs)
    if n_edges > 0:
        inclusive_scan_impl(
            outputs, inputs, n_rays, chunk_starts, chunk_cnts, 0.0, sum
        )
    return outputs


def inclusive_sum_bwd(
    chunk_starts: Tensor, chunk_cnts: Tensor, inputs: Tensor
) -> Tensor:
    assert chunk_starts.dim() == chunk_starts.dim() == inputs.dim() == 1
    assert chunk_starts.shape == chunk_cnts.shape
    n_rays = chunk_starts.numel()
    n_edges = inputs.size(0)
    outputs = torch.empty_like(inputs)
    if n_edges > 0:
        chunk_starts = n_edges - (chunk_starts + chunk_cnts)
        inputs = inputs.flip(0).contiguous()
        chunk_starts = chunk_starts.flip(0).contiguous()
        chunk_cnts = chunk_cnts.flip(0).contiguous()
        inclusive_scan_impl(
            outputs, inputs, n_rays, chunk_starts, chunk_cnts, 0.0, sum
        )
        outputs = outputs.flip(0).contiguous()
    return outputs


def exclusive_sum_fwd(
    chunk_starts: Tensor, chunk_cnts: Tensor, inputs: Tensor
) -> Tensor:
    assert chunk_starts.dim() == chunk_starts.dim() == inputs.dim() == 1
    assert chunk_starts.shape == chunk_cnts.shape
    n_rays = chunk_starts.numel()
    n_edges = inputs.size(0)
    outputs = torch.empty_like(inputs)
    if n_edges > 0:
        exclusive_scan_impl(
            outputs, inputs, n_rays, chunk_starts, chunk_cnts, 0.0, sum
        )
    return outputs


def exclusive_sum_bwd(
    chunk_starts: Tensor, chunk_cnts: Tensor, inputs: Tensor
) -> Tensor:
    assert chunk_starts.dim() == chunk_starts.dim() == inputs.dim() == 1
    assert chunk_starts.shape == chunk_cnts.shape
    n_rays = chunk_starts.numel()
    n_edges = inputs.size(0)
    outputs = torch.empty_like(inputs)
    if n_edges > 0:
        chunk_starts = n_edges - (chunk_starts + chunk_cnts)
        chunk_cnts = chunk_cnts
        exclusive_scan_impl(
            outputs,
            inputs.flip(0).contiguous(),
            n_rays,
            chunk_starts.flip(0).contiguous(),
            chunk_cnts.flip(0).contiguous(),
            0.0,
            sum,
        )
        outputs = outputs.flip(0).contiguous()
    return outputs


def inclusive_prod_fwd(
    chunk_starts: Tensor, chunk_cnts: Tensor, inputs: Tensor
) -> Tensor:
    assert chunk_starts.dim() == chunk_starts.dim() == inputs.dim() == 1
    assert chunk_starts.shape == chunk_cnts.shape
    n_rays = chunk_starts.numel()
    n_edges = inputs.size(0)
    outputs = torch.empty_like(inputs)
    if n_edges > 0:
        inclusive_scan_impl(
            outputs, inputs, n_rays, chunk_starts, chunk_cnts, 1.0, prod
        )
    return outputs


def inclusive_prod_bwd(
    chunk_starts: Tensor,
    chunk_cnts: Tensor,
    inputs: Tensor,
    outputs: Tensor,
    grad_outputs: Tensor,
) -> Tensor:
    assert chunk_starts.dim() == chunk_starts.dim() == inputs.dim() == 1
    assert chunk_starts.shape == chunk_cnts.shape
    n_rays = chunk_starts.numel()
    n_edges = inputs.size(0)
    grad_inputs = torch.empty_like(grad_outputs)
    if n_edges > 0:
        chunk_starts = n_edges - (chunk_starts + chunk_cnts)
        inclusive_scan_impl(
            grad_inputs,
            (grad_outputs * outputs).flip(0).contiguous(),
            n_rays,
            chunk_starts.flip(0).contiguous(),
            chunk_cnts.flip(0).contiguous(),
            0.0,
            sum,
        )
        grad_inputs = grad_inputs.flip(0).contiguous()
        # FIXME: the grad is not correct when inputs are zero!!
        grad_inputs = grad_inputs / inputs.clamp_min(1e-10)

    return grad_inputs


def exclusive_prod_fwd(
    chunk_starts: Tensor, chunk_cnts: Tensor, inputs: Tensor
) -> Tensor:
    assert chunk_starts.dim() == chunk_starts.dim() == inputs.dim() == 1
    assert chunk_starts.shape == chunk_cnts.shape
    n_rays = chunk_starts.numel()
    n_edges = inputs.size(0)
    outputs = torch.empty_like(inputs)
    if n_edges > 0:
        exclusive_scan_impl(
            outputs, inputs, n_rays, chunk_starts, chunk_cnts, 1.0, prod
        )
    return outputs


def exclusive_prod_bwd(
    chunk_starts: Tensor,
    chunk_cnts: Tensor,
    inputs: Tensor,
    outputs: Tensor,
    grad_outputs: Tensor,
) -> Tensor:
    assert chunk_starts.dim() == chunk_starts.dim() == inputs.dim() == 1
    assert chunk_starts.shape == chunk_cnts.shape
    n_rays = chunk_starts.numel()
    n_edges = inputs.size(0)
    grad_inputs = torch.empty_like(grad_outputs)
    if n_edges > 0:
        chunk_starts = n_edges - (chunk_starts + chunk_cnts)
        exclusive_scan_impl(
            grad_inputs,
            (grad_outputs * outputs).flip(0).contiguous(),
            n_rays,
            chunk_starts.flip(0).contiguous(),
            chunk_cnts.flip(0).contiguous(),
            0.0,
            sum,
        )
        grad_inputs = grad_inputs.flip(0).contiguous()
        # FIXME: the grad is not correct when inputs are zero!!
        grad_inputs = grad_inputs / inputs.clamp_min(1e-10)

    return grad_inputs


if __name__ == "__main__":
    import nerfacc.cuda as _C
    import torch
    import tqdm

    torch.manual_seed(42)
    device = "cuda"
    N = 128
    S = 10000
    # create data
    inputs = torch.rand((N * S), device=device, requires_grad=True)
    chunk_starts = torch.arange(0, N * S, S, device=device, dtype=torch.long)
    chunk_cnts = torch.full((N,), S, dtype=torch.long, device=device)

    # inclusive sum
    outputs = inclusive_sum_fwd(chunk_starts, chunk_cnts, inputs)
    _outputs = _C.inclusive_sum(chunk_starts, chunk_cnts, inputs, False, False)
    assert torch.allclose(outputs, _outputs)
    outputs = inclusive_sum_bwd(chunk_starts, chunk_cnts, inputs)
    _outputs = _C.inclusive_sum(chunk_starts, chunk_cnts, inputs, False, True)
    assert torch.allclose(outputs, _outputs)

    # exclusive sum
    outputs = exclusive_sum_fwd(chunk_starts, chunk_cnts, inputs)
    _outputs = _C.exclusive_sum(chunk_starts, chunk_cnts, inputs, False, False)
    assert torch.allclose(outputs, _outputs)
    outputs = exclusive_sum_bwd(chunk_starts, chunk_cnts, inputs)
    _outputs = _C.exclusive_sum(chunk_starts, chunk_cnts, inputs, False, True)
    assert torch.allclose(outputs, _outputs)

    # inclusive prod
    outputs = inclusive_prod_fwd(chunk_starts, chunk_cnts, inputs)
    _outputs = _C.inclusive_prod_forward(chunk_starts, chunk_cnts, inputs)
    assert torch.allclose(outputs, _outputs)
    grad_outputs = torch.rand_like(outputs)
    grad_inputs = inclusive_prod_bwd(
        chunk_starts, chunk_cnts, inputs, outputs, grad_outputs
    )
    _grad_inputs = _C.inclusive_prod_backward(
        chunk_starts, chunk_cnts, inputs, outputs, grad_outputs
    )
    assert torch.allclose(grad_inputs, _grad_inputs)

    # exclusive prod
    outputs = exclusive_prod_fwd(chunk_starts, chunk_cnts, inputs)
    _outputs = _C.exclusive_prod_forward(chunk_starts, chunk_cnts, inputs)
    assert torch.allclose(outputs, _outputs)
    grad_outputs = torch.rand_like(outputs)
    grad_inputs = exclusive_prod_bwd(
        chunk_starts, chunk_cnts, inputs, outputs, grad_outputs
    )
    _grad_inputs = _C.exclusive_prod_backward(
        chunk_starts, chunk_cnts, inputs, outputs, grad_outputs
    )
    assert torch.allclose(grad_inputs, _grad_inputs)

    # profiling
    # taichi: 1242 it/s
    ti.sync()
    torch.cuda.synchronize()
    for _ in tqdm.trange(1000):
        outputs = inclusive_sum_fwd(chunk_starts, chunk_cnts, inputs)
        ti.sync()
        torch.cuda.synchronize()
    # nerfacc: 2724 it/s
    ti.sync()
    torch.cuda.synchronize()
    for _ in tqdm.trange(1000):
        _outputs = _C.inclusive_sum(
            chunk_starts, chunk_cnts, inputs, False, False
        )
        ti.sync()
        torch.cuda.synchronize()

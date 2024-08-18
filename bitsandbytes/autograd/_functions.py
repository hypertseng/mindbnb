from dataclasses import dataclass
from functools import reduce  # Required in Python 3
import operator
from typing import Callable, Optional, Tuple
import warnings
from warnings import warn
import mindspore
from mindspore import ops, Tensor, nn, context
import subprocess
import numpy as np

import bitsandbytes.functional as F


# math.prod not compatible with python < 3.8
def prod(iterable):
    return reduce(operator.mul, iterable, 1)


def clone(tensor):
    return tensor.copy()


# The inverse transformation for the colTuring and colAmpere format were contributed by Alex Borzunov:
# https://github.com/bigscience-workshop/petals/blob/main/src/petals/utils/linear8bitlt_patch.py


"""
    This class pools outlier dimensions across layers.
    This is particularly important for small models where outlier features
    are less systematic and occur with low frequency.
"""


class GlobalOutlierPooler:
    _instance = None

    def __init__(self):
        raise RuntimeError("Call get_instance() instead")

    def initialize(self):
        self.outliers = set()
        self.model_dim = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            cls._instance.initialize()
        return cls._instance

    def add_outliers(self, outlier_idx, feature_dim):
        if self.model_dim is None:
            self.model_dim = feature_dim
        if feature_dim != self.model_dim:
            return  # we do not encode outliers for the 2nd FFN layer

        self.outliers.update(outlier_idx.tolist())

    def get_current_outlier_idx(self):
        return mindspore.Tensor(list(self.outliers)).to(mindspore.int64)


def get_inverse_transform_indices(
    transform_tile: Callable[[mindspore.Tensor], mindspore.Tensor],
    tile_size: Tuple[int, int],
):
    """
    Compute a permutation of indices that invert the specified (tiled) matrix transformation

    :param transform_tile: a function that applies forward transform to a tensor of shape [dim1, dim2]
    :param tile_size: higher-level tile dimensions, i.e. (8, 32) for Turing and (32, 32) for Ampere
    :note: we assume that tile_transform applies to a cpu-based int8 tensor of shape tile_size
    :example: transform_tile function for the turing layout (bitsandbytes.functional as F)
    :returns: indices
    """
    d1, d2 = tile_size
    assert 0 < d1 * d2 < 2**64
    tile_indices = ops.arange(d1 * d2, dtype=mindspore.int64).view(d1, d2)
    # encode each position in tile as a tuple of <= 8 unique bytes
    permuted_tile_indices = ops.zeros_like(tile_indices)
    for i in range(8):
        # select i-th byte, apply transformation and trace where each index ended up
        ith_dim_indices = ops.div(tile_indices, 256**i, rounding_mode="trunc") % 256
        sample_tile_i = (ith_dim_indices - 128).to(mindspore.int8).contiguous()
        assert ops.all(sample_tile_i.int() + 128 == ith_dim_indices), "int overflow"
        permuted_tile_i = transform_tile(sample_tile_i)
        ith_permuted_indices = permuted_tile_i.to(tile_indices.dtype) + 128
        permuted_tile_indices += ith_permuted_indices * (256**i)
        if d1 * d2 < 256**i:
            break  # if all indices fit in i bytes, stop early
    return permuted_tile_indices


def undo_layout(permuted_tensor: mindspore.Tensor, tile_indices: mindspore.Tensor) -> mindspore.Tensor:
    """
    Undo a tiled permutation such as turing or ampere layout

    :param permuted_tensor: mindspore tensor in a permuted layout
    :param tile_indices: reverse transformation indices, from get_inverse_transform_indices
    :return: contiguous row-major tensor
    """
    (rows, cols), (tile_rows, tile_cols) = permuted_tensor.shape, tile_indices.shape
    assert rows % tile_rows == cols % tile_cols == 0, "tensor must contain a whole number of tiles"
    tensor = permuted_tensor.reshape(-1, tile_indices.numel()).t()
    outputs = Tensor(
        shape=tensor.shape, dtype=tensor.dtype
    )  # note: not using .index_copy because it was slower on cuda
    outputs[tile_indices.flatten()] = tensor
    outputs = outputs.reshape(tile_rows, tile_cols, cols // tile_cols, rows // tile_rows)
    outputs = outputs.permute(3, 0, 2, 1)  # (rows // tile_rows, tile_rows), (cols // tile_cols, tile_cols)
    return outputs.reshape(rows, cols).contiguous()


class MatMul8bit():
    @staticmethod
    def forward(ctx, A, B, out=None, quant_type="vector", precision=None):
        if precision is None:
            precision = [8, 8, 8]
        if precision[0] != 8:
            output = ops.matmul(A, B)
        else:
            if len(B.shape) == 2:
                dim = 0
            else:
                dim = 1
            qA, SA = F.vectorwise_quant(A, dim=-1, quant_type=quant_type)
            qB, SB = F.vectorwise_quant(B, dim=dim, quant_type=quant_type)
            iout = F.igemm(qA, qB)
            output = F.vectorwise_mm_dequant(iout, SA, SB, A.dtype, quant_type)

        if A.requires_grad or B.requires_grad:
            ctx.save_for_backward(A, B)

        ctx.quant_type = quant_type
        ctx.precision = precision

        return output

    # @staticmethod
    # def backward(ctx, grad_output):
    #     A, B = ctx.saved_tensors
    #     quant_type = ctx.quant_type
    #     precision = ctx.precision
    #     grad_A = grad_B = None

    #     if B.requires_grad:
    #         if len(A.shape) == 3:
    #             dims = [0, 1]
    #             # bsi -> ibs
    #             permute_dim = [0, 2, 1]
    #         else:
    #             dims = [0]
    #             # bs -> sb
    #             permute_dim = [1, 0]

    #         if precision[1] != 8:
    #             grad_B = ops.matmul(A.permute(permute_dim), grad_output)
    #         else:
    #             if len(B.shape) == 2 and len(A.shape) == 3:
    #                 grad_output = grad_output.contiguous()
    #                 if not grad_output.is_contiguous():
    #                     grad_output.contiguous()
    #                 qgrad_output, S1 = F.vectorwise_quant(
    #                     grad_output.view(-1, grad_output.shape[2]),
    #                     dim=0,
    #                     quant_type=quant_type,
    #                 )
    #                 if not A.is_contiguous():
    #                     A = A.contiguous()
    #                 qA, S2 = F.vectorwise_quant(A.view(-1, A.shape[2]), dim=0, quant_type=quant_type)
    #                 igrad_B = F.igemm(qA.t(), qgrad_output)
    #                 grad_B = F.vectorwise_mm_dequant(igrad_B, S2.t(), S1, grad_output.dtype, quant_type)
    #             else:
    #                 qgrad_output, S1 = F.vectorwise_quant(grad_output, dim=dims, quant_type=quant_type)
    #                 qA, S2 = F.vectorwise_quant(A, dim=dims, quant_type=quant_type)
    #                 igrad_B = F.igemm(qA.permute(permute_dim), qgrad_output)
    #                 grad_B = F.vectorwise_mm_dequant(
    #                     igrad_B,
    #                     S2.permute(permute_dim),
    #                     S1,
    #                     grad_output.dtype,
    #                     quant_type,
    #                 )

    #     if A.requires_grad:
    #         if len(grad_output.shape) == 3:
    #             dims = [2]
    #         else:
    #             dims = [1]

    #         if len(B.shape) == 3:
    #             # bio -> boi
    #             permute_dim = [0, 2, 1]
    #             dim_B = dims
    #         else:
    #             # io -> oi
    #             permute_dim = [1, 0]
    #             dim_B = [1]

    #         if precision[2] != 8:
    #             grad_A = ops.matmul(grad_output, B.permute(permute_dim))
    #         else:
    #             qgrad_output, S1 = F.vectorwise_quant(grad_output, dim=dims, quant_type=quant_type)
    #             qB, S3 = F.vectorwise_quant(B, dim=dim_B, quant_type=quant_type)
    #             igrad_A = F.igemm(qgrad_output, qB.permute(permute_dim))
    #             grad_A = F.vectorwise_mm_dequant(
    #                 igrad_A,
    #                 S1,
    #                 S3.permute(permute_dim),
    #                 grad_output.dtype,
    #                 quant_type,
    #             )

    #     return grad_A, grad_B, None, None, None


# mm_cublas = MatMul8bit.apply
# bmm_cublas = MatMul8bit.apply
# matmul_cublas = MatMul8bit.apply


import mindspore.context as context


def supports_igemmlt() -> bool:
    """检查当前设备是否支持优化的 int8 内核"""
    if F.get_device_capability() < (7, 5):
        return False

    device_name = F.get_device_name()
    nvidia16_models = (
        "GTX 1630",
        "GTX 1650",
        "GTX 1660",
    )  # https://en.wikipedia.org/wiki/GeForce_16_series
    if any(model_name in device_name for model_name in nvidia16_models):
        return False  # 这些设备在技术上是 cuda 7.5 兼容的，但缺少张量核心

    return True


# # Example usage:
# device_index = "0"  # 假设我们只检查第一个 GPU
# print(supports_igemmlt(device_index))
# def supports_igemmlt(device_id: int = 0) -> bool:
#     """检查设备是否支持优化的 int8 内核"""
#     if get_device_capability(device_id) < (7, 5):
#         return False
#     device_name = get_device_name(device_id)
#     nvidia16_models = (
#         "GTX 1630",
#         "GTX 1650",
#         "GTX 1660",
#     )  # https://en.wikipedia.org/wiki/GeForce_16_series
#     if any(model_name in device_name for model_name in nvidia16_models):
#         return False  # 这些设备在技术上是 cuda 7.5 兼容的，但缺少张量核心
#     return True


def _get_tile_size(format):
    assert format in (
        "col_turing",
        "col_ampere",
    ), f"please find this assert and manually enter tile size for {format}"
    return (8, 32) if format == "col_turing" else (32, 32)


def get_tile_inds(format, device):
    transform = lambda x: F.transform(x, from_order="row", to_order=format)[0].to(x.device)
    return get_inverse_transform_indices(transform, _get_tile_size(format))


@dataclass
class MatmulLtState():
    
    _tile_indices: Optional[mindspore.Tensor] = None
    force_no_igemmlt: bool = False
    CB = None
    CxB = None
    SB = None
    SCB = None

    CxBt = None
    SBt = None
    CBt = None

    subB = None

    outlier_pool = None
    has_accumulated_gradients = False
    threshold = 0.0
    idx = None
    is_training = True
    has_fp16_weights = True
    memory_efficient_backward = False
    use_pool = False
    formatB = F.get_special_format_str()

    def reset_grads(self,):
        self.CB = None
        self.CxB = None
        self.SB = None
        self.SCB = None

        self.CxBt = None
        self.SBt = None
        self.CBt = None

    @property
    def tile_indices(self):
        if self._tile_indices is None:
            self._tile_indices = get_tile_inds(self.formatB, self.CxB.device)
        return self._tile_indices


class MatMul8bitLt(nn.Cell):
    # forward is the same, but we added the fallback for pre-turing GPUs
    # backward is mostly the same, but adds one extra clause (see "elif state.CxB is not None")
    def __init__(self,):
        super().__init__()
        self.needs_input_grad = [False, False, False, False, False]

    def construct(self, A, B, out=None, bias=None, state=MatmulLtState):
        using_igemmlt = supports_igemmlt() and not state.force_no_igemmlt
        # default of pymindspore behavior if inputs are empty
        self.is_empty = False
        if prod(A.shape) == 0:
            self.is_empty = True
            self.A = A
            self.B = B
            self.bias = bias
            if A.shape[-1] == B.shape[0]:
                return Tensor(np.empty(A.shape[:-1] + B.shape[1:],))
            else:
                return Tensor(np.empty(A.shape[:-1] + B.shape[:1],))

        # 1. Quantize A
        # 2. Quantize B
        # 3. Matmul
        # 4. Mixed-precision decomposition matmul
        # 5. Save state
        formatB = state.formatB
        input_shape = A.shape
        if state.outlier_pool is None:
            state.outlier_pool = GlobalOutlierPooler.get_instance()

        # Cast A to fp16
        if A.dtype != mindspore.float16:
            warnings.warn(f"MatMul8bitLt: inputs will be cast from {A.dtype} to float16 during quantization")

        # 1. Quantize A
        if len(A.shape) == 3:
            A = A.reshape(-1, A.shape[-1])
        CA, CAt, SCA, SCAt, coo_tensorA = F.double_quant(A.astype(mindspore.float16), threshold=state.threshold)

        if state.threshold > 0.0 and coo_tensorA is not None:
            if state.has_fp16_weights:
                _, idx = ops.unique(coo_tensorA.colidx)
                idx.astype(mindspore.int64)
                CA[:, idx] = 0
                CAt[:, idx] = 0
                subA = A[:, idx]
                state.subB = B[:, idx].t()
                state.idx = idx
            else:
                if state.CxB is None and using_igemmlt:
                    # B in in 8-bit row-major, we can transform it back to 16-bit to extract outlier dimensions
                    # we also need to convert it to the turing/ampere format
                    state.CxB, state.SB = F.transform(state.CB, to_order=formatB)
        else:
            if not state.has_fp16_weights and state.CxB is None and using_igemmlt:
                state.CxB, state.SB = F.transform(state.CB, to_order=formatB)
            subA = None

        # 2. Quantize B
        if state.has_fp16_weights:
            has_grad = True if (getattr(B, "grad", None) is not None) else False
            if (state.is_training and not has_grad) or state.CxB is None:
                state.reset_grads()
                (
                    CB,
                    state.CBt,
                    state.SCB,
                    state.SCBt,
                    coo_tensorB,
                ) = F.double_quant(B.to(mindspore.float16))
                if using_igemmlt:
                    state.CxB, state.SB = F.transform(CB, to_order=formatB)
                else:
                    state.CB = CB
        else:
            has_grad = False

        if coo_tensorA is not None and not state.has_fp16_weights:
            # extract outliers

            outlier_idx = ops.unique(coo_tensorA.colidx)
            state.idx = outlier_idx
            # state.outlier_pool.add_outliers(outlier_idx, A.shape[-1])
            # if state.use_pool and state.outlier_pool.model_dim == A.shape[-1]:
            #    # do not use pool for 2nd FFN layer
            #    state.idx = state.outlier_pool.get_current_outlier_idx().to(A.device)
            # else:
            #    state.idx = outlier_idx
            if state.CxB is not None:
                outliers = F.extract_outliers(state.CxB, state.SB, state.idx.int())
            else:
                outliers = state.CB[:, state.idx.long()].clone()

            state.subB = (outliers * state.SCB.view(-1, 1) / 127.0).t().to(A.dtype)
            CA[:, state.idx.long()] = 0
            CAt[:, state.idx.long()] = 0
            subA = A[:, state.idx.long()]

        shapeB = state.SB[0] if state.SB else B.shape

        if len(input_shape) == 3:
            output_shape = (input_shape[0], input_shape[1], shapeB[0])
        else:
            output_shape = (input_shape[0], shapeB[0])

        # 3. Matmul
        if using_igemmlt:
            C32A, SA = F.transform(CA, "col32")
            out32, Sout32 = F.igemmlt(C32A, state.CxB, SA, state.SB)
            if bias is None or bias.dtype == mindspore.float16:
                # we apply the fused bias here
                output = F.mm_dequant(out32, Sout32, SCA, state.SCB, bias=bias)
                output = output.to(A.dtype)
            else:  # apply bias separately
                output = F.mm_dequant(out32, Sout32, SCA, state.SCB, bias=None)
                output = output.to(A.dtype).add_(bias)

        else:
            A_wo_outliers = A.copy()
            if state.idx is not None:
                A_wo_outliers[:, state.idx.long()] = 0
            output = ops.dense(A_wo_outliers, state.CB.to(A.dtype))
            scb = state.SCB.unsqueeze(0)
            scb = scb * (1.0 / 127.0)
            output = output * scb
            if bias is not None:
                output = output + bias

        # 4. Mixed-precision decomposition matmul
        if coo_tensorA is not None and subA is not None:
            output += ops.matmul(subA, state.subB)

        # 5. Save state
        self.state = state

        self.formatB = formatB
        self.grad_shape = input_shape
        self.dtype_A, self.dtype_B, self.dtype_bias = A.dtype, B.dtype, None if bias is None else bias.dtype

        if any(self.needs_input_grad[:2]):
            self.tensors = (CAt, subA, A)
            self.tensor_states = (SCAt, state.idx)
        else:
            self.tensors = [None, None, A]
            self.tensor_states = (None, None)

        clone_func = clone if len(output_shape) == 3 else lambda x: x
        return clone_func(output.view(output_shape))

    # @staticmethod
    # def backward(ctx, grad_output):
    #     if ctx.is_empty:
    #         bias_grad = None if ctx.bias is None else torch.zeros_like(ctx.bias)
    #         return torch.zeros_like(ctx.A), torch.zeros_like(ctx.B), None, bias_grad, None
    #     req_gradA, req_gradB, _, req_gradBias, _ = ctx.needs_input_grad
    #     CAt, subA, A = ctx.tensors
    #     SCAt, idx = ctx.tensor_states
    #     formatB = ctx.formatB
    #     state = ctx.state
    #     grad_A = grad_B = grad_bias = None

    #     if req_gradBias:
    #         # compute grad_bias first before changing grad_output dtype
    #         grad_bias = grad_output.sum(0, dtype=ctx.dtype_bias)

    #     # Cast grad_output to fp16
    #     if len(grad_output.shape) == 3:
    #         grad_output = grad_output.reshape(-1, grad_output.shape[-1]).contiguous()

    #     Cgrad, Cgradt, SCgrad, SCgradt, coo_tensor = F.double_quant(grad_output.to(torch.float16))
    #     if req_gradB:
    #         CxAt, SAt = F.transform(CAt, formatB, transpose=True)
    #         C32grad, Sgrad = F.transform(Cgradt, "col32", transpose=True)
    #         gradB32, SgradB32 = F.igemmlt(C32grad, CxAt, Sgrad, SAt)
    #         grad_B = F.mm_dequant(gradB32, SgradB32, SCgradt, SCAt)
    #         if state.threshold > 0.0 and subA is not None:
    #             grad_B[:, idx] += torch.matmul(grad_output.t(), subA)

    #     if req_gradA:
    #         if state.CBt is not None:
    #             C32grad, Sgrad = F.transform(Cgrad, "col32")
    #             if state.CxBt is None:
    #                 state.CxBt, state.SBt = F.transform(state.CBt, to_order=formatB, transpose=True)
    #             gradA32, SgradA32 = F.igemmlt(C32grad, state.CxBt, Sgrad, state.SBt)
    #             grad_A = F.mm_dequant(gradA32, SgradA32, SCgrad, state.SCBt).view(ctx.grad_shape).to(ctx.dtype_A)

    #         elif state.CB is not None:
    #             CB = state.CB.to(ctx.dtype_A, copy=True).mul_(state.SCB.unsqueeze(1).mul(1.0 / 127.0))
    #             grad_A = torch.matmul(grad_output, CB).view(ctx.grad_shape).to(ctx.dtype_A)
    #         elif state.CxB is not None:
    #             CB = (
    #                 undo_layout(state.CxB, state.tile_indices)
    #                 .to(ctx.dtype_A)
    #                 .mul_(state.SCB.unsqueeze(1).mul(1.0 / 127.0))
    #             )
    #             grad_A = torch.matmul(grad_output, CB).view(ctx.grad_shape).to(ctx.dtype_A)
    #         else:
    #             raise Exception("State must contain either CBt or CB or CxB matrix for backward")

    #     return grad_A, grad_B, None, grad_bias, None


class MatMul4Bit(nn.Cell):
    # forward is the same, but we added the fallback for pre-turing GPUs
    # backward is mostly the same, but adds one extra clause (see "elif state.CxB is not None")

    @staticmethod
    def forward(ctx, A, B, out=None, bias=None, quant_state: Optional[F.QuantState] = None):
        # default of pytorch behavior if inputs are empty
        ctx.is_empty = False
        if prod(A.shape) == 0:
            ctx.is_empty = True
            ctx.A = A
            ctx.B = B
            ctx.bias = bias
            B_shape = quant_state.shape
            if A.shape[-1] == B_shape[0]:
                return Tensor(np.empty(A.shape[:-1] + B_shape[1:]), dtype=A.dtype,)
            else:
                return Tensor(np.empty(A.shape[:-1] + B_shape[:1]), dtype=A.dtype,)

        # 1. Dequantize
        # 2. MatmulnN
        output = ops.dense(A, F.dequantize_4bit(B, quant_state).to(A.dtype).t(), bias)

        # 3. Save state
        ctx.state = quant_state
        ctx.dtype_A, ctx.dtype_B, ctx.dtype_bias = A.dtype, B.dtype, None if bias is None else bias.dtype

        if any(ctx.needs_input_grad[:2]):
            ctx.tensors = (None, B)
        else:
            ctx.tensors = (None, None)

        return output

    # @staticmethod
    # def backward(ctx, grad_output):
    #     if ctx.is_empty:
    #         bias_grad = None if ctx.bias is None else ops.zeros_like(ctx.bias)
    #         return ops.zeros_like(ctx.A), ops.zeros_like(ctx.B), None, bias_grad, None

    #     req_gradA, _, _, req_gradBias, _ = ctx.needs_input_grad
    #     _, B = ctx.tensors

    #     grad_A, grad_B, grad_bias = None, None, None

    #     if req_gradBias:
    #         # compute grad_bias first before changing grad_output dtype
    #         grad_bias = grad_output.sum(0, dtype=ctx.dtype_bias)

    #     # not supported by PyTorch. TODO: create work-around
    #     # if req_gradB: grad_B = torch.matmul(grad_output.t(), A)
    #     if req_gradA:
    #         grad_A = ops.matmul(grad_output, F.dequantize_4bit(B, ctx.state).to(grad_output.dtype).t())

    #     return grad_A, grad_B, None, grad_bias, None


def matmul(
    A: mindspore.Tensor,
    B: mindspore.Tensor,
    out: Optional[mindspore.Tensor] = None,
    state: Optional[MatmulLtState] = None,
    threshold=0.0,
    bias=None,
):
    state = state or MatmulLtState()
    if threshold > 0.0:
        state.threshold = threshold
    # return MatMul8bitLt(A, B, out, bias, state)
    x = MatMul8bitLt()
    return x(A, B, out, bias, state)


def matmul_4bit(
    A: mindspore.Tensor,
    B: mindspore.Tensor,
    quant_state: F.QuantState,
    out: Optional[mindspore.Tensor] = None,
    bias=None,
):
    assert quant_state is not None
    if A.numel() == A.shape[-1] and A.requires_grad == False:
        if A.shape[-1] % quant_state.blocksize != 0:
            warn(
                f"Some matrices hidden dimension is not a multiple of {quant_state.blocksize} and efficient inference kernels are not supported for these (slow). Matrix input size found: {A.shape}",
            )
            return MatMul4Bit.apply(A, B, out, bias, quant_state)
        else:
            out = F.gemv_4bit(A, B.t(), out, state=quant_state)
            if bias is not None:
                out += bias
            return out
    else:
        return MatMul4Bit.apply(A, B, out, bias, quant_state)

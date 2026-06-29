"""
Copyright (c) 2023 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import contextlib
import math
import random
import time
from functools import partial
from typing import Tuple, Any, List, Optional, Callable

import os
import sys
import warnings

import numpy as np
import torch
from einops import rearrange, reduce, repeat

from flashinfer.utils import round_up

from . import statistics
from .statistics import BenchmarkStatistics


# =============================================================================
# Rotating Buffer Utilities for Cold-L2 Benchmarking
# =============================================================================


def get_l2_cache_size(device=None) -> int:
    """
    Get L2 cache size in bytes for the given CUDA device.

    Args:
        device: CUDA device (int, torch.device, or None for current device).

    Returns:
        L2 cache size in bytes.
    """
    if device is None:
        device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    return props.L2_cache_size


def _calculate_tensor_bytes(tensors: List[torch.Tensor]) -> int:
    """
    Calculate total bytes of tensors residing on GPU.
    Assumes all tensors are on the same device.

    Args:
        tensors: List of torch.Tensor objects.

    Returns:
        Total bytes occupied by GPU tensors (CPU tensors are ignored).
    """
    total = 0
    for t in tensors:
        if isinstance(t, torch.Tensor) and t.is_cuda:
            total += t.numel() * t.element_size()
    return total


def _extract_gpu_tensors(obj) -> List[torch.Tensor]:
    """
    Recursively extract all GPU-resident tensors from a nested structure
    of lists, tuples, and dicts.

    Args:
        obj: Object to extract tensors from (can be tensor, list, tuple, dict, or other).

    Returns:
        Flat list of tensors on GPU found in the structure.
    """
    tensors = []
    if isinstance(obj, torch.Tensor) and obj.is_cuda:
        tensors.append(obj)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            tensors.extend(_extract_gpu_tensors(item))
    elif isinstance(obj, dict):
        for v in obj.values():
            tensors.extend(_extract_gpu_tensors(v))
    return tensors


def calculate_rotation_count(
    tensors: List[torch.Tensor], device=None, min_rotations: int = 2
) -> int:
    """
    Calculate the number of buffer copies needed to ensure cold L2 cache.

    The function uses conservative thresholds to account for:
    - LRU eviction being gradual (not all data evicted when capacity exceeded)
    - Cache associativity effects (some data may persist in non-conflicting sets)
    - Hardware prefetching behavior

    Returns 1 (no rotation needed) only when tensor size substantially exceeds
    L2 cache (>= 5x), ensuring cache effects are truly negligible.

    Args:
        tensors: List of tensors to consider for rotation (must be on GPU).
        device: Device for L2 cache query (None for current device).
        min_rotations: Minimum number of rotations when rotation is needed.

    Returns:
        Number of buffer copies needed (1 means no rotation needed).
    """
    l2_size = get_l2_cache_size(device)
    total_bytes = _calculate_tensor_bytes(tensors)

    if total_bytes == 0:
        return 1  # No tensors to rotate

    # Use aggressive threshold: only skip rotation if tensors far exceed L2 (5x)
    # This ensures cache effects are truly negligible even with prefetching
    safe_cache_threshold = l2_size * 5
    if total_bytes >= safe_cache_threshold:
        return 1  # Tensors far exceed L2, no rotation needed

    # Conservative formula: ensure between any two uses of the same buffer,
    # we've accessed enough data to fully flush L2 with margin
    # Using safe_cache_threshold ensures we account for all cache effects
    num_rotations = math.ceil(safe_cache_threshold / total_bytes) + 1

    return max(min_rotations, num_rotations)


def _clone_structure(obj):
    """
    Deep clone a nested structure, cloning GPU tensors with detach().clone()
    while preserving scalars, booleans, and other non-tensor values.

    For non-contiguous tensors (e.g., created with as_strided), this function
    preserves the stride pattern using torch.empty_strided() + copy_(). This is
    important for backends like cuDNN that expect specific memory layouts.

    Args:
        obj: Object to clone (tensor, list, tuple, dict, or other).

    Returns:
        Cloned structure with GPU tensors cloned, other values preserved.
    """
    if isinstance(obj, torch.Tensor):
        if obj.is_cuda:
            if obj.is_contiguous():
                return obj.detach().clone()
            else:
                # Preserve stride pattern for non-contiguous tensors
                # (e.g., as_strided views used by cuDNN paged attention)
                result = torch.empty_strided(
                    obj.size(),
                    obj.stride(),
                    dtype=obj.dtype,
                    device=obj.device,
                )
                result.copy_(obj.detach())
                return result
        else:
            return obj  # CPU tensors returned as-is
    elif isinstance(obj, list):
        return [_clone_structure(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(_clone_structure(item) for item in obj)
    elif isinstance(obj, dict):
        return {k: _clone_structure(v) for k, v in obj.items()}
    else:
        # Non-tensor, non-container: return as-is (e.g., int, float, str, bool, None)
        return obj


def _create_rotated_buffer_copies(
    input_args: Tuple, input_kwargs: dict, num_rotations: int
) -> List[Tuple[Tuple, dict]]:
    """
    Create multiple copies of input_args and input_kwargs for buffer rotation.

    The first copy (index 0) uses the original args/kwargs.
    Subsequent copies clone all GPU tensors while preserving other values.

    Args:
        input_args: Positional arguments tuple.
        input_kwargs: Keyword arguments dict.
        num_rotations: Number of buffer copies to create.

    Returns:
        List of (args, kwargs) tuples, one for each rotation index.
    """
    if num_rotations <= 1:
        return [(input_args, input_kwargs)]

    copies = []
    # First copy uses original args/kwargs
    copies.append((input_args, input_kwargs))

    # Create cloned copies for remaining rotations
    for _ in range(num_rotations - 1):
        cloned_args = _clone_structure(input_args)
        cloned_kwargs = _clone_structure(input_kwargs)
        copies.append((cloned_args, cloned_kwargs))

    return copies


def _infer_device_from_tensors(input_args, input_kwargs, default="cuda"):
    """
    Infer CUDA device from GPU tensors in input_args/input_kwargs.

    Args:
        input_args: Positional arguments tuple.
        input_kwargs: Keyword arguments dict (can be None).
        default: Default device if no GPU tensors found.

    Returns:
        Device string or torch.device.
    """
    if input_kwargs is None:
        input_kwargs = {}
    gpu_tensors = _extract_gpu_tensors(input_args) + _extract_gpu_tensors(input_kwargs)
    if gpu_tensors:
        return gpu_tensors[0].device
    return default


def _ceil_to_ue8m0(x: torch.Tensor):
    """imported from DeepGEMM"""
    assert x.view(-1).amax().item() > 0
    return torch.pow(2.0, torch.ceil(torch.log2(x.abs())))


def per_token_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """imported from DeepGEMM"""
    assert x.dim() == 2 and x.size(1) % 128 == 0
    m, n = x.shape
    x_view = x.view(m, -1, 128)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    sf = _ceil_to_ue8m0(x_amax / 448.0)
    return (x_view * (1.0 / sf.unsqueeze(2))).to(torch.float8_e4m3fn).view(m, n), sf


def per_block_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """imported from DeepGEMM"""
    assert x.dim() == 2
    m, n = x.shape
    x_padded = torch.zeros(
        (round_up(m, 128), round_up(n, 128)), dtype=x.dtype, device=x.device
    )
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    sf = _ceil_to_ue8m0(x_amax / 448.0)
    x_scaled = (x_view * (1.0 / sf)).to(torch.float8_e4m3fn)
    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), sf.view(
        x_view.size(0), x_view.size(2)
    )


def quantize_fp8(x, scale_shape, tile_shape, scale_major_mode):
    """
    Quantizes a 2D or 3D tensor to FP8.

    Args:
        x (torch.Tensor): The 2D or 3D input tensor.
        scale_shape (tuple): The shape of the scale tensor.
        tile_shape (tuple): The shape of the tiles.
        scale_major_mode (str): The tiling order, "K" for row-major like,
                                or another value for column-major like.

    Returns:
        tuple: A tuple containing the quantized FP8 tensor and the
               calculated float32 scales.
    """
    # 1. Assertions and Initial Setup
    ndim = x.ndim
    assert ndim in [2, 3], f"x.ndim must be 2 or 3, but got {ndim}"
    assert ndim == len(scale_shape) == len(tile_shape)

    fp8_info = torch.finfo(torch.float8_e4m3fn)
    fp8_amax = torch.tensor(fp8_info.max, device=x.device, dtype=torch.float32)

    # 2. Tiling and Scale Calculation
    if ndim == 2:
        s0, s1 = scale_shape
        t0, t1 = tile_shape
        if scale_major_mode == "K":
            # Tile x and find the max absolute value in each tile
            x_tiled = rearrange(x, "(s0 t0) (s1 t1) -> s0 s1 t0 t1", s0=s0, s1=s1)
            abs_max = reduce(x_tiled.abs(), "s0 s1 t0 t1 -> s0 s1", "max").clamp(1e-4)
            x_scale = abs_max / fp8_amax
            x_scale = torch.pow(2.0, torch.ceil(torch.log2(x_scale.abs())))

            # Broadcast scales back to the original tensor shape
            scales_repeated = repeat(x_scale, "s0 s1 -> (s0 t0) (s1 t1)", t0=t0, t1=t1)
        else:
            # Handle column-major tiling
            x_tiled = rearrange(x, "(s1 t0) (s0 t1) -> s0 s1 t0 t1", s0=s0, s1=s1)
            abs_max = reduce(x_tiled.abs(), "s0 s1 t0 t1 -> s0 s1", "max").clamp(1e-4)
            x_scale = abs_max / fp8_amax
            x_scale = torch.pow(2.0, torch.ceil(torch.log2(x_scale.abs())))

            # Permute scale axes before repeating to match layout
            scales_permuted = rearrange(x_scale, "s0 s1 -> s1 s0")
            scales_repeated = repeat(
                scales_permuted, "s1 s0 -> (s1 t0) (s0 t1)", t0=t0, t1=t1
            )

    elif ndim == 3:
        s0, s1, s2 = scale_shape
        t0, t1, t2 = tile_shape
        if scale_major_mode == "K":
            # Tile x and find the max absolute value in each tile
            x_tiled = rearrange(
                x, "(s0 t0) (s1 t1) (s2 t2) -> s0 s1 s2 t0 t1 t2", s0=s0, s1=s1, s2=s2
            )
            abs_max = reduce(
                x_tiled.abs(), "s0 s1 s2 t0 t1 t2 -> s0 s1 s2", "max"
            ).clamp(1e-4)
            x_scale = abs_max / fp8_amax
            x_scale = torch.pow(2.0, torch.ceil(torch.log2(x_scale.abs())))

            # Broadcast scales back to the original tensor shape
            scales_repeated = repeat(
                x_scale, "s0 s1 s2 -> (s0 t0) (s1 t1) (s2 t2)", t0=t0, t1=t1, t2=t2
            )
        else:
            # Handle layout where the last two axes are swapped
            x_tiled = rearrange(
                x, "(s0 t0) (s2 t1) (s1 t2) -> s0 s1 s2 t0 t1 t2", s0=s0, s1=s1, s2=s2
            )
            abs_max = reduce(
                x_tiled.abs(), "s0 s1 s2 t0 t1 t2 -> s0 s1 s2", "max"
            ).clamp(1e-4)
            x_scale = abs_max / fp8_amax
            x_scale = torch.pow(2.0, torch.ceil(torch.log2(x_scale.abs())))

            # Permute scale axes before repeating to match layout
            scales_permuted = rearrange(x_scale, "s0 s1 s2 -> s0 s2 s1")
            scales_repeated = repeat(
                scales_permuted,
                "s0 s2 s1 -> (s0 t0) (s2 t1) (s1 t2)",
                t0=t0,
                t1=t1,
                t2=t2,
            )

    # 3. Final Quantization
    # Divide the original tensor by the broadcasted scales
    x_fp32 = x / (scales_repeated + 1e-8)

    # Convert the result to the target FP8 format
    x_fp8 = x_fp32.to(torch.float8_e4m3fn)

    return x_fp8, x_scale


def dequantize_fp8(x, x_scale, scale_major_mode):
    """
    Quantizes a 2D or 3D tensor to FP8.

    Args:
        x (torch.Tensor): The 2D or 3D input tensor.
        scale_shape (tuple): The shape of the scale tensor.
        tile_shape (tuple): The shape of the tiles.
        scale_major_mode (str): The tiling order, "K" for row-major like,
                                or another value for column-major like.

    Returns:
        tuple: A tuple containing the quantized FP8 tensor and the
               calculated float32 scales.
    """
    # 1. Assertions and Initial Setup
    ndim = x.ndim
    assert ndim in [2, 3], f"x.ndim must be 2 or 3, but got {ndim}"
    assert ndim == len(x_scale.shape)

    # 2. Tiling and Scale Calculation
    if ndim == 2:
        if scale_major_mode == "K":
            s0, s1 = x_scale.shape
        else:
            s1, s0 = x_scale.shape
        x = rearrange(
            x.to(torch.float32), "(s0 t0) (s1 t1) -> s0 s1 t0 t1", s0=s0, s1=s1
        )
        if scale_major_mode == "K":
            x_scale = rearrange(x_scale, "s0 s1 -> s0 s1 1 1")
        else:
            x_scale = rearrange(x_scale, "s0 s1 -> s1 s0 1 1")
        out = rearrange(x * x_scale, "s0 s1 t0 t1 -> (s0 t0) (s1 t1)")

    elif ndim == 3:
        if scale_major_mode == "K":
            s0, s1, s2 = x_scale.shape
        else:
            s0, s2, s1 = x_scale.shape
        x = rearrange(
            x.to(torch.float32),
            "(s0 t0) (s1 t1) (s2 t2)-> s0 s1 s2 t0 t1 t2",
            s0=s0,
            s1=s1,
            s2=s2,
        )
        if scale_major_mode == "K":
            x_scale = rearrange(x_scale, "s0 s1 s2 -> s0 s1 s2 1 1 1")
        else:
            x_scale = rearrange(x_scale, "s0 s1 s2 -> s0 s2 s1 1 1 1")
        out = rearrange(x * x_scale, "s0 s1 s2 t0 t1 t2 -> (s0 t0) (s1 t1) (s2 t2)")
    return out


def set_seed(random_seed):
    """
    Set random seed for reproducibility during testing.

    Args:
        random_seed (int): Random seed to set.

    Returns:
        None
    """
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)


def sleep_after_kernel_run(execution_time):
    """
    Sleep after kernel run. Dynamically adjust sleep time up to 1 sec based on execution time.

    Args:
        execution_time (float): Kernel execution time in milliseconds.

    Returns:
        None
    """
    if not math.isinf(execution_time):
        sleep_time = np.min([execution_time / 200, 1.0])
    else:
        sleep_time = 0.01
    time.sleep(sleep_time)
    return


def attention_flops(
    batch_size,
    qo_seqlen,
    kv_seqlen,
    head_dim_qk,
    head_dim_vo,
    num_qo_heads,
    causal,
):
    """
    Calculate FLOPs for a given attention layer. Assumes all sequence lengths are the same within the batch

    Args:
        batch_size (int): Batch size.
        qo_seqlen (int): Sequence length of the query. Assumed same within the batch.
        kv_seqlen (int): Sequence length of the key and value. Assumed same within the batch.
        head_dim_qk (int): Head dimension of the query and key.
        head_dim_vo (int): Head dimension of the value.
        num_qo_heads (int): Number of query heads.
        causal (bool): Whether to use causal masking. FLOPs is halved for causal masking.

    Returns:
        total_flops (int): Total FLOPs for the layer.
    """
    # Causal attention requires kv_len >= q_len
    if qo_seqlen > kv_seqlen:
        raise ValueError(
            "qo_seqlen must be less than or equal to kv_seqlen for causal attention"
        )

    if causal:
        bmm1_flops = (
            batch_size
            * (2 * kv_seqlen - qo_seqlen)
            * qo_seqlen
            * num_qo_heads
            * head_dim_qk
        )
        bmm2_flops = (
            batch_size
            * (2 * kv_seqlen - qo_seqlen)
            * qo_seqlen
            * num_qo_heads
            * head_dim_vo
        )
    else:
        bmm1_flops = 2 * batch_size * qo_seqlen * kv_seqlen * num_qo_heads * head_dim_qk
        bmm2_flops = 2 * batch_size * qo_seqlen * kv_seqlen * num_qo_heads * head_dim_vo
    total_flops = bmm1_flops + bmm2_flops
    return total_flops


def attention_flops_with_actual_seq_lens(
    actual_seq_lens_q,
    actual_seq_lens_kv,
    head_dim_qk,
    head_dim_vo,
    num_qo_heads,
    causal,
):
    """
    Calculate FLOPs for a given attention layer with actual sequence lengths where
    actual sequence lengths are provided as 1D tensors.

    Args:
        actual_seq_lens_q (torch.Tensor): Array of actual sequence lengths of the query.
        actual_seq_lens_kv (torch.Tensor): Array of actual sequence lengths of the key and value.
        head_dim_qk (int): Head dimension of the query and key.
        head_dim_vo (int): Head dimension of the value.
        num_qo_heads (int): Number of query heads.
        causal (bool): Whether to use causal masking.
        Note: Causal must be false for decode as this function assumes qo_seqlen == kv_seqlen.

    Returns:
        total_flops (int): Total FLOPs for the layer.
    """
    # Causal attention requires kv_len >= q_len
    # Otherwise right align if kv_len > q_len
    if causal and (actual_seq_lens_q > actual_seq_lens_kv).any():
        raise ValueError(
            "actual_seq_lens_q must be less than or equal to actual_seq_lens_kv for causal attention"
        )

    if causal:
        bmm1_flops = (
            torch.dot(
                2 * actual_seq_lens_kv.to(torch.float32)
                - actual_seq_lens_q.to(torch.float32),
                actual_seq_lens_q.to(torch.float32),
            )
            * num_qo_heads
            * head_dim_qk
        )
        bmm2_flops = (
            torch.dot(
                2 * actual_seq_lens_kv.to(torch.float32)
                - actual_seq_lens_q.to(torch.float32),
                actual_seq_lens_q.to(torch.float32),
            )
            * num_qo_heads
            * head_dim_vo
        )

    else:
        bmm1_flops = (
            2
            * torch.dot(
                actual_seq_lens_kv.to(torch.float32),
                actual_seq_lens_q.to(torch.float32),
            )
            * num_qo_heads
            * head_dim_qk
        )
        bmm2_flops = (
            2
            * torch.dot(
                actual_seq_lens_kv.to(torch.float32),
                actual_seq_lens_q.to(torch.float32),
            )
            * num_qo_heads
            * head_dim_vo
        )

    total_flops = bmm1_flops + bmm2_flops
    return total_flops


def attention_tflops_per_sec(
    batch_size,
    qo_seqlen,
    kv_seqlen,
    head_dim_qk,
    head_dim_vo,
    num_qo_heads,
    causal,
    time,
):
    """
    Calculate TFLOPS per second for a given attention layer. Assumes all sequence lengths are the same within the batch.

    Args:
        batch_size (int): Batch size.
        qo_seqlen (int): Sequence length of the query.
        kv_seqlen (int): Sequence length of the key and value.
        head_dim_qk (int): Head dimension of the query and key.
        head_dim_vo (int): Head dimension of the value.
        num_qo_heads (int): Number of query heads.
        causal (bool): Whether to use causal masking.
        time (float): Execution time in milliseconds.

    Returns:
        tflops_per_sec (float): TFLOPS per second for the layer.
    """
    f = attention_flops(
        batch_size,
        qo_seqlen,
        kv_seqlen,
        head_dim_qk,
        head_dim_vo,
        num_qo_heads,
        causal,
    )
    return f / time / 1e9 if not math.isnan(time) else 0.0


def attention_tflops_per_sec_with_actual_seq_lens(
    actual_seq_lens_q,
    actual_seq_lens_kv,
    head_dim_qk,
    head_dim_vo,
    num_qo_heads,
    causal,
    ms,
):
    """
    Calculate TFLOPS per second for a given attention layer with actual sequence lengths.
    Does not assume all sequence lengths are the same within the batch.

    Args:
        actual_seq_lens_q (torch.Tensor): Array of actual sequence lengths of the query.
        actual_seq_lens_kv (torch.Tensor): Array of actual sequence lengths of the key and value.
        head_dim_qk (int): Head dimension of the query and key.
        head_dim_vo (int): Head dimension of the value.
        num_qo_heads (int): Number of query heads.
        causal (bool): Whether to use causal masking.
        ms (float): Execution time in milliseconds.

    Returns:
        tflops_per_sec (float): TFLOPS per second for the layer.
    """
    f = attention_flops_with_actual_seq_lens(
        actual_seq_lens_q,
        actual_seq_lens_kv,
        head_dim_qk,
        head_dim_vo,
        num_qo_heads,
        causal,
    )
    return f.item() / ms / 1e9 if not math.isnan(ms) else 0.0


def attention_tb_per_sec(
    batch_size,
    qo_seqlen,
    kv_seqlen,
    head_dim_qk,
    head_dim_vo,
    num_qo_heads,
    num_kv_heads,
    time,
    q_dtype=torch.bfloat16,
    kv_dtype=torch.bfloat16,
    o_dtype=torch.bfloat16,
):
    """
    Calculate TB per second perf achieved for a given attention layer. Assumes all sequence lengths are the same within the batch.

    Args:
        batch_size (int): Batch size.
        qo_seqlen (int): Sequence length of the query.
        kv_seqlen (int): Sequence length of the key and value.
        head_dim_qk (int): Head dimension of the query and key.
        head_dim_vo (int): Head dimension of the value.
        num_qo_heads (int): Number of query heads.
        num_kv_heads (int): Number of key and value heads.
        time (float): Execution time in milliseconds.
        q_dtype (torch.dtype): Data type of the query.
        kv_dtype (torch.dtype): Data type of the key and value.
        o_dtype (torch.dtype): Data type of the output.

    Returns:
        tb_per_sec (float): TB per second for the layer.
    """
    q_bytes = batch_size * qo_seqlen * num_qo_heads * head_dim_qk * q_dtype.itemsize
    k_bytes = batch_size * kv_seqlen * num_kv_heads * head_dim_qk * kv_dtype.itemsize
    v_bytes = batch_size * kv_seqlen * num_kv_heads * head_dim_vo * kv_dtype.itemsize
    o_bytes = batch_size * qo_seqlen * num_qo_heads * head_dim_vo * o_dtype.itemsize
    total_bytes = q_bytes + k_bytes + v_bytes + o_bytes

    time_in_sec = time / 1e3
    bytes_in_tb = total_bytes / 1e12  # TB not TiB
    return bytes_in_tb / time_in_sec if not math.isnan(time) else 0.0


def attention_tb_per_sec_with_actual_seq_lens(
    actual_seq_lens_q,
    actual_seq_lens_kv,
    head_dim_qk,
    head_dim_vo,
    num_qo_heads,
    num_kv_heads,
    time,
    q_dtype=torch.bfloat16,
    kv_dtype=torch.bfloat16,
    o_dtype=torch.bfloat16,
):
    """
    Calculate TB per second perf achieved for a given attention layer with actual sequence lengths.
    Does not assume all sequence lengths are the same within the batch.

    Args:
        actual_seq_lens_q (torch.Tensor): Array of actual sequence lengths of the query.
        actual_seq_lens_kv (torch.Tensor): Array of actual sequence lengths of the key and value.
        head_dim_qk (int): Head dimension of the query and key.
        head_dim_vo (int): Head dimension of the value.
        num_qo_heads (int): Number of query heads.
        num_kv_heads (int): Number of key and value heads.
        time (float): Execution time in milliseconds.
        q_dtype (torch.dtype): Data type of the query.
        kv_dtype (torch.dtype): Data type of the key and value.
        o_dtype (torch.dtype): Data type of the output.

    Returns:
        tb_per_sec (float): TB per second for the layer.
    """
    q_bytes = (
        torch.sum(actual_seq_lens_q) * num_qo_heads * head_dim_qk * q_dtype.itemsize
    )
    k_bytes = (
        torch.sum(actual_seq_lens_kv) * num_kv_heads * head_dim_qk * kv_dtype.itemsize
    )
    v_bytes = (
        torch.sum(actual_seq_lens_kv) * num_kv_heads * head_dim_vo * kv_dtype.itemsize
    )
    o_bytes = (
        torch.sum(actual_seq_lens_q) * num_qo_heads * head_dim_vo * o_dtype.itemsize
    )

    total_bytes = (q_bytes + k_bytes + v_bytes + o_bytes).item()

    time_in_sec = time / 1e3
    bytes_in_tb = total_bytes / 1e12  # TB not TiB
    return bytes_in_tb / time_in_sec if not math.isnan(time) else 0.0


def aggregate_gpu_time_across_ranks(x, op):
    """
    Aggregate GPU time across ranks.

    Args:
        x (int | float | List[int] | List[float]): GPU time to aggregate.
        op (Callable): Operation to perform across ranks.

    Returns:
        int | float | List[int] | List[float]: Aggregated GPU time.
    """
    if torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
        if world_size > 1:
            x_all = [None] * world_size
            torch.distributed.all_gather_object(x_all, x)
            if isinstance(x, list):
                x = [op(val) for val in zip(*x_all, strict=True)]
            else:
                x = op(x_all)
    return x


def bench_gpu_time_with_cuda_event(
    fn,
    dry_run_iters: int = None,
    repeat_iters: int = None,
    dry_run_time_ms: int = 25,
    repeat_time_ms: int = 100,
    l2_flush: Optional[bool] = None,  # Deprecated. Use cold_l2_cache instead
    l2_flush_size_mb: Optional[int] = None,  # Deprecated. Use cold_l2_cache instead
    l2_flush_device: Optional[str] = None,  # Deprecated. Use cold_l2_cache instead
    sleep_after_run: bool = False,
    input_args: Tuple = (),
    input_kwargs: Optional[dict] = None,
    cold_l2_cache: bool = True,
    aggregate_op: Callable = max,
):
    """
    Benchmark kernel execution time using CUDA events (no CUDA graphs).

    This is the simplest benchmarking method. Best suited for kernels where launch overhead
    is negligible compared to execution time.

    The function performs:
    1. A quick estimation phase (5 iterations) to determine iteration counts
    2. Dry-run warmup iterations (not measured)
    3. Measured iterations with per-iteration timing via CUDA events

    Iteration counts can be specified directly or derived from target durations:
    - If dry_run_iters/repeat_iters are provided, those counts are used directly.
    - Otherwise, counts are computed from dry_run_time_ms/repeat_time_ms.

    Args:
        fn (Callable): The kernel function to benchmark.
        dry_run_iters (int, optional): Number of warmup iterations (not timed).
            If None, computed from dry_run_time_ms.
        repeat_iters (int, optional): Number of measured iterations.
            If None, computed from repeat_time_ms.
        dry_run_time_ms (int): Target warmup duration in ms (default: 25).
        repeat_time_ms (int): Target measurement duration in ms (default: 100).
        sleep_after_run (bool): If True, sleep briefly after each iteration to
            reduce thermal throttling (default: False).
        input_args (tuple): Positional arguments to pass to fn.
        input_kwargs (dict, optional): Keyword arguments to pass to fn.
        cold_l2_cache (bool): If True, flush L2 cache before each iteration to
            ensure cold-cache performance measurements (default: True).
        aggregate_op (Callable): Aggregate operation to perform across ranks (default: max).

    Returns:
        List[float]: Per-iteration execution times in milliseconds.

    Example:
        Basic usage:

        >>> def my_kernel(a, b):
        ...     return torch.matmul(a, b.T)
        >>> q = torch.randn(1024, 128, device="cuda")
        >>> k = torch.randn(1024, 128, device="cuda")
        >>> times = bench_gpu_time_with_cuda_event(
        ...     fn=my_kernel,
        ...     input_args=(q, k),
        ... )
        >>> print(f"Median time: {np.median(times):.3f} ms")

    Note:
        This method does NOT use CUDA graphs, so each iteration incurs kernel
        launch overhead. For microbenchmarking where launch latency matters,
        consider using ``bench_gpu_time_with_cudagraph`` instead.

    .. deprecated::
        The ``l2_flush``, ``l2_flush_size_mb``, and ``l2_flush_device`` parameters
        are deprecated. Use ``cold_l2_cache`` instead.
    """
    if input_kwargs is None:
        input_kwargs = {}

    # Handle deprecated parameters
    if any(p is not None for p in [l2_flush, l2_flush_size_mb, l2_flush_device]):
        warnings.warn(
            "l2_flush, l2_flush_size_mb, and l2_flush_device are deprecated. "
            "Use cold_l2_cache instead.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        _do_l2_flush = l2_flush if l2_flush is not None else True
        _l2_flush_size_mb = l2_flush_size_mb if l2_flush_size_mb is not None else 256
        _l2_flush_device = l2_flush_device if l2_flush_device is not None else "cuda"
    else:
        _do_l2_flush = cold_l2_cache
        # Dynamically determine L2 flush size and device
        _l2_flush_device = _infer_device_from_tensors(input_args, input_kwargs, "cuda")
        l2_size = get_l2_cache_size(_l2_flush_device)
        # Use 2x L2 size to ensure complete flush
        _l2_flush_size_mb = (l2_size * 2) // (1024 * 1024)

    # Check if args are provided (determines how we call fn)
    has_args = bool(input_args) or bool(input_kwargs)

    def call_fn():
        if has_args:
            fn(*input_args, **input_kwargs)
        else:
            fn()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    buffer = None
    if _do_l2_flush:
        l2_flush_size = int(_l2_flush_size_mb) * 1024 * 1024
        buffer = torch.empty(l2_flush_size, device=_l2_flush_device, dtype=torch.int8)

    ## Estimate kernel execution time by running the kernel 5 times
    measurement_iters = 5
    torch.cuda.synchronize()
    call_fn()  # Call once to exclude initial overhead
    torch.cuda.synchronize()
    start_event.record()
    for _ in range(measurement_iters):
        if _do_l2_flush:
            buffer.zero_()
        call_fn()
    end_event.record()
    torch.cuda.synchronize()
    estimated_kernel_execution_time = aggregate_gpu_time_across_ranks(
        start_event.elapsed_time(end_event) / measurement_iters,
        aggregate_op,
    )

    ## Set dry run and repeat iterations
    if dry_run_iters is None:
        dry_run_iters = max(1, int(dry_run_time_ms / estimated_kernel_execution_time))
    if repeat_iters is None:
        repeat_iters = max(1, int(repeat_time_ms / estimated_kernel_execution_time))

    # Dry runs
    torch.cuda.synchronize()
    for _ in range(dry_run_iters):
        if _do_l2_flush:
            buffer.zero_()
        call_fn()
    torch.cuda.synchronize()

    # Actual run
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat_iters)]
    torch.cuda.synchronize()
    for iter_idx in range(repeat_iters):
        if _do_l2_flush:
            buffer.zero_()
        start_events[iter_idx].record()
        call_fn()
        end_events[iter_idx].record()

        if sleep_after_run:
            sleep_after_kernel_run(estimated_kernel_execution_time)

    # Synchronize once outside of the loop to avoid synchronization overhead
    torch.cuda.synchronize()
    measured_times = []
    for iter_idx in range(repeat_iters):
        measured_times.append(start_events[iter_idx].elapsed_time(end_events[iter_idx]))
    measured_times = aggregate_gpu_time_across_ranks(measured_times, aggregate_op)
    return measured_times


def _import_cupti_or_raise():
    """Import CUPTI (>= 13) for the statistics timing path, or raise.

    The adaptive ``bench_gpu_time_with_statistics`` path intentionally does NOT
    fall back to CUDA events: launch-overhead-free, hardware per-kernel GPU time
    is what makes the convergence criteria trustworthy, so a missing/old CUPTI is
    a hard error rather than a silent downgrade.
    """
    from importlib.metadata import version as _pkg_version

    try:
        from cupti import cupti
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "bench_gpu_time_with_statistics requires CUPTI (cupti-python >= 13). "
            "Install with 'pip install -U cupti-python' (needs CUDA 13+). "
            "For fixed-count timing without CUPTI, use bench_gpu_time()."
        ) from e

    cupti_version = _pkg_version("cupti-python")
    if int(cupti_version.split(".")[0]) < 13:
        raise RuntimeError(
            "bench_gpu_time_with_statistics requires cupti-python >= 13.0.0 "
            f"(found {cupti_version}). Try 'pip install -U cupti-python'."
        )
    return cupti


def _cupti_kernel_spans(
    iter_timestamps,
    launches,
    kernels,
    *,
    start_index: int = 0,
    kernel_names=None,
):
    """Convert per-iteration CPU timestamp windows into GPU kernel spans (ms).

    Shared post-processing for the CUPTI timing paths. For each iteration's
    ``[start_cpu, end_cpu]`` host window, find the launch/runtime activities in
    that window (binary search), collect the GPU kernel activities they spawned
    (via correlation id), and return ``max(kernel.end) - min(kernel.start)`` in
    milliseconds. Processes ``iter_timestamps[start_index:]`` and threads
    ``kernel_names`` through so the kernel-set consistency check holds across
    incremental (batched) calls. Returns ``(spans, kernel_names)``.

    Algorithm matches the inline logic in :func:`bench_gpu_time_with_cupti`
    (O(N + M log M)).
    """
    import bisect

    sorted_launches = sorted(launches, key=lambda launch: launch[0])
    launch_starts = [launch[0] for launch in sorted_launches]

    corr_id_to_kernels: dict = {}
    for k in kernels:
        corr_id_to_kernels.setdefault(k[3], []).append(k)

    def kernel_string(k):
        # start/end/correlation_id are intentionally excluded from the identity.
        return f"{k[0]}_{k[4]}_{k[5]}_{k[6]}_{k[7]}"

    spans: list = []
    for idx in range(start_index, len(iter_timestamps)):
        start_cpu, end_cpu = iter_timestamps[idx]
        left_idx = bisect.bisect_left(launch_starts, start_cpu)
        right_idx = bisect.bisect_right(launch_starts, end_cpu)
        corr_ids = {sorted_launches[i][2] for i in range(left_idx, right_idx)}

        iter_kernels = []
        for corr_id in corr_ids:
            iter_kernels.extend(corr_id_to_kernels.get(corr_id, ()))
        if not iter_kernels:
            raise ValueError(f"No kernel activities recorded for iteration {idx}")

        current_names = {kernel_string(k) for k in iter_kernels}
        if kernel_names is None:
            kernel_names = current_names
        elif kernel_names != current_names:
            raise ValueError(
                f"Inconsistent kernel names: {kernel_names} != {current_names}"
            )

        min_start = min(k[1] for k in iter_kernels)
        max_end = max(k[2] for k in iter_kernels)
        spans.append((max_end - min_start) / 1e6)  # ns -> ms
    return spans, kernel_names


def bench_gpu_time_with_cupti(
    fn,
    dry_run_iters: int = None,
    repeat_iters: int = None,
    dry_run_time_ms: int = 25,
    repeat_time_ms: int = 100,
    l2_flush: Optional[bool] = None,  # Deprecated. Use cold_l2_cache instead
    l2_flush_size_mb: Optional[int] = None,  # Deprecated. Use cold_l2_cache instead
    l2_flush_device: Optional[str] = None,  # Deprecated. Use cold_l2_cache instead
    sleep_after_run: bool = False,
    use_cuda_graph: bool = False,
    input_args: Tuple = (),
    input_kwargs: Optional[dict] = None,
    cold_l2_cache: bool = True,
    aggregate_op: Callable = max,
):
    """
    Benchmark GPU time using CUPTI activity tracing for precise kernel timing.

    CUPTI (CUDA Profiling Tools Interface) provides hardware-level profiling that
    measures actual GPU kernel execution time, excluding CPU-side launch overhead.
    This gives the most accurate kernel performance measurements.

    Cold L2 cache is achieved via L2 flush between iterations. CUPTI measures
    per-iteration, so L2 flush works correctly regardless of ``use_cuda_graph``.

    Behavior:
    - Uses CUPTI (requires version >= 13, i.e., CUDA 13+) to trace kernel activities
      and compute per-iteration GPU time from recorded start/end timestamps.
    - Optionally captures operations in a CUDA graph (use_cuda_graph=True) for
      reduced launch overhead during measurement.
    - If CUPTI is unavailable, falls back to:
      - ``bench_gpu_time_with_cudagraph`` if use_cuda_graph=True (uses rotating buffers
        for cold L2)
      - ``bench_gpu_time_with_cuda_event`` otherwise (uses L2 flush for cold L2)

    Args:
        fn (Callable): The kernel function to benchmark.
        dry_run_iters (int, optional): Number of warmup iterations (not timed).
            If None, computed from dry_run_time_ms.
        repeat_iters (int, optional): Number of measured iterations.
            If None, computed from repeat_time_ms.
        dry_run_time_ms (int): Target warmup duration in ms (default: 25).
        repeat_time_ms (int): Target measurement duration in ms (default: 100).
        sleep_after_run (bool): If True, sleep briefly after each iteration (default: False).
        use_cuda_graph (bool): If True, capture and replay a CUDA graph (default: False).
        input_args (tuple): Positional arguments to pass to fn.
        input_kwargs (dict, optional): Keyword arguments to pass to fn.
        cold_l2_cache (bool): If True, flush L2 cache before each iteration to
            ensure cold-cache performance measurements (default: True).
        aggregate_op (Callable): Aggregate operation to perform across ranks (default: max).

    Returns:
        List[float]: Per-iteration GPU kernel execution times in milliseconds.

    Example:
        Basic CUPTI benchmarking (requires cupti-python >= 13):

        >>> def my_kernel(a, b):
        ...     return torch.matmul(a, b.T)
        >>> q = torch.randn(1024, 128, device="cuda")
        >>> k = torch.randn(1024, 128, device="cuda")
        >>> times = bench_gpu_time_with_cupti(
        ...     fn=my_kernel,
        ...     input_args=(q, k),
        ... )
        >>> print(f"Median GPU time: {np.median(times):.3f} ms")

    Note:
        Requires ``cupti-python`` package version >= 13.0.0:
        ``pip install -U cupti-python``

        If CUPTI is not available, a warning is issued and the function
        automatically falls back to CUDA event or CUDA graph timing.

    .. deprecated::
        The ``l2_flush``, ``l2_flush_size_mb``, and ``l2_flush_device`` parameters
        are deprecated. Use ``cold_l2_cache`` instead.
    """
    if input_kwargs is None:
        input_kwargs = {}

    # Handle deprecated parameters
    if any(p is not None for p in [l2_flush, l2_flush_size_mb, l2_flush_device]):
        warnings.warn(
            "l2_flush, l2_flush_size_mb, and l2_flush_device are deprecated. "
            "Use cold_l2_cache instead.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        _do_l2_flush = l2_flush if l2_flush is not None else True
        _l2_flush_size_mb = l2_flush_size_mb if l2_flush_size_mb is not None else 256
        _l2_flush_device = l2_flush_device if l2_flush_device is not None else "cuda"
    else:
        _do_l2_flush = cold_l2_cache
        # Dynamically determine L2 flush size and device
        _l2_flush_device = _infer_device_from_tensors(input_args, input_kwargs, "cuda")
        l2_size = get_l2_cache_size(_l2_flush_device)
        # Use 2x L2 size to ensure complete flush
        _l2_flush_size_mb = (l2_size * 2) // (1024 * 1024)

    # check if CUPTI is installed and its version is >= 13.0.0
    try:
        from cupti import cupti
        from importlib.metadata import version as importlib_metadata_version

        cupti_version = importlib_metadata_version("cupti-python")
        if int(cupti_version.split(".")[0]) < 13:
            raise Exception(
                "CUPTI needs to be >= 13.0.0. Try 'pip install -U cupti-python'."
            )
        from functools import partial
    except (ModuleNotFoundError, Exception) as e:
        if isinstance(e, ModuleNotFoundError):
            warnings.warn(
                "CUPTI is not installed. Try 'pip install -U cupti-python'. Falling back to CUDA events for benchmarking.",
                category=UserWarning,
                stacklevel=2,
            )
        else:
            warnings.warn(
                f"{e} Falling back to CUDA events for benchmarking.",
                category=UserWarning,
                stacklevel=2,
            )
        # Fallback: internally decide cold-L2 strategy based on use_cuda_graph
        if use_cuda_graph:
            # CUDA graph fallback uses rotating buffers for cold L2
            return bench_gpu_time_with_cudagraph(
                fn=fn,
                dry_run_iters=dry_run_iters,
                repeat_iters=repeat_iters,
                dry_run_time_ms=dry_run_time_ms,
                repeat_time_ms=repeat_time_ms,
                sleep_after_run=sleep_after_run,
                input_args=input_args,
                input_kwargs=input_kwargs,
                cold_l2_cache=cold_l2_cache,
                aggregate_op=aggregate_op,
            )
        else:
            # Non-graph fallback uses L2 flush for cold L2
            return bench_gpu_time_with_cuda_event(
                fn=fn,
                dry_run_iters=dry_run_iters,
                repeat_iters=repeat_iters,
                dry_run_time_ms=dry_run_time_ms,
                repeat_time_ms=repeat_time_ms,
                sleep_after_run=sleep_after_run,
                input_args=input_args,
                input_kwargs=input_kwargs,
                cold_l2_cache=cold_l2_cache,
                aggregate_op=aggregate_op,
            )

    # CUPTI buffer callbacks
    def func_buffer_requested():
        buffer_size = 8 * 1024 * 1024
        max_num_records = 0
        return buffer_size, max_num_records

    def set_kernel_name(activity):
        if activity.kind == cupti.ActivityKind.CONCURRENT_KERNEL:
            return activity.name
        elif activity.kind == cupti.ActivityKind.MEMCPY:
            return "MEMCPY"
        elif activity.kind == cupti.ActivityKind.MEMSET:
            return "MEMSET"

    def get_bytes(activity):
        if activity.kind in (cupti.ActivityKind.MEMCPY, cupti.ActivityKind.MEMSET):
            return activity.bytes
        else:
            return 0

    def get_copy_kind(activity):
        if activity.kind == cupti.ActivityKind.MEMCPY:
            return activity.copy_kind
        else:
            return 0

    def get_value(activity):
        if activity.kind == cupti.ActivityKind.MEMSET:
            return activity.value
        else:
            return 0

    def collect_kernel_info(activity):
        return (
            set_kernel_name(activity),
            activity.start,
            activity.end,
            activity.correlation_id,
            get_copy_kind(activity),
            get_bytes(activity),
            get_value(activity),
            activity.kind,
        )

    def func_buffer_completed(
        launches: list[tuple[float, float, int, int, int]],
        kernels: list[tuple[str, float, float, int, int, int, int, int]],
        activities: list,
    ):
        for activity in activities:
            if activity.kind in (
                cupti.ActivityKind.CONCURRENT_KERNEL,
                cupti.ActivityKind.MEMCPY,
                cupti.ActivityKind.MEMSET,
            ):
                # Kernel activity
                kernels.append(collect_kernel_info(activity))
            elif activity.kind in (
                cupti.ActivityKind.RUNTIME,
                cupti.ActivityKind.DRIVER,
            ):
                # Runtime or Driver activity
                launches.append(
                    (
                        activity.start,
                        activity.end,
                        activity.correlation_id,
                        activity.cbid,
                        activity.kind,
                    )
                )

    # Check if args are provided (determines how we call fn)
    has_args = bool(input_args) or bool(input_kwargs)

    def call_fn():
        if has_args:
            fn(*input_args, **input_kwargs)
        else:
            fn()

    buffer = None
    if _do_l2_flush:
        l2_flush_size = int(_l2_flush_size_mb) * 1024 * 1024
        buffer = torch.empty(l2_flush_size, device=_l2_flush_device, dtype=torch.int8)

    # Prepare runner (either direct fn or CUDA graph replay)
    runner = call_fn
    g = None
    if use_cuda_graph:
        # Warmup run to avoid capturing one-time inits
        torch.cuda.synchronize()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                call_fn()
        torch.cuda.current_stream().wait_stream(s)

        # Capture kernel in graph
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            call_fn()
        runner = g.replay

    ## Estimate kernel execution time by running the runner 5 times
    measurement_iters = 5
    torch.cuda.synchronize()
    call_fn()  # Call once to exclude initial overhead
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(measurement_iters):
        if _do_l2_flush:
            buffer.zero_()
        runner()
    end_event.record()
    torch.cuda.synchronize()
    estimated_kernel_execution_time = aggregate_gpu_time_across_ranks(
        start_event.elapsed_time(end_event) / measurement_iters,
        aggregate_op,
    )

    ## Set dry run and repeat iterations
    if dry_run_iters is None:
        dry_run_iters = max(1, int(dry_run_time_ms / estimated_kernel_execution_time))
    if repeat_iters is None:
        repeat_iters = max(1, int(repeat_time_ms / estimated_kernel_execution_time))

    # Dry runs
    torch.cuda.synchronize()
    for _ in range(dry_run_iters):
        if _do_l2_flush:
            buffer.zero_()
        runner()
    torch.cuda.synchronize()

    # CUPTI measurement
    launches: list[tuple[float, float, int, int, int]] = []
    kernels: list[tuple[str, float, float, int, int, int, int, int]] = []
    iter_timestamps = []
    cupti.activity_enable(cupti.ActivityKind.RUNTIME)
    cupti.activity_enable(cupti.ActivityKind.CONCURRENT_KERNEL)
    cupti.activity_enable(cupti.ActivityKind.DRIVER)
    cupti.activity_enable(cupti.ActivityKind.MEMCPY)
    cupti.activity_enable(cupti.ActivityKind.MEMSET)
    cupti.activity_register_callbacks(
        func_buffer_requested, partial(func_buffer_completed, launches, kernels)
    )
    for _ in range(repeat_iters):
        if _do_l2_flush:
            buffer.zero_()
        torch.cuda.synchronize()
        start_cpu = cupti.get_timestamp()
        runner()
        end_cpu = cupti.get_timestamp()
        torch.cuda.synchronize()
        iter_timestamps.append((start_cpu, end_cpu))
        if sleep_after_run:
            sleep_after_kernel_run(estimated_kernel_execution_time)
    cupti.activity_flush_all(0)
    cupti.activity_disable(cupti.ActivityKind.RUNTIME)
    cupti.activity_disable(cupti.ActivityKind.CONCURRENT_KERNEL)
    cupti.activity_disable(cupti.ActivityKind.DRIVER)
    cupti.activity_disable(cupti.ActivityKind.MEMCPY)
    cupti.activity_disable(cupti.ActivityKind.MEMSET)
    cupti.finalize()

    # Process activities into per-iteration GPU kernel spans (O(N + M log M)).
    measured_times, _ = _cupti_kernel_spans(iter_timestamps, launches, kernels)
    measured_times = aggregate_gpu_time_across_ranks(measured_times, aggregate_op)
    return measured_times


def bench_gpu_time_with_cudagraph(
    fn,
    dry_run_iters: int = None,
    repeat_iters: int = None,
    dry_run_time_ms: int = 25,
    repeat_time_ms: int = 100,
    num_iters_within_graph: int = 10,
    l2_flush: Optional[bool] = None,  # Deprecated. Use cold_l2_cache instead
    l2_flush_size_mb: Optional[int] = None,  # Deprecated. Use cold_l2_cache instead
    l2_flush_device: Optional[str] = None,  # Deprecated. Use cold_l2_cache instead
    sleep_after_run: bool = False,
    input_args: Tuple = (),
    input_kwargs: Optional[dict] = None,
    cold_l2_cache: bool = True,
    aggregate_op: Callable = max,
):
    """
    Benchmark GPU time using CUDA graphs with amortized kernel launch overhead.

    CUDA graphs capture a sequence of GPU operations and replay them with minimal
    CPU overhead. By running multiple iterations within a single graph, kernel
    launch latency is amortized, yielding measurements closer to pure GPU time.

    **Cold-L2 Benchmarking**:

    When ``cold_l2_cache=True``, the function uses **rotating buffers** to ensure
    cold L2 cache for each kernel invocation within the graph. Multiple copies of
    the GPU tensors in ``input_args``/``input_kwargs`` are created and rotated
    through during graph capture, ensuring each kernel invocation operates on
    different memory regions. The number of buffer copies is automatically
    calculated based on the device's L2 cache size.

    Args:
        fn (Callable): The kernel function to benchmark.
        dry_run_iters (int, optional): Number of warmup iterations (not timed).
            If None, computed from dry_run_time_ms.
        repeat_iters (int, optional): Number of measured iterations (graph replays).
            If None, computed from repeat_time_ms.
        dry_run_time_ms (int): Target warmup duration in ms (default: 25).
        repeat_time_ms (int): Target measurement duration in ms (default: 100).
        num_iters_within_graph (int): Number of kernel calls captured in the graph
            (default: 10). Higher values better amortize launch overhead but use
            more memory when rotating buffers.
        sleep_after_run (bool): If True, sleep briefly after each iteration (default: False).
        input_args (tuple): Positional arguments to pass to fn. GPU tensors in
            this structure will be cloned when ``cold_l2_cache=True``.
        input_kwargs (dict, optional): Keyword arguments to pass to fn. GPU tensors
            in this structure will be cloned when ``cold_l2_cache=True``.
        cold_l2_cache (bool): If True, use rotating buffers to ensure cold L2 cache
            for each kernel invocation within the graph (default: True).
        aggregate_op (Callable): Aggregate operation to perform across ranks (default: max).

    Returns:
        List[float]: Per-iteration execution times in milliseconds. Each time is
        the graph replay duration divided by ``num_iters_within_graph``.

    Example:
        Cold-L2 benchmarking (default, for memory-bound kernels):

        >>> def run_attention(q, k, v, o):
        ...     flashinfer.single_prefill_with_kv_cache(q, k, v, o)
        ...
        >>> q = torch.randn(batch, heads, seq_len, head_dim, device="cuda")
        >>> k = torch.randn(batch, heads, seq_len, head_dim, device="cuda")
        >>> v = torch.randn(batch, heads, seq_len, head_dim, device="cuda")
        >>> o = torch.empty_like(q)
        >>> times = bench_gpu_time_with_cudagraph(
        ...     fn=run_attention,
        ...     input_args=(q, k, v, o),
        ... )
        >>> print(f"Cold-L2 median time: {np.median(times):.3f} ms")

    Example:
        Hot L2 benchmarking (for compute-bound kernels):

        >>> times = bench_gpu_time_with_cudagraph(
        ...     fn=lambda: torch.matmul(q, k.T),
        ...     cold_l2_cache=False,
        ... )

    Note:
        - When using ``input_args``/``input_kwargs``, the function must accept the
          tensors as arguments (not capture them from closure).
        - GPU tensors are automatically detected and cloned. Non-tensor arguments
          (scalars, booleans, etc.) are preserved across all copies.
        - Memory usage scales with the number of rotations needed to exceed L2 cache.

    See Also:
        - ``calculate_rotation_count``: Computes required buffer copies for cold-L2.

    .. deprecated::
        The ``l2_flush``, ``l2_flush_size_mb``, and ``l2_flush_device`` parameters
        are deprecated. Use ``cold_l2_cache`` instead.
    """
    if input_kwargs is None:
        input_kwargs = {}

    # Handle deprecated parameters
    if any(p is not None for p in [l2_flush, l2_flush_size_mb, l2_flush_device]):
        warnings.warn(
            "l2_flush, l2_flush_size_mb, and l2_flush_device are deprecated. "
            "Use cold_l2_cache instead. For CUDA graphs, cold_l2_cache uses "
            "rotating buffers (not L2 flush) to ensure cold cache.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        # For CUDA graphs, l2_flush had limited effectiveness, so we translate
        # l2_flush=True to cold_l2_cache=True (rotating buffers)
        _do_rotate = l2_flush if l2_flush is not None else True
    else:
        _do_rotate = cold_l2_cache

    # Dynamically determine device from input tensors
    _device = _infer_device_from_tensors(input_args, input_kwargs, "cuda")

    # Check if args are provided (determines how we call fn)
    has_args = bool(input_args) or bool(input_kwargs)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # Determine rotation count if rotating buffers
    num_rotations = 1
    rotated_copies = None
    if _do_rotate:
        # Extract all GPU tensors from args and kwargs
        gpu_tensors = _extract_gpu_tensors(input_args) + _extract_gpu_tensors(
            input_kwargs
        )
        if len(gpu_tensors) == 0:
            warnings.warn(
                "cold_l2_cache=True but no GPU tensors found in input_args/input_kwargs. "
                "Cold L2 benchmarking disabled.",
                category=UserWarning,
                stacklevel=2,
            )
            _do_rotate = False
        else:
            num_rotations = calculate_rotation_count(gpu_tensors, _device)
            if num_rotations > 1:
                rotated_copies = _create_rotated_buffer_copies(
                    input_args, input_kwargs, num_rotations
                )
            else:
                # No rotation needed (tensors exceed L2)
                _do_rotate = False

    # Define how to call fn
    def call_fn():
        if has_args:
            fn(*input_args, **input_kwargs)
        else:
            fn()

    def call_fn_with_rotation(buf_idx: int):
        args, kwargs = rotated_copies[buf_idx]
        fn(*args, **kwargs)

    # Warmup run
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            call_fn()
    torch.cuda.current_stream().wait_stream(s)

    # Capture kernel in graph
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        if _do_rotate and num_rotations > 1:
            # Capture with rotating buffers: use buffer[iter % num_rotations]
            for iter_idx in range(num_iters_within_graph):
                buf_idx = iter_idx % num_rotations
                call_fn_with_rotation(buf_idx)
        else:
            # Non-rotating capture (uses original args if provided)
            for _ in range(num_iters_within_graph):
                call_fn()
    torch.cuda.synchronize()

    ## Estimate kernel execution time by running the kernel 5 times
    measurement_iters = 5
    start_event.record()
    for _ in range(measurement_iters):
        g.replay()
    end_event.record()
    torch.cuda.synchronize()
    estimated_kernel_execution_time = aggregate_gpu_time_across_ranks(
        start_event.elapsed_time(end_event) / measurement_iters,
        aggregate_op,
    )

    ## Set dry run and repeat iterations
    if dry_run_iters is None:
        dry_run_iters = max(1, int(dry_run_time_ms / estimated_kernel_execution_time))
    if repeat_iters is None:
        repeat_iters = max(1, int(repeat_time_ms / estimated_kernel_execution_time))

    # Dry run
    torch.cuda.synchronize()
    for _ in range(dry_run_iters):
        g.replay()
    torch.cuda.synchronize()

    # Actual run
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat_iters)]
    torch.cuda.synchronize()
    for iter_idx in range(repeat_iters):
        start_events[iter_idx].record()
        g.replay()
        end_events[iter_idx].record()

        if sleep_after_run:
            sleep_after_kernel_run(estimated_kernel_execution_time)

    # Synchronize once outside of the loop to avoid synchronization overhead
    torch.cuda.synchronize()
    measured_times = []
    for iter_idx in range(repeat_iters):
        measured_times.append(
            start_events[iter_idx].elapsed_time(end_events[iter_idx])
            / num_iters_within_graph
        )
    measured_times = aggregate_gpu_time_across_ranks(measured_times, aggregate_op)
    return measured_times


def bench_gpu_time(
    fn,
    dry_run_iters: int = None,
    repeat_iters: int = None,
    dry_run_time_ms: int = 25,
    repeat_time_ms: int = 100,
    l2_flush: Optional[bool] = None,  # Deprecated. Use cold_l2_cache instead
    l2_flush_size_mb: Optional[int] = None,  # Deprecated. Use cold_l2_cache instead
    l2_flush_device: Optional[str] = None,  # Deprecated. Use cold_l2_cache instead
    sleep_after_run: bool = False,
    enable_cupti: bool = False,
    use_cuda_graph: bool = False,
    num_iters_within_graph: int = 10,
    input_args: Tuple = (),
    input_kwargs: Optional[dict] = None,
    cold_l2_cache: bool = True,
    aggregate_op: Callable = max,
):
    """
    Unified GPU benchmarking interface with configurable timing backends.

    This is the recommended entry point for GPU kernel benchmarking. It provides
    a single interface that dispatches to the appropriate timing implementation
    based on the configuration flags.

    **Timing Backends** (in order of precedence):

    1. **CUPTI** (``enable_cupti=True``): Most accurate, measures pure GPU kernel
       time via hardware profiling. Requires cupti-python >= 13.
    2. **CUDA Graphs** (``use_cuda_graph=True``): Amortizes launch overhead by
       capturing and replaying multiple kernel calls. Good balance of accuracy
       and availability.
    3. **CUDA Events** (default): Simplest method, measures launch + execution.
       Available everywhere but includes CPU overhead.

    **Cold-L2 Strategy** (automatically selected based on timing backend):

    .. list-table::
       :header-rows: 1

       * - Timing Backend
         - Cold-L2 Strategy
         - How it Works
       * - CUPTI
         - L2 Flush
         - Flush L2 cache before each iter
       * - CUDA Events (no CUDA Graphs)
         - L2 Flush
         - Flush L2 cache before each iter
       * - CUDA Events + CUDA Graphs
         - Rotating Buffers
         - Clone GPU tensors in input_args/input_kwargs and rotate through them
        use_cuda_graph (bool): If True, use CUDA graph timing (default: False).
        num_iters_within_graph (int): Kernel calls per graph (CUDA graph mode only,
            default: 10).
        input_args (tuple): Positional arguments to pass to fn.
        input_kwargs (dict, optional): Keyword arguments to pass to fn.
        cold_l2_cache (bool): If True, ensure cold L2 cache for each iteration
            (default: True). The strategy is automatically selected based on timing
            backend.
        aggregate_op (Callable): Aggregate operation to perform across ranks (default: max).

    Returns:
        List[float]: Per-iteration execution times in milliseconds.

    Example:
        Simple benchmarking with CUDA events (default):

        >>> times = bench_gpu_time(fn=lambda: my_kernel())
        >>> print(f"Median: {np.median(times):.3f} ms")

    Example:
        CUDA graph benchmarking for reduced launch overhead:

        >>> def run_kernel(x, y, out):
        ...     my_memory_bound_kernel(x, y, out)
        >>> times = bench_gpu_time(
        ...     fn=run_kernel,
        ...     input_args=(x, y, out),
        ...     use_cuda_graph=True,
        ... )

    Example:
        CUPTI benchmarking for most accurate GPU kernel time:

        >>> times = bench_gpu_time(
        ...     fn=run_kernel,
        ...     input_args=(x, y, out),
        ...     enable_cupti=True,
        ... )

    See Also:
        - ``bench_gpu_time_with_cuda_event``: Direct CUDA event timing.
        - ``bench_gpu_time_with_cudagraph``: Direct CUDA graph timing.
        - ``bench_gpu_time_with_cupti``: Direct CUPTI timing.

    .. deprecated::
        The ``l2_flush``, ``l2_flush_size_mb``, and ``l2_flush_device``
        parameters are deprecated. Use ``cold_l2_cache`` instead.
    """
    # Handle deprecated parameters
    if any(p is not None for p in [l2_flush, l2_flush_size_mb, l2_flush_device]):
        warnings.warn(
            "l2_flush, l2_flush_size_mb, and l2_flush_device are deprecated. "
            "Use cold_l2_cache instead.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        # If l2_flush was explicitly set, use it as the cold_l2_cache value
        _cold_l2_cache = l2_flush if l2_flush is not None else cold_l2_cache
    else:
        _cold_l2_cache = cold_l2_cache

    if enable_cupti:
        return bench_gpu_time_with_cupti(
            fn=fn,
            dry_run_iters=dry_run_iters,
            repeat_iters=repeat_iters,
            dry_run_time_ms=dry_run_time_ms,
            repeat_time_ms=repeat_time_ms,
            sleep_after_run=sleep_after_run,
            use_cuda_graph=use_cuda_graph,
            input_args=input_args,
            input_kwargs=input_kwargs,
            cold_l2_cache=_cold_l2_cache,
            aggregate_op=aggregate_op,
        )
    if use_cuda_graph:
        return bench_gpu_time_with_cudagraph(
            fn=fn,
            dry_run_iters=dry_run_iters,
            repeat_iters=repeat_iters,
            dry_run_time_ms=dry_run_time_ms,
            repeat_time_ms=repeat_time_ms,
            num_iters_within_graph=num_iters_within_graph,
            sleep_after_run=sleep_after_run,
            input_args=input_args,
            input_kwargs=input_kwargs,
            cold_l2_cache=_cold_l2_cache,
            aggregate_op=aggregate_op,
        )
    return bench_gpu_time_with_cuda_event(
        fn=fn,
        dry_run_iters=dry_run_iters,
        repeat_iters=repeat_iters,
        dry_run_time_ms=dry_run_time_ms,
        repeat_time_ms=repeat_time_ms,
        sleep_after_run=sleep_after_run,
        input_args=input_args,
        input_kwargs=input_kwargs,
        cold_l2_cache=_cold_l2_cache,
        aggregate_op=aggregate_op,
    )


class empty_suppress:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


class suppress_stdout_stderr:
    def __enter__(self):
        self.outnull_file = open(os.devnull, "w")
        self.errnull_file = open(os.devnull, "w")

        self.old_stdout_fileno_undup = sys.stdout.fileno()
        self.old_stderr_fileno_undup = sys.stderr.fileno()

        self.old_stdout_fileno = os.dup(sys.stdout.fileno())
        self.old_stderr_fileno = os.dup(sys.stderr.fileno())

        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr

        os.dup2(self.outnull_file.fileno(), self.old_stdout_fileno_undup)
        os.dup2(self.errnull_file.fileno(), self.old_stderr_fileno_undup)

        sys.stdout = self.outnull_file
        sys.stderr = self.errnull_file
        return self

    def __exit__(self, *_):
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr

        os.dup2(self.old_stdout_fileno, self.old_stdout_fileno_undup)
        os.dup2(self.old_stderr_fileno, self.old_stderr_fileno_undup)

        os.close(self.old_stdout_fileno)
        os.close(self.old_stderr_fileno)

        self.outnull_file.close()
        self.errnull_file.close()


# copied from DeepGEMM
def bench_kineto(
    fn,
    kernel_names,
    num_tests: int = 30,
    suppress_kineto_output: bool = False,
    trace_path: str = None,
    flush_l2: bool = True,
    with_multiple_kernels: bool = False,
):
    # Conflict with Nsight Systems
    using_nsys = int(os.environ.get("DG_NSYS_PROFILING", 0))

    # By default, flush L2 with an excessive 8GB memset to give the GPU some (literal) chill time without full idle
    flush_l2_size = int(8e9 // 4)

    # For some auto-tuning kernels with prints
    fn()

    # Profile
    suppress = (
        suppress_stdout_stderr
        if suppress_kineto_output and not using_nsys
        else empty_suppress
    )
    with suppress():
        schedule = (
            torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=1)
            if not using_nsys
            else None
        )
        profiler: Any = (
            torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA], schedule=schedule
            )
            if not using_nsys
            else empty_suppress()
        )
        with profiler:
            for _i in range(2):
                for _ in range(num_tests):
                    if flush_l2:
                        torch.empty(
                            flush_l2_size, dtype=torch.int, device="cuda"
                        ).zero_()
                    fn()

                if not using_nsys:
                    profiler.step()

    # Return 1 if using Nsight Systems
    if using_nsys:
        return 1

    # Parse the profiling table
    assert isinstance(kernel_names, (str, tuple))
    is_tuple = isinstance(kernel_names, tuple)
    prof_lines = (
        profiler.key_averages()
        .table(sort_by="cuda_time_total", max_name_column_width=100)
        .split("\n")
    )
    kernel_names = (kernel_names,) if isinstance(kernel_names, str) else kernel_names
    assert all([isinstance(name, str) for name in kernel_names])
    if not with_multiple_kernels:
        for name in kernel_names:
            assert sum([name in line for line in prof_lines]) == 1, (
                f"Errors of the kernel {name} in the profiling table"
            )

    # Save chrome traces
    if trace_path is not None:
        profiler.export_chrome_trace(trace_path)

    # Return average kernel times
    units = {"ms": 1e3, "us": 1e6}
    kernel_times = []
    for name in kernel_names:
        total_time = 0.0
        total_num = 0
        for line in prof_lines:
            if name in line:
                time_str = line.split()[-2]
                num_str = line.split()[-1]
                for unit, scale in units.items():
                    if unit in time_str:
                        total_time += (
                            float(time_str.replace(unit, "")) / scale * int(num_str)
                        )
                        total_num += int(num_str)
                        break
        kernel_times.append(total_time / total_num)

    return tuple(kernel_times) if is_tuple else kernel_times[0]


def count_bytes(*tensors):
    total = 0
    for t in tensors:
        if isinstance(t, (tuple, list)):
            total += count_bytes(*t)
        elif t is not None:
            total += t.numel() * t.element_size()
    return total


def compute_statistics(times) -> BenchmarkStatistics:
    """Summarize a list of per-iteration timings into NVBench-style statistics.

    Drop-in companion for :func:`bench_gpu_time` (which returns ``List[float]``):
    it produces the classic mean/stdev/coefficient-of-variation ("noise") plus
    the outlier-resistant median/IQR/relative-IQR pair, matching one row of an
    NVBench cold-time table.

    Args:
        times: Per-iteration execution times (any consistent unit; FlashInfer's
            timers return milliseconds).

    Returns:
        BenchmarkStatistics: see :class:`flashinfer.testing.statistics.BenchmarkStatistics`.

    Example:
        >>> times = bench_gpu_time(fn=lambda: my_kernel())
        >>> stats = compute_statistics(times)
        >>> print(stats.summary_str())
    """
    return BenchmarkStatistics.from_samples(times)


def _device_index(device) -> int:
    """Resolve an int CUDA device index from int / str / torch.device."""
    if isinstance(device, int):
        return device
    if not isinstance(device, torch.device):
        device = torch.device(device)
    return device.index if device.index is not None else torch.cuda.current_device()


class _ThrottleMonitor:
    """Host-side GPU clock throttle screen via NVML.

    Approximates NVBench's throttle rejection (``gpu_frequency.cxx``): if the SM
    clock sampled right after a kernel completes falls below
    ``threshold * reference_clock``, the sample is throttled and should be
    discarded + cooled down. NVBench reads the GPU clock register on-device at
    kernel start/stop for an exact in-kernel frequency; this reads the SM clock
    from NVML on the host immediately after ``cudaDeviceSynchronize``, which is a
    coarser proxy but needs no extra kernel. ``reference_clock`` is the device's
    max SM clock. Degrades to a no-op (with a warning) if NVML is unavailable.
    """

    def __init__(self, device, threshold: float) -> None:
        self.threshold = threshold
        self._ok = False
        self._pynvml = None
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(_device_index(device))
            self._pynvml = pynvml
            self._handle = handle
            self.reference_mhz = pynvml.nvmlDeviceGetMaxClockInfo(
                handle, pynvml.NVML_CLOCK_SM
            )
            self._ok = self.reference_mhz > 0
        except Exception as e:  # NVML missing / no permission / no device
            warnings.warn(
                f"Throttle screening unavailable (NVML error: {e}); "
                "continuing without throttle rejection.",
                stacklevel=2,
            )

    def is_throttled(self) -> bool:
        if not self._ok:
            return False
        try:
            cur = self._pynvml.nvmlDeviceGetClockInfo(
                self._handle, self._pynvml.NVML_CLOCK_SM
            )
        except Exception:
            return False
        return cur < self.threshold * self.reference_mhz

    def close(self) -> None:
        if self._pynvml is not None:
            with contextlib.suppress(Exception):
                self._pynvml.nvmlShutdown()


def bench_gpu_time_with_statistics(
    fn,
    *,
    stopping_criterion: str = "stdrel",
    max_noise: float = 0.005,
    min_time_ms: float = 500.0,
    max_time_ms: float = 10000.0,
    min_samples: int = 10,
    max_samples: int = 100000,
    target_samples: int = 100,
    batch_size: Optional[int] = None,
    dry_run_iters: Optional[int] = None,
    dry_run_time_ms: float = 25.0,
    use_cuda_graph: bool = False,
    throttle_screen: bool = False,
    throttle_threshold: float = 0.75,
    throttle_cooldown_ms: float = 5.0,
    max_consecutive_throttled: int = 100,
    input_args: Tuple = (),
    input_kwargs: Optional[dict] = None,
    cold_l2_cache: bool = True,
    return_samples: bool = True,
):
    """Adaptive, statistically-rigorous GPU timing (NVBench cold-measurement port).

    **Requires CUPTI** (``cupti-python >= 13``, CUDA 13+) and raises if it is not
    available -- there is no CUDA-event fallback. CUPTI gives hardware per-kernel
    GPU time that excludes CPU launch overhead, which is exactly what makes the
    convergence criteria meaningful (it is also how this path closes NVBench's
    launch-isolation gap without a host-side blocking kernel). For fixed-count
    timing without CUPTI, use :func:`bench_gpu_time`.

    Unlike :func:`bench_gpu_time` (which runs a *fixed* iteration count and
    returns the raw list), this drives a **sequential sampling procedure**: run a
    small batch of cold, L2-flushed, optionally throttle-screened iterations,
    measure each one's pure GPU time with CUPTI, update running statistics, then
    ask a swappable stopping criterion "have we seen enough?". Termination is
    bounded below by a min-sample floor and above by a wall-clock ceiling, so
    every reported number is the output of a convergence rule -- not "median of N
    fixed runs". This is the timing methodology you want when the number gates a
    CI regression or feeds an autotuner's reward.

    Sampling proceeds in batches (``batch_size`` iterations) because CUPTI
    collects activity records and reports per-iteration spans only after a buffer
    flush; the criterion is therefore evaluated at batch boundaries (it still
    sees every individual sample). Smaller batches converge sooner; larger
    batches amortize CUPTI flush overhead.

    Args:
        fn (Callable): Kernel function to benchmark.
        stopping_criterion (str): ``"stdrel"`` (default; converge relative stdev
            to ``max_noise`` with noise-plateau + invalid-estimate fallbacks),
            ``"entropy"`` (information-theoretic convergence), or
            ``"sample-count"`` (deterministic ``target_samples``, for
            reproducible CI). See :mod:`flashinfer.testing.statistics`.
        max_noise (float): Target relative stdev for ``stdrel`` (default 0.005 =
            0.5%, matching NVBench).
        min_time_ms (float): Minimum *accumulated GPU* time before ``stdrel`` may
            stop (default 500 ms = NVBench's 0.5 s lower bound).
        max_time_ms (float): Wall-clock ceiling; the loop always terminates by
            this (default 10 s).
        min_samples (int): Hard floor on samples before any criterion can stop.
        max_samples (int): Hard cap on samples (safety bound).
        target_samples (int): Sample count for the ``sample-count`` criterion.
        batch_size (int, optional): Iterations per CUPTI flush / criterion check
            (default 16). Capped per batch so ``sample-count`` lands exactly on
            ``target_samples`` and the run never exceeds ``max_samples``.
        dry_run_iters (int, optional): Warmup iterations (not timed). If None,
            derived from ``dry_run_time_ms`` and a 5-run estimate.
        dry_run_time_ms (float): Target warmup duration in ms (default 25).
        use_cuda_graph (bool): Capture ``fn`` into a CUDA graph and time its
            replay (default False). CUPTI measures the replayed kernels.
        throttle_screen (bool): If True, discard samples taken while the GPU SM
            clock is throttled (below ``throttle_threshold`` of max), with a
            growing cooldown. Requires NVML (already a FlashInfer dependency);
            off by default since it adds per-sample host overhead.
        throttle_threshold (float): Fraction of max SM clock below which a sample
            is considered throttled (default 0.75).
        throttle_cooldown_ms (float): Base post-throttle sleep; grows (capped at
            500 ms) while throttling persists, mirroring NVBench.
        max_consecutive_throttled (int): Give up screening after this many
            back-to-back throttled samples (avoids an infinite discard loop).
        input_args (tuple): Positional arguments to ``fn``.
        input_kwargs (dict, optional): Keyword arguments to ``fn``.
        cold_l2_cache (bool): Flush L2 before each sample (default True).
        return_samples (bool): If True (default) return ``(samples, stats)``;
            if False return just ``stats``.

    Returns:
        Tuple[List[float], BenchmarkStatistics] | BenchmarkStatistics:
        the per-sample GPU times (ms) and/or their summary. ``stats.median`` is
        the recommended central value; ``stats.noise`` is the coefficient of
        variation; ``stats.relative_iqr`` is the robust analog.

    Raises:
        RuntimeError: if CUPTI (cupti-python >= 13) is unavailable.

    Note:
        Timing is single-device (no cross-rank aggregation). A
        ``cudaDeviceSynchronize`` per iteration is required to delimit each
        sample's CUPTI window, so this is heavier per sample than the batched
        fixed-count path -- the cost of an adaptive procedure.

    Example:
        >>> samples, stats = bench_gpu_time_with_statistics(
        ...     fn=run_kernel, input_args=(x, y, out),
        ...     stopping_criterion="stdrel", throttle_screen=True,
        ... )
        >>> print(stats.summary_str())
    """
    cupti = _import_cupti_or_raise()

    if input_kwargs is None:
        input_kwargs = {}
    has_args = bool(input_args) or bool(input_kwargs)

    def call_fn():
        if has_args:
            fn(*input_args, **input_kwargs)
        else:
            fn()

    device = _infer_device_from_tensors(input_args, input_kwargs, "cuda")

    buffer = None
    if cold_l2_cache:
        # 2x L2 size to ensure a complete flush, matching the fixed-count path.
        l2_flush_size = get_l2_cache_size(device) * 2
        buffer = torch.empty(l2_flush_size, device=device, dtype=torch.int8)

    criterion = statistics.make_criterion(
        stopping_criterion,
        max_noise=max_noise,
        min_time=min_time_ms,
        target_samples=target_samples,
    )

    # --- CUPTI activity buffer callbacks (mirror bench_gpu_time_with_cupti) ---
    def func_buffer_requested():
        return 8 * 1024 * 1024, 0

    def kernel_activity_name(activity):
        if activity.kind == cupti.ActivityKind.CONCURRENT_KERNEL:
            return activity.name
        if activity.kind == cupti.ActivityKind.MEMCPY:
            return "MEMCPY"
        if activity.kind == cupti.ActivityKind.MEMSET:
            return "MEMSET"

    def activity_bytes(activity):
        if activity.kind in (cupti.ActivityKind.MEMCPY, cupti.ActivityKind.MEMSET):
            return activity.bytes
        return 0

    def activity_copy_kind(activity):
        return activity.copy_kind if activity.kind == cupti.ActivityKind.MEMCPY else 0

    def activity_value(activity):
        return activity.value if activity.kind == cupti.ActivityKind.MEMSET else 0

    def func_buffer_completed(launches, kernels, activities):
        for activity in activities:
            if activity.kind in (
                cupti.ActivityKind.CONCURRENT_KERNEL,
                cupti.ActivityKind.MEMCPY,
                cupti.ActivityKind.MEMSET,
            ):
                kernels.append(
                    (
                        kernel_activity_name(activity),
                        activity.start,
                        activity.end,
                        activity.correlation_id,
                        activity_copy_kind(activity),
                        activity_bytes(activity),
                        activity_value(activity),
                        activity.kind,
                    )
                )
            elif activity.kind in (
                cupti.ActivityKind.RUNTIME,
                cupti.ActivityKind.DRIVER,
            ):
                launches.append(
                    (
                        activity.start,
                        activity.end,
                        activity.correlation_id,
                        activity.cbid,
                        activity.kind,
                    )
                )

    # --- Prepare runner (direct call or CUDA graph replay) ---
    runner = call_fn
    if use_cuda_graph:
        torch.cuda.synchronize()
        capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(capture_stream):
            for _ in range(3):
                call_fn()
        torch.cuda.current_stream().wait_stream(capture_stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call_fn()
        runner = graph.replay

    # --- 5-run estimate (CUDA events) to size dry-run + batch ---
    torch.cuda.synchronize()
    call_fn()  # exclude one-time setup overhead
    torch.cuda.synchronize()
    est_start = torch.cuda.Event(enable_timing=True)
    est_end = torch.cuda.Event(enable_timing=True)
    est_start.record()
    for _ in range(5):
        if buffer is not None:
            buffer.zero_()
        runner()
    est_end.record()
    torch.cuda.synchronize()
    estimated_ms = est_start.elapsed_time(est_end) / 5

    if dry_run_iters is None:
        dry_run_iters = max(1, int(dry_run_time_ms / max(estimated_ms, 1e-9)))
    batch = 16 if batch_size is None else max(1, int(batch_size))

    torch.cuda.synchronize()
    for _ in range(dry_run_iters):
        if buffer is not None:
            buffer.zero_()
        runner()
    torch.cuda.synchronize()

    monitor = _ThrottleMonitor(device, throttle_threshold) if throttle_screen else None

    launches: list = []
    kernels: list = []
    iter_timestamps: list = []  # (start_cpu, end_cpu) per executed iteration
    samples: List[float] = []
    converted = 0  # number of iter_timestamps already turned into spans
    kernel_names = None
    consecutive_throttled = 0
    num_throttled = 0
    wall_start = time.perf_counter()
    timed_out = False
    stop = False

    cupti.activity_enable(cupti.ActivityKind.RUNTIME)
    cupti.activity_enable(cupti.ActivityKind.CONCURRENT_KERNEL)
    cupti.activity_enable(cupti.ActivityKind.DRIVER)
    cupti.activity_enable(cupti.ActivityKind.MEMCPY)
    cupti.activity_enable(cupti.ActivityKind.MEMSET)
    cupti.activity_register_callbacks(
        func_buffer_requested, partial(func_buffer_completed, launches, kernels)
    )
    try:
        while not stop:
            # Size this batch: never exceed max_samples; for sample-count, land
            # exactly on target_samples (preserves CI determinism).
            this_batch = min(batch, max_samples - len(samples))
            if isinstance(criterion, statistics.SampleCountCriterion):
                this_batch = min(this_batch, criterion.target_samples - len(samples))
            this_batch = max(1, this_batch)

            batch_discarded: List[bool] = []
            for _ in range(this_batch):
                if buffer is not None:
                    buffer.zero_()
                torch.cuda.synchronize()
                start_cpu = cupti.get_timestamp()
                runner()
                end_cpu = cupti.get_timestamp()
                # Best-effort throttle read while the kernel is likely still
                # running; reading post-sync risks catching the idle clock. This
                # is a host-side proxy -- NVBench reads the clock on-device
                # (%globaltimer + clock64) across the exact window.
                throttled = monitor.is_throttled() if monitor is not None else False
                torch.cuda.synchronize()
                iter_timestamps.append((start_cpu, end_cpu))
                batch_discarded.append(throttled)
                if throttled:
                    num_throttled += 1
                    consecutive_throttled += 1
                    # Cooldown grows while throttling persists (capped at 500 ms),
                    # letting the card recover -- NVBench's dynamic recovery delay.
                    cooldown = min(
                        throttle_cooldown_ms * max(1, consecutive_throttled - 2),
                        500.0,
                    )
                    time.sleep(cooldown / 1000.0)
                    if consecutive_throttled >= max_consecutive_throttled:
                        warnings.warn(
                            f"Throttle screen discarded {consecutive_throttled} "
                            "consecutive samples; GPU appears persistently "
                            "throttled. Reporting collected samples (if any).",
                            stacklevel=2,
                        )
                        stop = True
                        break
                else:
                    consecutive_throttled = 0

            # Drain CUPTI buffers and convert this batch's iterations to GPU
            # spans. new_spans aligns 1:1 with iter_timestamps[converted:], which
            # is exactly the iterations just executed (so with batch_discarded).
            cupti.activity_flush_all(0)
            new_spans, kernel_names = _cupti_kernel_spans(
                iter_timestamps,
                launches,
                kernels,
                start_index=converted,
                kernel_names=kernel_names,
            )
            converted = len(iter_timestamps)
            # new_spans aligns 1:1 with the iterations just executed; strict=True
            # asserts that invariant rather than silently truncating.
            for span, discarded in zip(new_spans, batch_discarded, strict=True):
                if discarded:
                    continue
                samples.append(span)
                criterion.add_measurement(span)

            n = len(samples)
            wall_ms = (time.perf_counter() - wall_start) * 1000.0
            if n >= min_samples and criterion.is_finished():
                stop = True
            elif n >= max_samples:
                warnings.warn(
                    f"Reached max_samples={max_samples} before the "
                    f"'{stopping_criterion}' criterion converged.",
                    stacklevel=2,
                )
                stop = True
            elif wall_ms >= max_time_ms:
                timed_out = True
                stop = True
    finally:
        cupti.activity_flush_all(0)
        cupti.activity_disable(cupti.ActivityKind.RUNTIME)
        cupti.activity_disable(cupti.ActivityKind.CONCURRENT_KERNEL)
        cupti.activity_disable(cupti.ActivityKind.DRIVER)
        cupti.activity_disable(cupti.ActivityKind.MEMCPY)
        cupti.activity_disable(cupti.ActivityKind.MEMSET)
        cupti.finalize()
        if monitor is not None:
            monitor.close()

    stats = BenchmarkStatistics.from_samples(samples)

    if timed_out:
        noise = "n/a" if stats.noise is None else f"{stats.noise * 100:.3f}%"
        if len(samples) < min_samples:
            warnings.warn(
                f"Benchmark timed out at max_time_ms={max_time_ms} with only "
                f"{len(samples)} samples (< min_samples={min_samples}); "
                "result is low-confidence.",
                stacklevel=2,
            )
        elif stopping_criterion == "stdrel" and (
            stats.noise is None or stats.noise >= max_noise
        ):
            warnings.warn(
                f"Benchmark timed out at max_time_ms={max_time_ms} before "
                f"converging: noise={noise} still above max_noise="
                f"{max_noise * 100:.3f}% after {len(samples)} samples.",
                stacklevel=2,
            )
    if num_throttled:
        warnings.warn(
            f"Throttle screen discarded {num_throttled} sample(s) during "
            "measurement; reported statistics exclude them.",
            stacklevel=2,
        )

    if return_samples:
        return samples, stats
    return stats

"""Shared utilities for PyTorch Distributed + NCCL examples."""

from __future__ import annotations

import os
from typing import Tuple

import torch
import torch.distributed as dist


def _read_required_int_env(name: str) -> int:
    """Read an integer environment variable injected by torchrun."""
    value = os.environ.get(name)
    if value is None:
        raise RuntimeError(
            f"Missing environment variable {name!r}. Launch this program with torchrun."
        )

    try:
        return int(value)
    except ValueError as exc:
        raise RuntimeError(
            f"Environment variable {name!r} must be an integer, got {value!r}."
        ) from exc


def setup_distributed() -> Tuple[int, int, int, torch.device]:
    """
    Initialize one NCCL process per GPU.

    Returns:
        rank: Global rank inside the process group.
        local_rank: Rank local to the current node, mapped to a CUDA device.
        world_size: Total number of processes in the group.
        device: CUDA device assigned to the current process.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for these NCCL examples.")

    if not dist.is_nccl_available():
        raise RuntimeError("This PyTorch build does not expose the NCCL backend.")

    rank = _read_required_int_env("RANK")
    local_rank = _read_required_int_env("LOCAL_RANK")
    world_size = _read_required_int_env("WORLD_SIZE")

    device_count = torch.cuda.device_count()
    if local_rank >= device_count:
        raise RuntimeError(
            f"LOCAL_RANK={local_rank}, but only {device_count} CUDA device(s) "
            "are visible to this process."
        )

    torch.cuda.set_device(local_rank)

    # torchrun provides the rendezvous environment variables expected by env://.
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
    )

    device = torch.device("cuda", local_rank)
    return rank, local_rank, world_size, device


def cleanup_distributed() -> None:
    """
    Destroy the current process group if it was initialized.

    We intentionally do not place a barrier here. A cleanup barrier can make
    failure recovery worse when another rank has already crashed.
    """
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def require_world_size(world_size: int, expected: int = 2) -> None:
    """Raise a clear error when a teaching example expects a fixed world size."""
    if world_size != expected:
        raise RuntimeError(
            f"This teaching example expects WORLD_SIZE={expected}, "
            f"but received WORLD_SIZE={world_size}."
        )


def ordered_print(
    rank: int,
    world_size: int,
    message: str,
    *,
    flush: bool = True,
) -> None:
    """
    Print one message per rank in rank order.

    This helper is intentionally synchronization-heavy. It is useful for
    teaching and readable logs, not as a pattern for performance-sensitive
    production code.
    """
    for current_rank in range(world_size):
        dist.barrier()
        if rank == current_rank:
            print(message, flush=flush)

    dist.barrier()


def tensor_to_list(tensor: torch.Tensor):
    """Move a tensor to CPU and convert it to a Python list for clean logging."""
    return tensor.detach().cpu().tolist()

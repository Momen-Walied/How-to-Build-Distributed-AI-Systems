"""Manual gradient synchronization demo used as a bridge to DDP."""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from .distributed_utils import ordered_print, require_world_size, tensor_to_list


def synchronize_gradients(model: nn.Module, world_size: int) -> None:
    """
    Average gradients across all ranks using AllReduce.

    This mirrors the core idea behind data-parallel gradient synchronization.
    Real DDP is more sophisticated: it uses gradient buckets and overlaps
    communication with backward computation.
    """
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.grad is None:
                continue

            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
            parameter.grad.div_(world_size)


def demo_manual_gradient_sync(rank: int, world_size: int, device: torch.device) -> None:
    """
    Compute different local gradients, synchronize them, then update identically.

    The demo uses equal local batch sizes on every rank. Therefore, averaging
    the local mean gradients by WORLD_SIZE gives the intended global average.
    """
    require_world_size(world_size, expected=2)

    model = nn.Linear(in_features=2, out_features=1, bias=False).to(device)

    # Make the initial model state explicitly identical on every rank.
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[0.5, -0.25]], device=device))

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    if rank == 0:
        features = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device=device)
        targets = torch.tensor([[1.0], [2.0]], device=device)
    else:
        features = torch.tensor([[2.0, 1.0], [1.0, 3.0]], device=device)
        targets = torch.tensor([[3.0], [5.0]], device=device)

    optimizer.zero_grad(set_to_none=True)

    predictions = model(features)
    loss = F.mse_loss(predictions, targets)
    loss.backward()

    ordered_print(
        rank,
        world_size,
        f"[Rank {rank}] INITIAL WEIGHT: {tensor_to_list(model.weight)}",
    )
    ordered_print(
        rank,
        world_size,
        f"[Rank {rank}] LOCAL LOSS={loss.item():.6f} | LOCAL GRAD={tensor_to_list(model.weight.grad)}",
    )

    synchronize_gradients(model=model, world_size=world_size)

    ordered_print(
        rank,
        world_size,
        f"[Rank {rank}] SYNCED GRAD: {tensor_to_list(model.weight.grad)}",
    )

    optimizer.step()

    ordered_print(
        rank,
        world_size,
        f"[Rank {rank}] UPDATED WEIGHT: {tensor_to_list(model.weight)}",
    )

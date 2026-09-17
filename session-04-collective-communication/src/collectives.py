"""Small, explicit collective communication demos."""

from __future__ import annotations

import time

import torch
import torch.distributed as dist

from .distributed_utils import ordered_print, require_world_size, tensor_to_list


def demo_process_identity(rank: int, local_rank: int, world_size: int, device: torch.device) -> None:
    """Show the mapping from process identity to CUDA device."""
    ordered_print(
        rank,
        world_size,
        f"Rank {rank} | LOCAL_RANK={local_rank} | WORLD_SIZE={world_size} | device={device}",
    )


def demo_barrier(rank: int, world_size: int) -> None:
    """Demonstrate barrier semantics by intentionally delaying Rank 1."""
    require_world_size(world_size, expected=2)

    dist.barrier()
    start = time.perf_counter()

    if rank == 1:
        time.sleep(2)

    reached_at = time.perf_counter() - start
    print(f"[Rank {rank}] reached barrier at ~{reached_at:.2f}s", flush=True)

    dist.barrier()

    passed_at = time.perf_counter() - start
    print(f"[Rank {rank}] passed barrier at  ~{passed_at:.2f}s", flush=True)


def demo_broadcast(rank: int, world_size: int, device: torch.device) -> None:
    """
    Broadcast a tensor from Rank 0 to every participating rank.

    Before:
        Rank 0 -> [10, 20]
        Rank 1 -> [ 0,  0]

    After:
        Rank 0 -> [10, 20]
        Rank 1 -> [10, 20]
    """
    require_world_size(world_size, expected=2)

    tensor = (
        torch.tensor([10.0, 20.0], device=device)
        if rank == 0
        else torch.tensor([0.0, 0.0], device=device)
    )

    ordered_print(rank, world_size, f"[Rank {rank}] BEFORE broadcast: {tensor_to_list(tensor)}")
    dist.broadcast(tensor, src=0)
    ordered_print(rank, world_size, f"[Rank {rank}] AFTER  broadcast: {tensor_to_list(tensor)}")


def demo_reduce(rank: int, world_size: int, device: torch.device) -> None:
    """
    Sum tensors from every rank and place the final result on Rank 0.

    Inputs:
        Rank 0 -> [1, 2]
        Rank 1 -> [3, 4]

    Semantic result on Rank 0:
        [4, 6]
    """
    require_world_size(world_size, expected=2)

    tensor = (
        torch.tensor([1.0, 2.0], device=device)
        if rank == 0
        else torch.tensor([3.0, 4.0], device=device)
    )

    ordered_print(rank, world_size, f"[Rank {rank}] BEFORE reduce: {tensor_to_list(tensor)}")
    dist.reduce(tensor, dst=0, op=dist.ReduceOp.SUM)

    if rank == 0:
        print(f"[Rank 0] REDUCE RESULT: {tensor_to_list(tensor)}", flush=True)

    dist.barrier()


def demo_all_reduce(rank: int, world_size: int, device: torch.device) -> None:
    """
    Sum tensors across all ranks and return the reduced result to every rank.

    Inputs:
        Rank 0 -> [1, 2]
        Rank 1 -> [3, 4]

    After:
        Rank 0 -> [4, 6]
        Rank 1 -> [4, 6]
    """
    require_world_size(world_size, expected=2)

    tensor = (
        torch.tensor([1.0, 2.0], device=device)
        if rank == 0
        else torch.tensor([3.0, 4.0], device=device)
    )

    ordered_print(rank, world_size, f"[Rank {rank}] BEFORE all_reduce: {tensor_to_list(tensor)}")
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    ordered_print(rank, world_size, f"[Rank {rank}] AFTER  all_reduce: {tensor_to_list(tensor)}")


def demo_all_gather(rank: int, world_size: int, device: torch.device) -> None:
    """Gather one local tensor from each rank and return all pieces to every rank."""
    local_tensor = torch.tensor([float((rank + 1) * 10)], device=device)
    gathered = [torch.zeros_like(local_tensor) for _ in range(world_size)]

    ordered_print(rank, world_size, f"[Rank {rank}] LOCAL INPUT: {tensor_to_list(local_tensor)}")
    dist.all_gather(gathered, local_tensor)

    result = [tensor_to_list(piece) for piece in gathered]
    ordered_print(rank, world_size, f"[Rank {rank}] GATHERED: {result}")


def demo_reduce_scatter(rank: int, world_size: int, device: torch.device) -> None:
    """
    Reduce element-wise across ranks, then scatter the reduced chunks.

    Inputs:
        Rank 0 -> [ 1,  2,  3,  4]
        Rank 1 -> [10, 20, 30, 40]

    Conceptual reduced tensor:
        [11, 22, 33, 44]

    Outputs:
        Rank 0 -> [11, 22]
        Rank 1 -> [33, 44]
    """
    require_world_size(world_size, expected=2)

    input_tensor = (
        torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)
        if rank == 0
        else torch.tensor([10.0, 20.0, 30.0, 40.0], device=device)
    )
    output_tensor = torch.empty(2, device=device)

    ordered_print(rank, world_size, f"[Rank {rank}] INPUT : {tensor_to_list(input_tensor)}")
    dist.reduce_scatter_tensor(output_tensor, input_tensor, op=dist.ReduceOp.SUM)
    ordered_print(rank, world_size, f"[Rank {rank}] OUTPUT: {tensor_to_list(output_tensor)}")


def demo_all_to_all(rank: int, world_size: int, device: torch.device) -> None:
    """
    Route a different piece from each sender to each destination rank.

    Inputs:
        Rank 0 -> [10, 11]
        Rank 1 -> [20, 21]

    Outputs:
        Rank 0 -> [10, 20]
        Rank 1 -> [11, 21]
    """
    require_world_size(world_size, expected=2)

    input_tensor = (
        torch.tensor([10.0, 11.0], device=device)
        if rank == 0
        else torch.tensor([20.0, 21.0], device=device)
    )
    output_tensor = torch.empty_like(input_tensor)

    ordered_print(rank, world_size, f"[Rank {rank}] INPUT : {tensor_to_list(input_tensor)}")
    dist.all_to_all_single(output_tensor, input_tensor)
    ordered_print(rank, world_size, f"[Rank {rank}] OUTPUT: {tensor_to_list(output_tensor)}")

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse

from src.collectives import (
    demo_all_gather,
    demo_all_reduce,
    demo_all_to_all,
    demo_barrier,
    demo_broadcast,
    demo_process_identity,
    demo_reduce,
    demo_reduce_scatter,
)
from src.distributed_utils import cleanup_distributed, setup_distributed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one Session 4 collective communication demo."
    )
    parser.add_argument(
        "--demo",
        required=True,
        choices=[
            "identity",
            "barrier",
            "broadcast",
            "reduce",
            "all_reduce",
            "all_gather",
            "reduce_scatter",
            "all_to_all",
        ],
        help="Collective demo to run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rank, local_rank, world_size, device = setup_distributed()

    try:
        if args.demo == "identity":
            demo_process_identity(rank, local_rank, world_size, device)
        elif args.demo == "barrier":
            demo_barrier(rank, world_size)
        elif args.demo == "broadcast":
            demo_broadcast(rank, world_size, device)
        elif args.demo == "reduce":
            demo_reduce(rank, world_size, device)
        elif args.demo == "all_reduce":
            demo_all_reduce(rank, world_size, device)
        elif args.demo == "all_gather":
            demo_all_gather(rank, world_size, device)
        elif args.demo == "reduce_scatter":
            demo_reduce_scatter(rank, world_size, device)
        elif args.demo == "all_to_all":
            demo_all_to_all(rank, world_size, device)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.collectives import demo_reduce
from src.distributed_utils import cleanup_distributed, setup_distributed


def main() -> None:
    rank, local_rank, world_size, device = setup_distributed()

    try:
        demo_reduce(rank, world_size, device)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()

"""Keep slow rank-zero filesystem/validation work off NCCL collectives."""

from contextlib import contextmanager
from datetime import timedelta

import torch.distributed as dist


def make_control_group(world):
    # A larger CPU control timeout is separate from the training collectives'
    # timeout. Peers must not enqueue a GPU all-reduce while rank zero saves.
    return dist.new_group(backend="gloo", timeout=timedelta(hours=1)) if world > 1 else None


@contextmanager
def rank_zero_section(rank, group):
    failure = None
    try:
        yield
    except BaseException as error:
        failure = error
        raise
    finally:
        if group is not None:
            status = [f"{type(failure).__name__}: {failure}" if failure is not None else None]
            dist.broadcast_object_list(status, src=0, group=group)
            if status[0] is not None and failure is None:
                raise RuntimeError(f"rank-zero work failed: {status[0]}")

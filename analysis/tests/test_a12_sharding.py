"""How a12 splits its repeats across workers.

A cell's cost depends on the arm, outcome and tier config, not on the repeat
index, so equal repeat ranges are equal work. Splitting by outcome would leave
most workers idle.
"""

import pytest

from analysis.pipelines.a12_delta_r2_ci import shard_ranges


@pytest.mark.parametrize("repeats,shards", [(15, 1), (15, 3), (15, 5), (15, 12), (15, 15)])
def test_every_repeat_is_fitted_exactly_once(repeats, shards):
    covered = [r for lo, hi in shard_ranges(repeats, shards) for r in range(lo, hi)]
    assert covered == list(range(repeats))


def test_ranges_differ_by_at_most_one_repeat(repeats=15, shards=12):
    sizes = [hi - lo for lo, hi in shard_ranges(repeats, shards)]
    assert max(sizes) - min(sizes) <= 1


def test_more_shards_than_repeats_makes_no_empty_worker():
    """An empty worker would load the master, fit nothing and write a header."""
    ranges = shard_ranges(4, 12)
    assert ranges == [(0, 1), (1, 2), (2, 3), (3, 4)]


def test_one_shard_is_the_whole_range():
    assert shard_ranges(15, 1) == [(0, 15)]


def test_merge_only_does_not_register_the_age_hook():
    """A merge pass fits nothing, so it has no age ledger of its own.

    Registering the hook would overwrite the fitting run's ledger with the
    "no load_master call in this run" placeholder.
    """
    import inspect

    from analysis.pipelines import a12_delta_r2_ci as a12

    src = inspect.getsource(a12.main)
    emit = src.index("age_method.emit_at_exit")
    guard = src.rindex("if not args.merge_only:", 0, emit)
    # the guard is the line immediately above the call
    assert src[guard:emit].count("\n") == 1

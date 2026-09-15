from __future__ import annotations

from sentrysem.experiments.validation import selection_bias_experiment, sequential_null_experiment


def test_selection_bias_counts_reproduce_paper() -> None:
    rows = selection_bias_experiment(0.7717004776000975)
    assert [row["naive_reuse_false_accepts"] for row in rows] == [47, 414, 2515]
    assert [row["fresh_false_accepts"] for row in rows] == [57, 50, 45]


def test_sequential_null_counts_reproduce_paper() -> None:
    rows = sequential_null_experiment(0.7717004776000975)
    assert [row["familywise_false_accepts"] for row in rows] == [44, 52]

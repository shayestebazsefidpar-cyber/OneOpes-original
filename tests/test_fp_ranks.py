"""FP ranking of the candidate atoms."""

from protein_waterfp.convergence import fp_ranks


def test_rank_one_is_most_negative_fp():
    ranks = fp_ranks({"x": -3.0, "y": -40.0, "z": 2.0}, ["x", "y", "z"])
    assert ranks.to_dict() == {"x": 2.0, "y": 1.0, "z": 3.0}

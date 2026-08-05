"""Tests for bundled seven-history and learned result records."""

from deja_cue.reference_results import validate_reference_results


def test_reference_result_statistics_are_self_consistent() -> None:
    checks = validate_reference_results()
    assert checks["seven_history"] == {
        "histories": 7,
        "states": 16,
        "descriptions": 32,
        "episodes": 59,
    }
    assert checks["learned"] == {
        "models": 8,
        "seeds_per_model": 3,
        "comparisons": 16,
    }
    assert checks["vocabulary"] == {
        "table_rows": 4,
        "duplication_factors": 3,
    }

"""Regression checks for seven-history identifier normalization."""

from deja_cue.seven_history import prediction_window_map


def test_prediction_map_uses_reference_scene_identifiers() -> None:
    condition = {
        "histories": [
            {
                "history_id": "S01",
                "queries": [
                    {
                        "state_id": "state_a",
                        "text": "example description",
                        "window": [11, 23],
                    }
                ],
            }
        ]
    }

    assert prediction_window_map(condition, {"S01": "example_scene"}) == {
        ("example_scene", "state_a", "example description"): (11, 23)
    }

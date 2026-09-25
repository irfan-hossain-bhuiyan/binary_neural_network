"""I13 schedule and provenance tests."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from run_i13_loss_curriculum import EXPECTED_INITIAL_HASHES, curriculum_alpha


def test_curriculum_alpha_schedule():
    assert curriculum_alpha(100, 500) == (1.5, "POWER_1_5")
    alpha, phase = curriculum_alpha(750, 500)
    assert abs(alpha - 1.375) < 1e-12 and phase == "ANNEAL"
    assert curriculum_alpha(1000, 500) == (1.25, "POWER_1_25")
    assert curriculum_alpha(3000, None) == (1.5, "POWER_1_5")


def test_i12_initial_hashes_are_explicit():
    assert len(EXPECTED_INITIAL_HASHES) == 5
    assert all(len(value) == 64 for value in EXPECTED_INITIAL_HASHES.values())

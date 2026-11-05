import numpy as np
import pytest

from accountpulse.resampling import grouped_row_positions


@pytest.mark.parametrize("count", [0, 1, 17, 1003])
def test_group_positions_preserve_order_and_multiplicity(count):
    keys = np.resize(np.array(["b", "a", "nan", "None", "b"]), count)
    grouped = grouped_row_positions(keys)
    expected_keys = list(dict.fromkeys(keys))
    assert list(grouped) == expected_keys
    for key in expected_keys:
        np.testing.assert_array_equal(grouped[key], np.flatnonzero(keys == key))


def test_repeated_cluster_selection_preserves_repeated_rows():
    grouped = grouped_row_positions(np.array(["b", "a", "b", "c"]))
    np.testing.assert_array_equal(
        np.concatenate([grouped[key] for key in ["b", "a", "b"]]), [0, 2, 1, 0, 2]
    )

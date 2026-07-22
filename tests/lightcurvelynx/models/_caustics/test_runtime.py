import numpy as np
import pytest

from lightcurvelynx.models._caustics import runtime


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        (2, 2.0),
        (0.25, 0.25),
        (np.array(0.75), 0.75),
        ("0.5", 0.5),
    ],
)
def test_validate_optional_positive_fraction_accepts_scalar_values(value, expected):
    """Normalize every supported optional scalar representation."""
    result = runtime._validate_optional_positive_fraction("fraction", value)
    assert result == expected


@pytest.mark.parametrize("value", [np.array([0.5]), "not-a-number", object()])
def test_validate_optional_positive_fraction_rejects_non_scalars(value):
    """Reject arrays and objects that cannot represent one floating scalar."""
    with pytest.raises(TypeError, match="None or a scalar value convertible to float"):
        runtime._validate_optional_positive_fraction("fraction", value)


@pytest.mark.parametrize("value", [0.0, -0.1, np.nan, np.inf, -np.inf])
def test_validate_optional_positive_fraction_rejects_non_positive_or_non_finite(value):
    """Reject normalized fractions outside the finite positive domain."""
    with pytest.raises(ValueError, match="fraction must be finite and positive"):
        runtime._validate_optional_positive_fraction("fraction", value)


def test_sample_value_preserves_single_sample_and_indexes_multiple_samples():
    """Keep a single-system object intact and index only multi-sample values."""
    single = np.array([1.0, 2.0])
    first = object()
    second = object()

    assert runtime._sample_value(single, 0, 1) is single
    assert runtime._sample_value([first, second], 1, 2) is second
    np.testing.assert_array_equal(
        runtime._sample_value(np.array([[1.0, 2.0], [3.0, 4.0]]), 1, 2),
        [3.0, 4.0],
    )


def test_to_numpy_detaches_moves_to_cpu_then_converts():
    """Detach and transfer a tensor before requesting its NumPy view."""
    events = []

    class RecordingTensor:
        def detach(self):
            events.append("detach")
            return self

        def cpu(self):
            events.append("cpu")
            return self

        def numpy(self):
            events.append("numpy")
            return np.array([1.0, 2.0])

    np.testing.assert_array_equal(runtime._to_numpy(RecordingTensor()), [1.0, 2.0])
    assert events == ["detach", "cpu", "numpy"]

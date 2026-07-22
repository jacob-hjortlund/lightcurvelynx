import sys
from types import SimpleNamespace

import numpy as np
import pytest

from lightcurvelynx.models._caustics import image_recovery, runtime


class _FakeTensor:
    """Provide the tensor-to-NumPy protocol used by recovery certification."""

    def __init__(self, values):
        self.values = np.asarray(values, dtype=float)

    def detach(self):
        """Return this already detached tensor double."""
        return self

    def cpu(self):
        """Return this already CPU-resident tensor double."""
        return self

    def numpy(self):
        """Return the stored NumPy representation."""
        return self.values

    def __getitem__(self, key):
        """Return a tensor double for indexed root-refinement values."""
        return _FakeTensor(self.values[key])


class _FakeTorch:
    """Provide tensor conversion and absolute-value operations under test."""

    float64 = np.float64

    @staticmethod
    def as_tensor(values, dtype=None):
        """Convert values to the minimal tensor double."""
        del dtype
        return _FakeTensor(values)

    @staticmethod
    def abs(values):
        """Return elementwise absolute values through the tensor protocol."""
        return _FakeTensor(np.abs(values.values))


class _MappedRecoveryLens:
    """Return predetermined source positions for candidate recovery roots."""

    def __init__(self, mapped_x, mapped_y):
        self.mapped_x = mapped_x
        self.mapped_y = mapped_y

    def raytrace(self, image_x, image_y):
        """Return the configured mapped positions in candidate order."""
        del image_x, image_y
        return _FakeTensor(self.mapped_x), _FakeTensor(self.mapped_y)


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (RuntimeError("torch.linalg.solve: input matrix is singular"), True),
        (RuntimeError("LINALG.SOLVE failed with SINGULAR U"), True),
        (ValueError("linalg.solve: input matrix is singular"), False),
        (RuntimeError("input matrix is singular"), False),
        (RuntimeError("linalg.solve failed"), False),
    ],
)
def test_singular_forward_raytrace_error_classifier_is_exact(error, expected):
    """Recognize only the two documented singular linear-solve messages."""
    assert image_recovery._is_singular_forward_raytrace_error(error) is expected


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (IndexError("index 0 is out of bounds for dimension 0"), True),
        (IndexError("INDEX 0 IS OUT OF BOUNDS"), True),
        (RuntimeError("index 0 is out of bounds"), False),
        (IndexError("index 1 is out of bounds"), False),
        (IndexError("index 0 was outside bounds"), False),
    ],
)
def test_retryable_forward_raytrace_error_classifier_is_exact(error, expected):
    """Recognize only the documented empty-candidate IndexError."""
    assert image_recovery._is_retryable_forward_raytrace_error(error) is expected


def test_recovery_neighborhoods_handle_empty_and_occupied_inputs():
    """Mark recovery neighborhoods empty only when no image lies within radius."""
    recovery_points = np.array([[0.0, 0.0], [2.0, 0.0]])
    np.testing.assert_array_equal(
        image_recovery._recovery_neighborhoods_are_empty(
            np.empty((0, 2)),
            recovery_points,
            1.0,
        ),
        [True, True],
    )
    assert image_recovery._recovery_neighborhoods_are_empty(
        np.array([[0.0, 0.0]]),
        np.empty((0, 2)),
        1.0,
    ).shape == (0,)

    np.testing.assert_array_equal(
        image_recovery._recovery_neighborhoods_are_empty(
            np.array([[0.0, 0.0], [3.0, 0.0]]),
            np.array([[0.5, 0.0], [1.0, 0.0], [2.0, 0.0], [5.0, 0.0]]),
            1.0,
        ),
        [False, False, False, True],
    )


def test_recovery_image_seeds_selects_nearest_circle_point_per_recovery_point(
    monkeypatch,
):
    """Choose each seed from its own mapped circle with no cross-point coupling."""
    recovery_points = np.array([[1.0, 1.0], [4.0, -2.0]])
    source_position = np.array([2.0, -3.0])
    selected_indices = (64, 128)
    traced_circles = []

    def raytrace_curve(lens, circle):
        assert lens is fake_lens
        traced_circles.append(circle.copy())
        mapped = np.full((256, 2), 100.0)
        mapped[selected_indices[len(traced_circles) - 1]] = source_position
        return mapped

    fake_lens = object()
    monkeypatch.setattr(runtime, "_raytrace_curve", raytrace_curve)

    empty = image_recovery._recovery_image_seeds(
        fake_lens,
        (),
        source_x=source_position[0],
        source_y=source_position[1],
        radius=0.5,
    )
    seeds = image_recovery._recovery_image_seeds(
        fake_lens,
        recovery_points,
        source_x=source_position[0],
        source_y=source_position[1],
        radius=0.5,
    )

    assert empty.shape == (0, 2)
    assert len(traced_circles) == 2
    for circle, center in zip(traced_circles, recovery_points, strict=True):
        assert circle.shape == (256, 2)
        np.testing.assert_allclose(np.linalg.norm(circle - center, axis=1), 0.5)
    np.testing.assert_allclose(seeds, [[1.0, 1.5], [3.5, -2.0]], atol=1.0e-15)


def test_refine_image_seeds_runs_eight_passes_then_hands_roots_to_certification(
    monkeypatch,
):
    """Preserve paired roots through the fixed refinement count and handoff."""
    rootfind_calls = []
    raytrace = object()
    lens = SimpleNamespace(raytrace=raytrace)
    beta_x = _FakeTensor(0.25)
    beta_y = _FakeTensor(-0.5)
    recovery_points = np.array([[0.0, 0.0], [10.0, 10.0]])

    def rootfind(image_x, image_y, passed_beta_x, passed_beta_y, passed_raytrace):
        rootfind_calls.append((image_x.values.copy(), image_y.values.copy()))
        assert passed_beta_x is beta_x
        assert passed_beta_y is beta_y
        assert passed_raytrace is raytrace
        return _FakeTensor(np.column_stack((image_x.values + 1.0, image_y.values - 1.0)))

    monkeypatch.setitem(
        sys.modules,
        "caustics.lenses.func",
        SimpleNamespace(forward_raytrace_rootfind=rootfind),
    )
    certification_calls = []
    certified = np.array([[9.0, -6.0]])

    def certify(*args):
        certification_calls.append(args)
        return certified

    monkeypatch.setattr(image_recovery, "_validated_recovery_images", certify)

    result = image_recovery._refine_image_seeds(
        lens,
        _FakeTorch,
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        recovery_points,
        beta_x,
        beta_y,
        0.01,
        0.2,
    )

    assert image_recovery._RECOVERY_ROOT_REFINEMENTS == 8
    assert len(rootfind_calls) == 8
    np.testing.assert_array_equal(rootfind_calls[0][0], [1.0, 3.0])
    np.testing.assert_array_equal(rootfind_calls[0][1], [2.0, 4.0])
    np.testing.assert_array_equal(rootfind_calls[-1][0], [8.0, 10.0])
    np.testing.assert_array_equal(rootfind_calls[-1][1], [-5.0, -3.0])
    assert len(certification_calls) == 1
    handoff = certification_calls[0]
    assert handoff[0] is lens
    assert handoff[1] is _FakeTorch
    np.testing.assert_array_equal(handoff[2].values, [9.0, 11.0])
    np.testing.assert_array_equal(handoff[3].values, [-6.0, -4.0])
    assert handoff[4] is recovery_points
    assert handoff[5] is beta_x
    assert handoff[6] is beta_y
    assert handoff[7:] == (0.01, 0.2)
    assert result is certified


def test_validated_recovery_images_applies_strict_residual_and_inclusive_locality():
    """Retain ordered duplicates only when both recovery certifications pass."""
    lens = _MappedRecoveryLens(
        mapped_x=[0.09, 0.1, 0.05, 0.0, 0.05],
        mapped_y=[0.0, 0.0, 0.0, 0.0, 0.0],
    )
    image_x = _FakeTensor([0.0, 1.0, 2.0, 3.0, 2.0])
    image_y = _FakeTensor([0.0, 0.0, 0.0, 0.0, 0.0])
    recovery_points = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.5, 0.0],
            [3.5001, 0.0],
            [2.5, 0.0],
        ]
    )

    certified = image_recovery._validated_recovery_images(
        lens,
        _FakeTorch,
        image_x,
        image_y,
        recovery_points,
        _FakeTensor(0.0),
        _FakeTensor(0.0),
        epsilon=0.1,
        neighborhood_radius=0.5,
    )

    np.testing.assert_array_equal(
        certified,
        [[0.0, 0.0], [2.0, 0.0], [2.0, 0.0]],
    )

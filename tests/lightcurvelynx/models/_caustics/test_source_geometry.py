from types import SimpleNamespace

import numpy as np
import pytest

from lightcurvelynx.models._caustics import lens_system, runtime, source_geometry


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


class _RecordingParameter:
    """Record dtype staticization for one fake Caustics parameter."""

    def __init__(self, value, *, events=None, owner=None, name=None):
        self.value = _FakeTensor(value)
        self.dtype_calls = []
        self.events = events
        self.owner = owner
        self.name = name

    def to(self, *, dtype):
        """Record the requested dtype and preserve parameter identity."""
        self.dtype_calls.append(dtype)
        if self.events is not None:
            self.events.append(("parameter_to", self.owner, self.name, dtype))
        return self


def _closed_square(center=(0.0, 0.0), half_width=1.0):
    """Return one counterclockwise producer-closed square."""
    center_x, center_y = center
    return np.array(
        [
            [center_x - half_width, center_y - half_width],
            [center_x + half_width, center_y - half_width],
            [center_x + half_width, center_y + half_width],
            [center_x - half_width, center_y + half_width],
            [center_x - half_width, center_y - half_width],
        ],
        dtype=float,
    )


def test_trace_pseudo_caustics_caps_initial_radius_halves_to_convergence_and_closes(
    monkeypatch,
):
    """Apply the generator cap, refine by halves, and repeat the final vertex."""
    values = object()
    lens = object()
    center = np.array([1.0, -2.0])
    traced_loops = []

    class GeneratorAdapter:
        def pseudo_caustic_generators(self, passed_values):
            assert passed_values is values
            return (
                lens_system._PseudoCausticGenerator(
                    center=tuple(center),
                    max_initial_radius=0.4,
                ),
            )

    def raytrace_curve(passed_lens, coordinates):
        assert passed_lens is lens
        traced_loops.append(coordinates.copy())
        return coordinates.copy()

    monkeypatch.setattr(runtime, "_raytrace_curve", raytrace_curve)

    (curve,) = source_geometry._trace_pseudo_caustics(
        lens,
        GeneratorAdapter(),
        values,
        num_points=4,
        epsilon=1.0,
        geometry_tolerance=0.11,
    )

    assert len(traced_loops) == 3
    np.testing.assert_allclose(
        [np.linalg.norm(loop[0] - center) for loop in traced_loops],
        [0.4, 0.2, 0.1],
    )
    np.testing.assert_allclose(
        curve,
        [
            [1.1, -2.0],
            [1.0, -1.9],
            [0.9, -2.0],
            [1.0, -2.1],
            [1.1, -2.0],
        ],
        atol=1.0e-15,
    )


def test_trace_pseudo_caustics_reports_bounded_halving_exhaustion(monkeypatch):
    """Stop after exactly 32 failed refinements and report final movement."""
    lens = object()
    traced_loops = []
    adapter = SimpleNamespace(
        pseudo_caustic_generators=lambda values: (lens_system._PseudoCausticGenerator(center=(0.0, 0.0)),)
    )

    def nonconverging_raytrace(passed_lens, coordinates):
        assert passed_lens is lens
        traced_loops.append(coordinates.copy())
        return coordinates + np.array([len(traced_loops), 0.0])

    monkeypatch.setattr(
        runtime,
        "_raytrace_curve",
        nonconverging_raytrace,
    )

    with pytest.raises(RuntimeError) as error:
        source_geometry._trace_pseudo_caustics(
            lens,
            adapter,
            {},
            num_points=4,
            epsilon=0.8,
            geometry_tolerance=0.01,
        )

    assert len(traced_loops) == 33
    assert np.linalg.norm(traced_loops[0][0]) == pytest.approx(0.8)
    assert np.linalg.norm(traced_loops[-1][0]) == pytest.approx(0.8 / 2**32)
    message = str(error.value)
    assert "did not converge after 32 refinements" in message
    assert "final boundary change" in message


def test_outer_grid_boundary_uses_rows_then_side_interiors():
    """Return every square-grid boundary value once in documented order."""
    grid = np.arange(9).reshape(3, 3)
    np.testing.assert_array_equal(
        source_geometry._outer_grid_boundary(grid),
        [0, 1, 2, 6, 7, 8, 3, 5],
    )


def test_close_curve_removes_near_duplicates_and_recloses_exactly():
    """Remove consecutive and cyclic near-duplicates before exact closure."""
    curve = np.array(
        [
            [0.0, 0.0],
            [0.0001, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.0, 0.0001],
            [0.0, 0.0],
        ]
    )
    expected = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ]
    )

    np.testing.assert_array_equal(
        source_geometry._close_curve(curve, tolerance=0.001),
        expected,
    )
    with pytest.raises(RuntimeError, match="fewer than three unique vertices"):
        source_geometry._close_curve(
            np.array([[0.0, 0.0], [0.0001, 0.0], [0.0, 0.0001], [0.0, 0.0]]),
            tolerance=0.001,
        )


def test_resample_closed_curve_returns_equal_arclength_square_points():
    """Sample a square at fixed half-edge arclength increments."""
    expected = np.array(
        [
            [-1.0, -1.0],
            [0.0, -1.0],
            [1.0, -1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [-1.0, 1.0],
            [-1.0, 0.0],
        ]
    )
    np.testing.assert_allclose(
        source_geometry._resample_closed_curve(_closed_square(), num_points=8),
        expected,
    )


def test_curve_orientation_detects_reversal_after_translation_and_cyclic_shift():
    """Ignore start point and translation while detecting traversal reversal."""
    reference = _closed_square()
    shifted_vertices = np.roll(reference[:-1], 2, axis=0) + np.array([3.0, -2.0])
    shifted = np.concatenate((shifted_vertices, shifted_vertices[:1]), axis=0)

    assert source_geometry._curve_orientation_is_preserved(reference, shifted) is True
    assert source_geometry._curve_orientation_is_preserved(reference, shifted[::-1]) is False


def test_curve_center_diameter_and_raw_hausdorff_use_vertex_geometry():
    """Measure bounding-box geometry and bidirectional nearest-vertex distance."""
    rectangle = np.array([[2.0, -1.0], [6.0, -1.0], [6.0, 2.0], [2.0, 2.0], [2.0, -1.0]])
    center, diameter = source_geometry._curve_center_and_diameter(rectangle)

    np.testing.assert_array_equal(center, [4.0, 0.5])
    assert diameter == 5.0
    assert source_geometry._raw_curve_hausdorff_distance(
        _closed_square(),
        _closed_square(center=(3.0, 0.0)),
    ) == pytest.approx(3.0)


def test_match_raw_caustic_curves_handles_empty_unequal_and_reordered_sets():
    """Return count sentinels and reorder equal sets by minimum displacement."""
    assert source_geometry._match_raw_caustic_curves((), ()) == ((), True)

    unmatched = _closed_square(center=(20.0, 0.0))
    candidates, counts_match = source_geometry._match_raw_caustic_curves(
        (_closed_square(),),
        (unmatched, _closed_square(center=(40.0, 0.0))),
    )
    assert candidates[0] is unmatched
    assert counts_match is False

    reference_first = _closed_square()
    reference_second = _closed_square(center=(10.0, 0.0))
    candidate_first = reference_first.copy()
    candidate_second = reference_second.copy()
    ordered, counts_match = source_geometry._match_raw_caustic_curves(
        (reference_first, reference_second),
        (candidate_second, candidate_first),
    )
    assert ordered[0] is candidate_first
    assert ordered[1] is candidate_second
    assert counts_match is True


def test_partition_axisymmetric_point_caustics_returns_count_mismatch_sentinel():
    """Return ``None`` when either consecutive curve count differs."""
    assert (
        source_geometry._partition_axisymmetric_point_caustics(
            (_closed_square(), _closed_square(center=(10.0, 0.0))),
            (_closed_square(),),
            (_closed_square(),),
            boundary_tolerance=0.1,
        )
        is None
    )


def test_partition_axisymmetric_point_caustics_extracts_contracting_point():
    """Classify a centered factor-of-two contraction as a point caustic."""
    older = _closed_square(half_width=4.0)
    previous = _closed_square(center=(0.05, 0.0), half_width=2.0)
    current = _closed_square(center=(0.08, 0.0), half_width=1.0)

    partition = source_geometry._partition_axisymmetric_point_caustics(
        (older,),
        (previous,),
        (current,),
        boundary_tolerance=0.1,
    )

    assert partition.previous_curves == ()
    assert partition.current_curves == ()
    assert len(partition.previous_points) == 1
    assert len(partition.current_points) == 1
    np.testing.assert_allclose(partition.previous_points[0], [0.05, 0.0])
    np.testing.assert_allclose(partition.current_points[0], [0.08, 0.0])


def test_partition_axisymmetric_point_caustics_retains_noncontracting_curve():
    """Keep a curve regular when its diameter does not contract fast enough."""
    older = _closed_square(half_width=1.0)
    previous = _closed_square(half_width=0.75)
    current = _closed_square(half_width=0.5)

    partition = source_geometry._partition_axisymmetric_point_caustics(
        (older,),
        (previous,),
        (current,),
        boundary_tolerance=0.1,
    )

    assert partition.previous_curves[0] is previous
    assert partition.current_curves[0] is current
    assert partition.previous_points == ()
    assert partition.current_points == ()


def test_boundary_regions_repairs_invalid_curves_and_rejects_zero_area():
    """Repair a bow-tie while rejecting a collinear non-area boundary."""
    pytest.importorskip("shapely")
    bow_tie = np.array([[0.0, 0.0], [2.0, 2.0], [0.0, 2.0], [2.0, 0.0], [0.0, 0.0]])

    (region,) = source_geometry._boundary_regions(
        (bow_tie,),
        geometry_tolerance=1.0e-6,
    )

    assert region.is_valid
    assert not region.is_empty
    assert np.isfinite(float(region.area))
    assert float(region.area) > 0.0

    collinear = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 0.0]])
    with pytest.raises(
        RuntimeError,
        match="Caustic boundary 0 did not enclose a finite positive-area polygonal region",
    ):
        source_geometry._boundary_regions(
            (collinear,),
            geometry_tolerance=1.0e-6,
        )


def test_match_boundary_curves_reports_assignment_displacement_counts_and_orientation():
    """Match geometry by distance while retaining typed count and direction flags."""
    pytest.importorskip("shapely")
    assert source_geometry._match_boundary_curves(
        (),
        (),
        geometry_tolerance=1.0e-6,
    ) == ((), 0.0, True, True)

    reference_first = _closed_square()
    reference_second = _closed_square(center=(10.0, 0.0))
    candidate_first = reference_first + np.array([0.1, 0.0])
    candidate_second = reference_second + np.array([0.0, 0.25])
    ordered, displacement, counts_stable, orientation_stable = source_geometry._match_boundary_curves(
        (reference_first, reference_second),
        (candidate_second, candidate_first),
        geometry_tolerance=1.0e-6,
    )

    assert ordered[0] is candidate_first
    assert ordered[1] is candidate_second
    assert displacement == pytest.approx(0.25)
    assert counts_stable is True
    assert orientation_stable is True

    reversed_ordered, reversed_displacement, counts_stable, orientation_stable = (
        source_geometry._match_boundary_curves(
            (reference_first,),
            (reference_first[::-1],),
            geometry_tolerance=1.0e-6,
        )
    )
    np.testing.assert_array_equal(reversed_ordered[0], reference_first[::-1])
    assert reversed_displacement == pytest.approx(0.0)
    assert counts_stable is True
    assert orientation_stable is False

    unmatched = _closed_square(center=(20.0, 0.0))
    candidates, displacement, counts_stable, orientation_stable = source_geometry._match_boundary_curves(
        (reference_first,),
        (candidate_first, unmatched),
        geometry_tolerance=1.0e-6,
    )
    assert candidates[0] is candidate_first
    assert candidates[1] is unmatched
    assert np.isinf(displacement)
    assert counts_stable is False
    assert orientation_stable is False


def test_boundary_topology_signature_is_stable_under_translation_and_point_changes():
    """Retain typed containment topology while ignoring positions and points."""
    pytest.importorskip("shapely")
    true_curve = _closed_square(half_width=2.0)
    pseudo_curve = _closed_square(half_width=0.5)
    geometry = source_geometry._BoundaryGeometry(
        caustic_curves=(true_curve,),
        pseudo_caustic_curves=(pseudo_curve,),
        critical_curve_fov=4.0,
        pixelscale=0.1,
        pseudo_caustic_points=64,
        point_caustics=(np.array([0.0, 0.0]),),
    )
    translated = source_geometry._BoundaryGeometry(
        caustic_curves=(true_curve + np.array([7.0, -3.0]),),
        pseudo_caustic_curves=(pseudo_curve + np.array([7.0, -3.0]),),
        critical_curve_fov=8.0,
        pixelscale=0.05,
        pseudo_caustic_points=128,
        point_caustics=(np.array([100.0, 100.0]),),
    )

    signature = source_geometry._boundary_topology_signature(
        geometry,
        geometry_tolerance=1.0e-6,
    )

    assert signature == (
        (((1, 0),), ((1, 0),)),
        ((False, False, True, False, False),),
    )
    assert signature == source_geometry._boundary_topology_signature(
        translated,
        geometry_tolerance=1.0e-6,
    )


def test_compare_boundary_geometry_reorders_stable_topology_and_preserves_metadata():
    """Reorder matched curves while forwarding the current snapshot metadata."""
    pytest.importorskip("shapely")
    reference_first = _closed_square()
    reference_second = _closed_square(center=(10.0, 0.0))
    candidate_first = reference_first + np.array([0.1, 0.0])
    candidate_second = reference_second + np.array([0.0, 0.25])
    previous = source_geometry._BoundaryGeometry(
        caustic_curves=(reference_first, reference_second),
        pseudo_caustic_curves=(),
        critical_curve_fov=4.0,
        pixelscale=0.1,
        pseudo_caustic_points=64,
    )
    point_caustic = np.array([0.0, 0.0])
    current = source_geometry._BoundaryGeometry(
        caustic_curves=(candidate_second, candidate_first),
        pseudo_caustic_curves=(),
        critical_curve_fov=8.0,
        pixelscale=0.05,
        pseudo_caustic_points=128,
        point_caustics=(point_caustic,),
    )

    reordered, displacement, topology_stable = source_geometry._compare_boundary_geometry(
        previous,
        current,
        1.0e-6,
    )

    assert reordered.caustic_curves[0] is candidate_first
    assert reordered.caustic_curves[1] is candidate_second
    assert reordered.critical_curve_fov == 8.0
    assert reordered.pixelscale == 0.05
    assert reordered.pseudo_caustic_points == 128
    assert reordered.point_caustics[0] is point_caustic
    assert displacement == pytest.approx(0.25)
    assert topology_stable is True


def test_build_strong_lensing_region_unions_boundaries_and_rejects_empty_input():
    """Union overlapping typed interiors into one finite positive-area region."""
    pytest.importorskip("shapely")
    left = _closed_square(center=(1.0, 1.0), half_width=1.0)
    right = _closed_square(center=(2.0, 1.0), half_width=1.0)

    region = source_geometry._build_strong_lensing_region(
        (left,),
        (right,),
        geometry_tolerance=1.0e-6,
    )

    assert region.is_valid
    assert tuple(float(value) for value in region.bounds) == (0.0, 0.0, 3.0, 2.0)
    assert float(region.area) == pytest.approx(6.0)

    with pytest.raises(RuntimeError, match="strong-lensing region has no finite positive area"):
        source_geometry._build_strong_lensing_region(
            (),
            (),
            geometry_tolerance=1.0e-6,
        )


def test_source_boundary_clearance_uses_regular_curves_and_ignores_point_caustics():
    """Measure to the nearest line without assigning radius to point caustics."""
    pytest.importorskip("shapely")
    geometry = source_geometry._BoundaryGeometry(
        caustic_curves=(_closed_square(),),
        pseudo_caustic_curves=(_closed_square(center=(10.0, 0.0)),),
        critical_curve_fov=4.0,
        pixelscale=0.1,
        pseudo_caustic_points=64,
        point_caustics=(np.array([0.0, 0.0]),),
    )

    assert source_geometry._source_boundary_clearance(
        0.0,
        0.0,
        geometry,
        1.0e-6,
    ) == pytest.approx(1.0)


def test_sample_position_rejects_outside_draws_with_fixed_rng():
    """Return the third fixed draw pair after two triangle rejections."""
    shapely = pytest.importorskip("shapely")
    region = shapely.Polygon([(0.0, 0.0), (2.0, 0.0), (0.0, 2.0)])

    source_x, source_y, area, attempts = source_geometry._sample_position(
        region,
        np.random.default_rng(1),
        max_attempts=4,
        lens_identifier="lens_0",
        geometry_settings={"pixelscale": 0.1},
    )

    assert source_x == pytest.approx(0.6236629040209709)
    assert source_y == pytest.approx(0.8466528979451513)
    assert area == pytest.approx(2.0)
    assert attempts == 3


def test_sample_position_rejects_only_exact_excluded_points():
    """Reject an exactly sampled point while accepting a nearby exclusion."""
    shapely = pytest.importorskip("shapely")
    region = shapely.box(0.0, 0.0, 1.0, 1.0)
    first_draw = np.array([0.2616121342493164, 0.2984911434141233])

    source_x, source_y, area, attempts = source_geometry._sample_position(
        region,
        np.random.default_rng(2),
        max_attempts=2,
        lens_identifier="lens_1",
        geometry_settings={"pixelscale": 0.1},
        excluded_points=(first_draw,),
    )

    assert source_x == pytest.approx(0.8142257405942803)
    assert source_y == pytest.approx(0.0919159421350969)
    assert area == pytest.approx(1.0)
    assert attempts == 2

    nearby = np.array([np.nextafter(first_draw[0], np.inf), first_draw[1]])
    source_x, source_y, _, attempts = source_geometry._sample_position(
        region,
        np.random.default_rng(2),
        max_attempts=1,
        lens_identifier="lens_1",
        geometry_settings={"pixelscale": 0.1},
        excluded_points=(nearby,),
    )
    assert source_x == first_draw[0]
    assert source_y == first_draw[1]
    assert attempts == 1


def test_sample_position_reports_bounded_attempt_exhaustion_context():
    """Report lens, area, attempt, and geometry context after bounded failure."""
    shapely = pytest.importorskip("shapely")
    region = shapely.box(0.0, 0.0, 1.0, 1.0)
    preview = np.random.default_rng(3)
    excluded_points = tuple(
        np.array([preview.uniform(0.0, 1.0), preview.uniform(0.0, 1.0)]) for _ in range(2)
    )

    with pytest.raises(RuntimeError) as error:
        source_geometry._sample_position(
            region,
            np.random.default_rng(3),
            max_attempts=2,
            lens_identifier="lens[3]",
            geometry_settings={"pixelscale": 0.1, "fov": 4.0},
            excluded_points=excluded_points,
        )

    message = str(error.value)
    assert "Unable to sample the strong-lensing region for lens[3] after 2 attempts" in message
    assert "bounding-box area=1.0 arcsec^2" in message
    assert "polygon area=1.0 arcsec^2" in message
    assert "pixelscale=0.1" in message
    assert "fov=4.0" in message


def test_validate_source_geometry_support_rejects_nested_steep_epl_with_component_path():
    """Report the recursive generated name of the first unsupported EPL."""
    ordinary_spec = SimpleNamespace(model="SIS", parameters={})
    epl_spec = SimpleNamespace(model="EPL", parameters={})
    inner_spec = SimpleNamespace(
        model="SinglePlane",
        parameters={"lenses": (epl_spec,)},
    )
    root_spec = SimpleNamespace(
        model="SinglePlane",
        parameters={"lenses": (ordinary_spec, inner_spec)},
    )
    epl_lens = SimpleNamespace(t=_RecordingParameter(1.0))
    root_lens = SimpleNamespace(
        lenses=(
            SimpleNamespace(),
            SimpleNamespace(lenses=(epl_lens,)),
        )
    )

    assert source_geometry._validate_source_geometry_support(root_spec, root_lens) is None

    epl_lens.t.value = _FakeTensor(1.2)
    with pytest.raises(NotImplementedError) as error:
        source_geometry._validate_source_geometry_support(root_spec, root_lens)

    message = str(error.value)
    assert "Component 'lens_1_0'" in message
    assert "EPL slope t=1.2 > 1" in message
    assert "Restrict the prior to t <= 1 or use CausticsLensImageNode" in message

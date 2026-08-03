"""Private source-plane geometry and sampling helpers for Caustics models."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from . import lens_system as _lens_system
from . import runtime as _runtime


def _trace_pseudo_caustics(
    lens,
    geometry_adapter,
    values,
    *,
    num_points,
    epsilon,
    geometry_tolerance,
):
    """Trace every registered pseudo-caustic through the total lens.

    Parameters
    ----------
    lens : object
        Realized total Caustics lens implementing ``raytrace(x, y)``.
    geometry_adapter : _GeometryAdapter
        Realized total-lens adapter supplying ordered component-owned loop
        generators.
    values : Mapping
        Numeric inputs for the same total-lens realization.
    num_points : int
        Number of unique, evenly spaced vertices on each image-plane loop.
    epsilon : float
        Initial loop radius in image-plane arcseconds before any
        generator-owned cap is applied.
    geometry_tolerance : float
        Maximum allowed pointwise source-plane change, in arcseconds,
        between successive loop refinements.

    Returns
    -------
    tuple of numpy.ndarray
        One closed source-plane boundary per registered generator, in
        generator order. Each array has shape ``(num_points + 1, 2)`` in
        arcseconds, with the first vertex repeated at the end.

    Raises
    ------
    ImportError
        If a registered generator is present and the optional Caustics or Torch
        runtime needed to map its loop is unavailable.
    RuntimeError
        If a shrinking-loop sequence does not converge within 32
        refinements.

    Notes
    -----
    Every loop is mapped through ``lens`` rather than through its owning
    component. Increasing-angle producer order is retained in the returned
    source-plane boundary.
    """
    angles = 2.0 * np.pi * np.arange(num_points) / num_points
    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    pseudo_caustics = []

    for generator in geometry_adapter.pseudo_caustic_generators(values):
        center = np.asarray(generator.center, dtype=float)
        initial_radius = epsilon
        if generator.max_initial_radius is not None:
            initial_radius = min(initial_radius, generator.max_initial_radius)
        radius = initial_radius
        previous_curve = _runtime._raytrace_curve(lens, center + radius * directions)
        last_change = np.inf

        for _ in range(32):
            radius *= 0.5
            current_curve = _runtime._raytrace_curve(lens, center + radius * directions)
            last_change = float(np.max(np.linalg.norm(current_curve - previous_curve, axis=1)))
            if last_change <= geometry_tolerance:
                pseudo_caustics.append(np.concatenate((current_curve, current_curve[:1]), axis=0))
                break
            previous_curve = current_curve
        else:
            raise RuntimeError(
                "Pseudo-caustic extraction did not converge after 32 refinements; "
                f"initial_radius={initial_radius} arcsec, epsilon={epsilon} arcsec, "
                f"geometry_tolerance={geometry_tolerance} arcsec, "
                f"final boundary change was {last_change} arcsec."
            )

    return tuple(pseudo_caustics)


class _CausticFOVError(RuntimeError):
    """Signal a critical-curve state selected for bounded FOV retry.

    Notes
    -----
    ``_find_all_caustics`` raises this private control-flow exception only for
    conditions treated as potentially recoverable by enlarging the square
    image-plane field of view: an exterior boundary that is not yet positive
    definite, no detected critical curve, or a critical curve that reaches the
    grid boundary. Expansion is bounded and need not resolve every such state.
    ``CausticsSourcePositionNode`` owns the bounded expansion policy and
    converts final exhaustion to a public ``RuntimeError``. Structural contour
    and mapping failures remain ordinary ``RuntimeError`` instances and are
    never reclassified as FOV-retry signals.
    """


def _outer_grid_boundary(values):
    """Extract a square grid's outer boundary without repeated corners.

    Parameters
    ----------
    values : numpy.ndarray, shape (N, N, ...)
        Trusted square grid with ``N >= 2``. Units and trailing dimensions
        are inherited by the result.

    Returns
    -------
    numpy.ndarray, shape (4 * N - 4, ...)
        Top row, bottom row, and side interiors concatenated without duplicated
        corner entries.

    Notes
    -----
    The result preserves the input row/column directions but is not a cyclic
    walk around the perimeter: both full rows precede the two side interiors.
    The trusted caller owns shape validation; NumPy indexing and concatenation
    errors for malformed inputs propagate unchanged.
    """
    return np.concatenate(
        (
            values[0],
            values[-1],
            values[1:-1, 0],
            values[1:-1, -1],
        ),
        axis=0,
    )


def _find_all_caustics(
    lens,
    *,
    center,
    fov,
    pixelscale,
    geometry_tolerance,
    jacobian_mask_points=(),
):
    """Locate all complete critical curves and map them into the source plane.

    The implementation uses the generic Caustics lens-equation Jacobian and
    raytrace protocols; it does not depend on a particular analytic lens class.

    Parameters
    ----------
    lens : object
        Realized Caustics lens implementing ``jacobian_lens_equation`` and
        ``raytrace``.
    center : tuple of float
        Image-plane x/y center of the square search grid in arcseconds.
    fov : float
        Width of the square image-plane search region in arcseconds.
    pixelscale : float
        Requested maximum grid spacing in arcseconds. The actual spacing evenly
        divides ``fov`` and is no larger than this value.
    geometry_tolerance : float
        Maximum allowed endpoint gap for a mapped source-plane caustic, in
        arcseconds.
    jacobian_mask_points : iterable of tuple of float, optional
        Image-plane locations in arcseconds requiring a one-grid-spacing
        determinant mask so a discontinuity is not misidentified as a true
        critical curve.

    Returns
    -------
    list of numpy.ndarray
        Producer-closed source-plane caustic curves. Each element has shape
        ``(P, 2)`` and contains x/y offsets in arcseconds, with an endpoint gap
        no larger than ``geometry_tolerance``. Disconnected critical curves
        remain separate list entries in ContourPy order.

    Raises
    ------
    ImportError
        If Caustics, Torch, or ContourPy is unavailable.
    RuntimeError
        If singularity masking leaves no valid grid or boundary sample, or
        ContourPy returns a malformed, non-finite, or open curve, or a mapped
        caustic is not closed within ``geometry_tolerance``.
    _CausticFOVError
        If a selected outer-boundary eigenvalue compares less than or equal to
        zero, no critical curve is found, or a curve reaches the image-plane
        boundary.

    Notes
    -----
    The interval count is rounded up to an even value, so the actual grid
    spacing is no larger than the requested ``pixelscale``. Caustics
    Jacobian types, shapes, and finiteness are trusted except at registered
    Jacobian mask points: non-finite determinant samples and a one-spacing
    neighborhood around each registered point are deliberately masked as part
    of critical-curve extraction. A point at exactly one actual grid spacing
    is masked. The post-mask grid and outer-boundary non-emptiness checks remain
    algorithmic completeness guards. For finite derived boundary eigenvalues,
    values less than or equal to zero trigger ``_CausticFOVError``. Non-finite
    eigenvalues arising during symmetrization are not explicitly rejected by
    that comparison. Critical-curve closure and grid-boundary contact use an
    internal scale-aware tolerance distinct from source-plane
    ``geometry_tolerance``. ContourPy output is independently validated, and an
    in-tolerance but nonzero critical-curve endpoint gap is closed by appending
    the first vertex before raytracing. The mapped raytrace's paired shape,
    type, and finiteness remain trusted backend postconditions apart from the
    endpoint-gap calculation.
    """
    contourpy = _runtime._import_contourpy()
    _, torch = _runtime._import_caustics_dependencies()

    center_x, center_y = center
    num_intervals = int(np.ceil(fov / pixelscale))
    if num_intervals % 2:
        num_intervals += 1
    actual_pixelscale = fov / num_intervals
    half_fov = 0.5 * fov
    x_axis = torch.linspace(
        center_x - half_fov,
        center_x + half_fov,
        num_intervals + 1,
        dtype=torch.float64,
    )
    y_axis = torch.linspace(
        center_y - half_fov,
        center_y + half_fov,
        num_intervals + 1,
        dtype=torch.float64,
    )
    grid_y, grid_x = torch.meshgrid(y_axis, x_axis, indexing="ij")

    jacobian = lens.jacobian_lens_equation(grid_x, grid_y, method="autograd")
    determinant = _runtime._to_numpy(torch.linalg.det(jacobian))

    x_coordinates = _runtime._to_numpy(x_axis)
    y_coordinates = _runtime._to_numpy(y_axis)
    invalid = ~np.isfinite(determinant)
    for mask_x, mask_y in jacobian_mask_points:
        squared_distance = (x_coordinates[np.newaxis, :] - mask_x) ** 2 + (
            y_coordinates[:, np.newaxis] - mask_y
        ) ** 2
        invalid |= squared_distance <= actual_pixelscale**2

    if np.all(invalid):
        raise RuntimeError("The lens-equation Jacobian grid contains no finite values.")

    symmetric_jacobian = 0.5 * (jacobian + jacobian.transpose(-1, -2))
    eigenvalues = _runtime._to_numpy(torch.linalg.eigvalsh(symmetric_jacobian))

    boundary_invalid = _outer_grid_boundary(invalid)
    boundary_eigenvalues = _outer_grid_boundary(eigenvalues)
    valid_boundary = ~boundary_invalid
    if not np.any(valid_boundary):
        raise RuntimeError("The lens-equation Jacobian grid boundary contains no finite values.")
    if np.any(boundary_eigenvalues[valid_boundary] <= 0.0):
        raise _CausticFOVError(
            "The image-plane field-of-view boundary has not reached the "
            "positive-definite exterior lens-mapping region; increase fov."
        )

    contour_generator = contourpy.contour_generator(
        x=x_coordinates,
        y=y_coordinates,
        z=np.ma.array(determinant, mask=invalid),
        line_type=contourpy.LineType.Separate,
    )
    critical_curves = contour_generator.lines(0.0)
    if not critical_curves:
        raise _CausticFOVError(
            "No critical curves were found in the configured image-plane field of view; increase fov."
        )
    caustic_curves = []
    boundary_tolerance = max(
        actual_pixelscale * 1.0e-6,
        np.finfo(float).eps * max(1.0, abs(center_x), abs(center_y), fov) * 16.0,
    )

    for critical_curve in critical_curves:
        critical_curve = np.asarray(critical_curve, dtype=float)
        if critical_curve.ndim != 2 or critical_curve.shape[1] != 2:
            raise RuntimeError("Contour extraction returned an invalid critical curve.")
        if len(critical_curve) < 4:
            raise RuntimeError("A critical curve has fewer than three vertices.")
        if not np.all(np.isfinite(critical_curve)):
            raise RuntimeError("A critical curve contains non-finite coordinates.")

        reaches_boundary = (
            np.isclose(critical_curve[:, 0], x_coordinates[0], atol=boundary_tolerance, rtol=0.0)
            | np.isclose(critical_curve[:, 0], x_coordinates[-1], atol=boundary_tolerance, rtol=0.0)
            | np.isclose(critical_curve[:, 1], y_coordinates[0], atol=boundary_tolerance, rtol=0.0)
            | np.isclose(critical_curve[:, 1], y_coordinates[-1], atol=boundary_tolerance, rtol=0.0)
        )
        if np.any(reaches_boundary):
            raise _CausticFOVError(
                "A critical curve reaches the configured image-plane field-of-view boundary; increase fov."
            )

        critical_gap = float(np.linalg.norm(critical_curve[0] - critical_curve[-1]))
        if critical_gap > boundary_tolerance:
            raise RuntimeError(
                f"Contour extraction returned an open critical curve with endpoint gap {critical_gap} arcsec."
            )
        if critical_gap > 0.0:
            critical_curve = np.concatenate((critical_curve, critical_curve[:1]), axis=0)

        caustic_curve = _runtime._raytrace_curve(lens, critical_curve)
        caustic_gap = float(np.linalg.norm(caustic_curve[0] - caustic_curve[-1]))
        if caustic_gap > geometry_tolerance:
            raise RuntimeError(f"A mapped caustic is open with endpoint gap {caustic_gap} arcsec.")
        caustic_curves.append(caustic_curve)

    return caustic_curves


@dataclass(frozen=True)
class _BoundaryGeometry:
    """Store one typed source-boundary certification snapshot.

    Attributes
    ----------
    caustic_curves : tuple of numpy.ndarray
        Closed true-caustic curves, each with shape ``(P, 2)`` in
        source-plane arcseconds.
    pseudo_caustic_curves : tuple of numpy.ndarray
        Closed pseudo-caustic curves, each with shape ``(P, 2)`` in
        source-plane arcseconds.
    critical_curve_fov : float
        Successful square image-plane critical-curve search FOV in arcseconds.
    pixelscale : float
        Requested snapshot grid-spacing upper bound in arcseconds. This is not
        necessarily the actual even-grid spacing used by critical-curve
        extraction.
    pseudo_caustic_points : int
        Number of unique image-plane loop vertices requested for each
        pseudo-caustic.
    point_caustics : tuple of numpy.ndarray
        Certified source-plane point-caustic centers, each with shape ``(2,)``
        in arcseconds.

    Notes
    -----
    The frozen dataclass prevents field reassignment but is only shallowly
    immutable. All contained NumPy arrays remain mutable, including both the
    curve arrays and the ``point_caustics`` centers.
    """

    caustic_curves: tuple[np.ndarray, ...]
    pseudo_caustic_curves: tuple[np.ndarray, ...]
    critical_curve_fov: float
    pixelscale: float
    pseudo_caustic_points: int
    point_caustics: tuple[np.ndarray, ...] = ()


_POINT_CAUSTIC_CONTRACTION_FACTOR = 0.5


@dataclass(frozen=True)
class _PointCausticPartition:
    """Separate contracting axisymmetric point caustics from regular curves.

    Attributes
    ----------
    previous_curves : tuple of numpy.ndarray
        Matched regular true-caustic curves from the penultimate snapshot.
        Each producer-closed array has shape ``(P, 2)`` in source-plane
        arcseconds.
    current_curves : tuple of numpy.ndarray
        Matched regular true-caustic curves from the final snapshot, ordered
        one-to-one with ``previous_curves``. Each array has shape ``(P, 2)``
        in source-plane arcseconds.
    previous_points : tuple of numpy.ndarray
        Axis-aligned bounding-box centers for contracting curves in the
        penultimate snapshot. Each array has shape ``(2,)`` in source-plane
        arcseconds.
    current_points : tuple of numpy.ndarray
        Corresponding final-snapshot point-caustic centers, each with shape
        ``(2,)`` in source-plane arcseconds.

    Notes
    -----
    The frozen dataclass prevents field reassignment but is only shallowly
    immutable. NumPy arrays contained by all four tuples remain mutable.
    """

    previous_curves: tuple[np.ndarray, ...]
    current_curves: tuple[np.ndarray, ...]
    previous_points: tuple[np.ndarray, ...]
    current_points: tuple[np.ndarray, ...]


def _curve_center_and_diameter(curve):
    """Measure a raw curve's axis-aligned center and bounding-box diagonal.

    Parameters
    ----------
    curve : array-like, shape (P, 2)
        Trusted source-plane x/y curve in arcseconds. An exactly repeated final
        vertex is ignored when present.

    Returns
    -------
    center : numpy.ndarray, shape (2,)
        Midpoint of the curve's axis-aligned bounding box in arcseconds.
    diameter : float
        Euclidean length of that bounding-box diagonal in arcseconds.

    Notes
    -----
    ``diameter`` is neither an arclength nor a maximum pairwise-vertex
    distance, and ``center`` is not a centroid. Production callers guarantee a
    nonempty curve with two coordinate columns; NumPy conversion and reduction
    errors for malformed inputs propagate unchanged.
    """
    points = np.asarray(curve, dtype=float)
    if np.array_equal(points[0], points[-1]):
        points = points[:-1]
    lower = np.min(points, axis=0)
    upper = np.max(points, axis=0)
    return 0.5 * (lower + upper), float(np.linalg.norm(upper - lower))


def _raw_curve_hausdorff_distance(first_curve, second_curve):
    """Compute a bidirectional nearest-vertex distance between raw curves.

    Parameters
    ----------
    first_curve : array-like, shape (P, 2)
        First trusted source-plane curve in arcseconds.
    second_curve : array-like, shape (Q, 2)
        Second trusted source-plane curve in arcseconds.

    Returns
    -------
    float
        Maximum of the two directed nearest-vertex distances, in arcseconds.

    Notes
    -----
    An exactly repeated closing endpoint is omitted from each curve before
    building its ``scipy.spatial.cKDTree``. This is a discrete vertex-set
    Hausdorff distance computed directly from cKDTree vertex clouds; boundary
    certification instead delegates to Shapely's LineString Hausdorff
    implementation. Production callers guarantee nonempty finite curves with
    two coordinate columns; conversion and tree-query errors otherwise
    propagate unchanged.
    """
    first = np.asarray(first_curve, dtype=float)
    second = np.asarray(second_curve, dtype=float)
    if np.array_equal(first[0], first[-1]):
        first = first[:-1]
    if np.array_equal(second[0], second[-1]):
        second = second[:-1]
    first_tree = cKDTree(first)
    second_tree = cKDTree(second)
    first_to_second = second_tree.query(first)[0]
    second_to_first = first_tree.query(second)[0]
    return float(max(np.max(first_to_second), np.max(second_to_first)))


def _match_raw_caustic_curves(reference_curves, candidate_curves):
    """Match raw true-caustic curves by discrete Hausdorff assignment.

    Parameters
    ----------
    reference_curves : sequence of array-like
        Reference producer-closed curves, each with shape ``(P, 2)`` in
        source-plane arcseconds.
    candidate_curves : sequence of array-like
        Candidate curves of the same form and units.

    Returns
    -------
    ordered_candidates : tuple
        Candidate objects reordered into reference-index order. Unequal counts
        return the candidates unchanged; two empty sequences return ``()``.
    counts_match : bool
        Whether the curve counts agree. Two empty sequences count as a match.

    Notes
    -----
    For equal nonzero counts, the Hungarian assignment minimizes the sum of
    discrete bidirectional nearest-vertex distances. Matching does not build or
    repair polygonal geometry, and the returned tuple retains the original
    candidate array objects.
    """
    if len(reference_curves) != len(candidate_curves):
        return tuple(candidate_curves), False
    if not reference_curves:
        return (), True
    costs = np.array(
        [
            [
                _raw_curve_hausdorff_distance(reference_curve, candidate_curve)
                for candidate_curve in candidate_curves
            ]
            for reference_curve in reference_curves
        ]
    )
    rows, columns = linear_sum_assignment(costs)
    assignment = dict(zip(rows.tolist(), columns.tolist(), strict=True))
    return (
        tuple(candidate_curves[assignment[index]] for index in range(len(reference_curves))),
        True,
    )


def _partition_axisymmetric_point_caustics(
    older_curves,
    previous_curves,
    current_curves,
    *,
    boundary_tolerance,
):
    """Partition certified contractions from regular true boundaries.

    Parameters
    ----------
    older_curves : sequence of array-like
        Raw true-caustic curves from the oldest of three successive snapshots,
        each with shape ``(P, 2)`` in source-plane arcseconds.
    previous_curves : sequence of array-like
        Penultimate raw true-caustic curves in the same units.
    current_curves : sequence of array-like
        Final raw true-caustic curves in the same units.
    boundary_tolerance : float
        Inclusive maximum displacement of successive bounding-box centers, in
        arcseconds.

    Returns
    -------
    _PointCausticPartition or None
        Regular penultimate/final curves plus corresponding point centers,
        ordered by the oldest-snapshot assignment. ``None`` is the count
        mismatch sentinel for either consecutive assignment.

    Notes
    -----
    Curves are first matched from oldest to penultimate and then from the
    reordered penultimate set to the final set. A matched triple is classified
    as a point caustic only when each successive bounding-box diagonal is no
    larger than half its predecessor and each successive center displacement
    is no larger than ``boundary_tolerance``. All other triples remain regular
    boundaries; there is no separate absolute-diameter threshold. The caller's
    exact-axisymmetry capability gate, rather than this helper, decides whether
    contraction classification is permitted.
    """
    previous_curves, previous_counts_match = _match_raw_caustic_curves(
        older_curves,
        previous_curves,
    )
    current_curves, current_counts_match = _match_raw_caustic_curves(
        previous_curves,
        current_curves,
    )
    if not previous_counts_match or not current_counts_match:
        return None

    regular_previous = []
    regular_current = []
    previous_points = []
    current_points = []
    for older_curve, previous_curve, current_curve in zip(
        older_curves,
        previous_curves,
        current_curves,
        strict=True,
    ):
        older_center, older_diameter = _curve_center_and_diameter(older_curve)
        previous_center, previous_diameter = _curve_center_and_diameter(previous_curve)
        current_center, current_diameter = _curve_center_and_diameter(current_curve)
        is_point_caustic = (
            previous_diameter <= _POINT_CAUSTIC_CONTRACTION_FACTOR * older_diameter
            and current_diameter <= _POINT_CAUSTIC_CONTRACTION_FACTOR * previous_diameter
            and np.linalg.norm(previous_center - older_center) <= boundary_tolerance
            and np.linalg.norm(current_center - previous_center) <= boundary_tolerance
        )
        if is_point_caustic:
            previous_points.append(previous_center)
            current_points.append(current_center)
        else:
            regular_previous.append(previous_curve)
            regular_current.append(current_curve)

    return _PointCausticPartition(
        previous_curves=tuple(regular_previous),
        current_curves=tuple(regular_current),
        previous_points=tuple(previous_points),
        current_points=tuple(current_points),
    )


def _close_curve(curve, *, tolerance):
    """Normalize and exactly reclose a trusted source-plane boundary.

    Parameters
    ----------
    curve : array-like, shape (N, 2)
        Producer-certified closed source-plane x/y coordinates in arcseconds.
    tolerance : float
        Minimum retained separation between cyclic consecutive vertices, in
        arcseconds.

    Returns
    -------
    numpy.ndarray, shape (M, 2)
        Source-plane floating-point coordinates in arcseconds, with at least
        three unique vertices and an exactly repeated first/last vertex.
        Consecutive vertices within ``tolerance`` are removed, so ``M`` may be
        smaller than ``N``.

    Raises
    ------
    RuntimeError
        If normalization leaves fewer than three unique vertices.

    Notes
    -----
    Input shape, finiteness, closure, and positive tolerance are established by
    production producers and constructor validation. This helper removes
    consecutive near-duplicates, removes the cyclic near-duplicate before the
    endpoint, and appends the first retained vertex exactly.
    """
    coordinates = np.asarray(curve, dtype=float)

    retained = [coordinates[0]]
    for coordinate in coordinates[1:-1]:
        if np.linalg.norm(coordinate - retained[-1]) > tolerance:
            retained.append(coordinate)

    while len(retained) > 1 and np.linalg.norm(retained[-1] - retained[0]) <= tolerance:
        retained.pop()
    retained.append(retained[0])
    coordinates = np.asarray(retained, dtype=float)

    if len(coordinates) < 4 or len(np.unique(coordinates[:-1], axis=0)) < 3:
        raise RuntimeError("A caustic boundary has fewer than three unique vertices.")
    return coordinates


def _resample_closed_curve(curve, num_points=64):
    """Resample a normalized closed curve at equal arclength intervals.

    Parameters
    ----------
    curve : numpy.ndarray, shape (N, 2)
        Normalized source-plane boundary whose last vertex exactly repeats its
        first.
    num_points : int, optional
        Number of unique equally spaced points to return. Production callers
        provide a positive integer.

    Returns
    -------
    numpy.ndarray, shape (num_points, 2)
        Source-plane coordinates in arcseconds, sampled without repeating the
        endpoint.

    Notes
    -----
    Sampling is piecewise linear in cumulative arclength over every segment,
    including the final segment back to the exactly repeated first vertex.
    The trusted normalized curve has positive total arclength; NumPy errors for
    malformed curves or invalid sample counts propagate unchanged.
    """
    coordinates = np.asarray(curve, dtype=float)
    segment_lengths = np.linalg.norm(np.diff(coordinates, axis=0), axis=1)
    cumulative_arclength = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    sample_arclength = np.linspace(
        0.0,
        cumulative_arclength[-1],
        num_points,
        endpoint=False,
    )
    return np.column_stack(
        (
            np.interp(sample_arclength, cumulative_arclength, coordinates[:, 0]),
            np.interp(sample_arclength, cumulative_arclength, coordinates[:, 1]),
        )
    )


def _curve_orientation_is_preserved(reference, candidate):
    """Compare the traversal directions of two matched closed curves.

    Parameters
    ----------
    reference : numpy.ndarray, shape (P, 2)
        Normalized reference curve in source-plane arcseconds, with its first
        vertex exactly repeated at the end.
    candidate : numpy.ndarray, shape (Q, 2)
        Normalized candidate curve in the same units and closure convention.

    Returns
    -------
    bool
        Whether forward candidate traversal is no farther from the reference
        than reversed traversal after optimal cyclic registration.

    Notes
    -----
    Each curve is resampled to 64 unique equal-arclength points. Comparison is
    cyclically registered at that 64-sample resolution, reducing sensitivity
    to the chosen starting vertex while intentionally detecting a
    producer-direction reversal. It is not a continuous phase optimization.
    An exact forward/reverse tie is treated as preserved orientation.
    """
    reference_points = _resample_closed_curve(reference)
    candidate_points = _resample_closed_curve(candidate)

    def minimum_cyclic_distance(points):
        """Minimize mean squared displacement over all cyclic registrations.

        Parameters
        ----------
        points : numpy.ndarray, shape (64, 2)
            Candidate equal-arclength samples in source-plane arcseconds.

        Returns
        -------
        float
            Minimum mean squared pointwise distance to the enclosing
            ``reference_points``, in square arcseconds.

        Notes
        -----
        The closure compares all cyclic shifts and trusts the candidate and
        reference sample counts to agree.
        """
        return min(
            float(
                np.mean(
                    np.sum(
                        (reference_points - np.roll(points, shift, axis=0)) ** 2,
                        axis=1,
                    )
                )
            )
            for shift in range(len(points))
        )

    forward_distance = minimum_cyclic_distance(candidate_points)
    reverse_distance = minimum_cyclic_distance(candidate_points[::-1])
    return forward_distance <= reverse_distance


def _boundary_regions(curves, *, geometry_tolerance):
    """Convert typed source-boundary curves to repaired regions.

    Parameters
    ----------
    curves : iterable of array-like
        Closed source-plane curves, each with shape ``(P, 2)`` in
        arcseconds.
    geometry_tolerance : float
        Consecutive-vertex curve-normalization tolerance in arcseconds.

    Returns
    -------
    tuple of shapely geometry
        Repaired finite positive-area polygonal region corresponding to each
        input curve, in input order. Areas are in square arcseconds.

    Raises
    ------
    ImportError
        If the optional Shapely dependency is unavailable.
    RuntimeError
        If curve normalization collapses a boundary or a repaired region is
        empty, non-finite, or has non-positive area.

    Notes
    -----
    Each normalized ring is passed independently through
    ``shapely.make_valid(..., method="structure")``. Invalid or
    self-intersecting boundaries may therefore become multi-component regions,
    but different input boundaries are neither matched nor unioned here.
    Precision-grid snapping is not performed here; the tolerance controls only
    curve normalization. Final union snapping belongs to
    ``_build_strong_lensing_region``.
    """
    shapely = _runtime._import_shapely()
    regions = []
    for boundary_index, curve in enumerate(curves):
        coordinates = _close_curve(curve, tolerance=geometry_tolerance)
        region = shapely.make_valid(shapely.Polygon(coordinates), method="structure")
        area = float(region.area)
        if region.is_empty or not np.isfinite(area) or area <= 0.0:
            raise RuntimeError(
                f"Caustic boundary {boundary_index} did not enclose a finite positive-area polygonal region."
            )
        regions.append(region)
    return tuple(regions)


def _match_boundary_curves(reference_curves, candidate_curves, *, geometry_tolerance):
    """Match one typed boundary set by minimum Hausdorff displacement.

    Parameters
    ----------
    reference_curves : sequence of array-like
        Reference closed curves, each with shape ``(P, 2)`` in source-plane
        arcseconds.
    candidate_curves : sequence of array-like
        Candidate closed curves of the same boundary type and units.
    geometry_tolerance : float
        Consecutive-vertex curve-normalization tolerance in arcseconds.

    Returns
    -------
    ordered_candidates : tuple
        Candidate curves reordered to the reference assignment. If counts
        differ, the original candidate order is returned.
    displacement : float
        Maximum assigned Hausdorff distance in arcseconds. Unequal counts
        return ``inf``; two empty sequences return zero.
    counts_stable : bool
        Whether the two typed sets have equal counts. Two empty sequences are
        stable.
    orientation_stable : bool
        Whether every assigned candidate retains its reference producer
        direction. Unequal counts are unstable; two empty sequences are
        stable.

    Raises
    ------
    ImportError
        If the optional Shapely dependency is unavailable.
    RuntimeError
        If normalization collapses a curve below three unique vertices.

    Notes
    -----
    Equal nonzero sets are assigned with the Hungarian algorithm to minimize
    total Shapely line Hausdorff distance, then returned in reference-index
    order. Orientation is evaluated only after that assignment. Unlike
    ``_match_raw_caustic_curves``, this helper normalizes curves and delegates
    each distance to Shapely LineString geometry rather than querying cKDTree
    vertex clouds directly.

    Count mismatch is a non-exceptional convergence sentinel: displacement is
    ``inf``, both stability flags are false, and candidate order is unchanged.
    Two empty sets return zero displacement with both flags true.
    """
    if len(reference_curves) != len(candidate_curves):
        return tuple(candidate_curves), np.inf, False, False
    if not reference_curves:
        return (), 0.0, True, True
    shapely = _runtime._import_shapely()
    reference_coordinates = [_close_curve(curve, tolerance=geometry_tolerance) for curve in reference_curves]
    candidate_coordinates = [_close_curve(curve, tolerance=geometry_tolerance) for curve in candidate_curves]
    reference_lines = [shapely.LineString(coordinates) for coordinates in reference_coordinates]
    candidate_lines = [shapely.LineString(coordinates) for coordinates in candidate_coordinates]
    costs = np.array(
        [
            [float(reference.hausdorff_distance(candidate)) for candidate in candidate_lines]
            for reference in reference_lines
        ]
    )
    rows, columns = linear_sum_assignment(costs)
    assignment = dict(zip(rows.tolist(), columns.tolist(), strict=True))
    ordered = tuple(candidate_curves[assignment[index]] for index in range(len(reference_curves)))
    displacement = max(float(costs[index, assignment[index]]) for index in range(len(reference_curves)))
    orientation_stable = all(
        _curve_orientation_is_preserved(
            reference_coordinates[index],
            candidate_coordinates[assignment[index]],
        )
        for index in range(len(reference_curves))
    )
    return ordered, displacement, True, orientation_stable


def _boundary_topology_signature(geometry, *, geometry_tolerance):
    """Project a boundary snapshot to a typed topology signature.

    Parameters
    ----------
    geometry : _BoundaryGeometry
        Typed true- and pseudo-caustic boundary snapshot.
    geometry_tolerance : float
        Curve-normalization tolerance in arcseconds.

    Returns
    -------
    typed_counts : tuple
        Pair ``(true_counts, pseudo_counts)``. Each element is an ordered
        tuple of ``(component_count, interior_ring_count)`` values, one per
        boundary region.
    relations : tuple
        Pairwise Boolean relation tuples in combined true-then-pseudo region
        order. Each tuple contains ``(disjoint, within, contains, overlaps,
        touches)`` for one increasing index pair. Directional ``within`` and
        ``contains`` predicates describe the lower-index region relative to the
        higher-index region.

    Raises
    ------
    ImportError
        If the optional Shapely dependency is unavailable.
    RuntimeError
        If a boundary cannot be normalized into positive-area geometry.

    Notes
    -----
    True and pseudo boundary types remain distinct in ``typed_counts``. Within
    each type, the caller-established matching order controls both count and
    pairwise-relation order. Geometry positions and metric displacements are
    deliberately absent from the signature, and certified point caustics are
    ignored.
    """
    true_regions = _boundary_regions(
        geometry.caustic_curves,
        geometry_tolerance=geometry_tolerance,
    )
    pseudo_regions = _boundary_regions(
        geometry.pseudo_caustic_curves,
        geometry_tolerance=geometry_tolerance,
    )

    def region_counts(region):
        """Count polygon components and their interior rings.

        Parameters
        ----------
        region : shapely.Polygon or shapely.MultiPolygon
            Repaired positive-area polygonal region.

        Returns
        -------
        component_count : int
            Number of polygon components.
        interior_ring_count : int
            Total number of holes across all components.

        Notes
        -----
        ``_boundary_regions`` guarantees repaired polygonal input. A Polygon
        counts as one component; otherwise every member of the trusted
        MultiPolygon's ``geoms`` sequence is counted.
        """
        polygons = (region,) if region.geom_type == "Polygon" else tuple(region.geoms)
        return len(polygons), sum(len(polygon.interiors) for polygon in polygons)

    typed_counts = (
        tuple(region_counts(region) for region in true_regions),
        tuple(region_counts(region) for region in pseudo_regions),
    )
    regions = (*true_regions, *pseudo_regions)
    relations = tuple(
        (
            bool(regions[first].disjoint(regions[second])),
            bool(regions[first].within(regions[second])),
            bool(regions[first].contains(regions[second])),
            bool(regions[first].overlaps(regions[second])),
            bool(regions[first].touches(regions[second])),
        )
        for first in range(len(regions))
        for second in range(first + 1, len(regions))
    )
    return typed_counts, relations


def _compare_boundary_geometry(previous, current, geometry_tolerance):
    """Match successive typed snapshots and test geometric convergence.

    Parameters
    ----------
    previous : _BoundaryGeometry
        Penultimate boundary snapshot.
    current : _BoundaryGeometry
        Newly refined boundary snapshot.
    geometry_tolerance : float
        Curve-normalization tolerance in arcseconds.

    Returns
    -------
    reordered_current : _BoundaryGeometry
        Current snapshot with true and pseudo curves independently reordered to
        their previous assignments.
    displacement : float
        Maximum true-or-pseudo matched Hausdorff displacement in arcseconds.
    topology_stable : bool
        Whether typed curve counts, producer directions, and the complete
        topology signature match.

    Raises
    ------
    ImportError
        If the optional Shapely dependency is unavailable.
    RuntimeError
        If any curve collapses during normalization or cannot produce a finite
        positive-area repaired region.

    Notes
    -----
    Matching never crosses true/pseudo boundary types. Reordered arrays remain
    realization-local and are not cached on a node or across snapshots.
    Unequal counts yield infinite displacement through the matching sentinel.
    This helper does not apply ``boundary_tolerance``; the certification caller
    owns the displacement-threshold decision.
    ``point_caustics`` are copied through on the current snapshot but are not
    matched, measured, or included in topology here; the axisymmetric
    certification caller owns their count and center-stability checks.
    """
    caustic_curves, caustic_displacement, caustic_counts_stable, caustic_orientation_stable = (
        _match_boundary_curves(
            previous.caustic_curves,
            current.caustic_curves,
            geometry_tolerance=geometry_tolerance,
        )
    )
    (
        pseudo_caustic_curves,
        pseudo_displacement,
        pseudo_counts_stable,
        pseudo_orientation_stable,
    ) = _match_boundary_curves(
        previous.pseudo_caustic_curves,
        current.pseudo_caustic_curves,
        geometry_tolerance=geometry_tolerance,
    )
    current = _BoundaryGeometry(
        caustic_curves=caustic_curves,
        pseudo_caustic_curves=pseudo_caustic_curves,
        critical_curve_fov=current.critical_curve_fov,
        pixelscale=current.pixelscale,
        pseudo_caustic_points=current.pseudo_caustic_points,
        point_caustics=current.point_caustics,
    )
    displacement = max(caustic_displacement, pseudo_displacement)
    typed_counts_stable = caustic_counts_stable and pseudo_counts_stable
    orientation_stable = caustic_orientation_stable and pseudo_orientation_stable
    topology_stable = (
        typed_counts_stable
        and orientation_stable
        and _boundary_topology_signature(
            previous,
            geometry_tolerance=geometry_tolerance,
        )
        == _boundary_topology_signature(
            current,
            geometry_tolerance=geometry_tolerance,
        )
    )
    return current, displacement, topology_stable


def _source_boundary_clearance(source_x, source_y, geometry, geometry_tolerance):
    """Measure source distance to the nearest certified typed boundary.

    Parameters
    ----------
    source_x : float
        Source-plane x position in arcseconds.
    source_y : float
        Source-plane y position in arcseconds.
    geometry : _BoundaryGeometry
        Certified snapshot containing at least one regular true or pseudo
        boundary.
    geometry_tolerance : float
        Curve-normalization tolerance in arcseconds.

    Returns
    -------
    float
        Minimum distance to any true- or pseudo-caustic line in source-plane
        arcseconds.

    Raises
    ------
    ImportError
        If the optional Shapely dependency is unavailable.
    RuntimeError
        If normalization collapses a producer-certified boundary.
    ValueError
        If the snapshot contains no regular true or pseudo boundary from which
        a minimum distance can be taken.

    Notes
    -----
    Clearance is the minimum Euclidean distance to normalized boundary
    LineStrings. Certified point caustics are intentionally excluded because
    they have no radius or boundary uncertainty.
    """
    curves = (*geometry.caustic_curves, *geometry.pseudo_caustic_curves)
    shapely = _runtime._import_shapely()
    source = shapely.Point(source_x, source_y)
    return min(
        float(source.distance(shapely.LineString(_close_curve(curve, tolerance=geometry_tolerance))))
        for curve in curves
    )


def _build_strong_lensing_region(
    caustic_curves,
    pseudo_caustic_curves,
    *,
    geometry_tolerance,
):
    """Build the geometric union of all caustic-enclosed interiors.

    Parameters
    ----------
    caustic_curves : iterable of array-like
        Closed true-caustic source-plane boundaries with shape ``(P, 2)`` in
        arcseconds.
    pseudo_caustic_curves : iterable of array-like
        Closed pseudo-caustic source-plane boundaries with shape ``(P, 2)``
        in arcseconds.
    geometry_tolerance : float
        Consecutive-vertex curve-normalization tolerance and Shapely
        precision-grid spacing in arcseconds.

    Returns
    -------
    shapely geometry
        Valid, finite, positive-area strong-lensing geometry. Disconnected
        components, concavities, and holes are preserved subject to overlap
        and precision-grid snapping; area is measured in square arcseconds.

    Raises
    ------
    ImportError
        If Shapely is unavailable.
    RuntimeError
        If normalization collapses a boundary, or precision snapping, union,
        and final repair produce an empty, non-finite, or non-positive-area
        result.

    Notes
    -----
    Each boundary is converted to its own interior before union. The final
    union applies a Shapely precision grid with spacing
    ``geometry_tolerance``; this can genuinely collapse geometry and is why
    the final area check is retained. Per-boundary construction represents each
    input's concavities and holes before union, subject to subsequent overlap
    and precision snapping, and avoids accepting bounded faces collectively
    formed by several curves but inside none of their individual interiors.
    An empty input set reaches the same final non-positive-area
    ``RuntimeError`` guard.
    """
    shapely = _runtime._import_shapely()
    curves = [*caustic_curves, *pseudo_caustic_curves]

    enclosed_regions = _boundary_regions(
        curves,
        geometry_tolerance=geometry_tolerance,
    )

    region = shapely.union_all(enclosed_regions, grid_size=geometry_tolerance)
    region = shapely.make_valid(region, method="structure")
    area = float(region.area)
    if region.is_empty or not np.isfinite(area) or area <= 0.0:
        raise RuntimeError("The strong-lensing region has no finite positive area.")
    return region


def _sample_position(
    region,
    rng,
    *,
    max_attempts,
    lens_identifier,
    geometry_settings,
    excluded_points=(),
):
    """Uniformly sample one point from a polygonal source-plane region.

    Parameters
    ----------
    region : shapely.Polygon or shapely.MultiPolygon
        Strong-lensing source-plane region.
    rng : numpy.random.Generator
        Per-sample random number generator. It is intentionally isolated from
        generators used by other graph samples.
    max_attempts : int
        Maximum number of bounding-box rejection draws.
    lens_identifier : str
        Human-readable lens/sample identifier included in failure diagnostics.
    geometry_settings : Mapping
        Numerical geometry configuration included in failure diagnostics.
    excluded_points : iterable of array-like, optional
        Certified source-plane point-caustic centers rejected only on exact
        coordinate equality.

    Returns
    -------
    source_x : float
        Sampled source-plane x position in arcseconds.
    source_y : float
        Sampled source-plane y position in arcseconds.
    strong_lensing_area : float
        Area of ``region`` in square arcseconds.
    sampling_attempts : int
        Number of bounding-box draws consumed before acceptance.

    Raises
    ------
    ImportError
        If the optional Shapely dependency is unavailable.
    RuntimeError
        If no point is accepted within ``max_attempts``, with lens identity,
        geometry settings, bounding-box area, and polygon area in the message.

    Notes
    -----
    Production callers provide a validated finite positive-area polygonal
    region. Sampling uses uniform draws over its axis-aligned bounding box
    followed by Shapely's strict ``contains_xy`` interior test, so boundary
    points are rejected. An excluded point is rejected only when both sampled
    Python floats are exactly equal to its two coordinates; no tolerance or
    exclusion radius is applied. Each attempt consumes the x draw before the y
    draw. The returned attempt count is one-based, and the supplied sample-local
    generator isolates variable rejection counts from other graph samples.
    """
    shapely = _runtime._import_shapely()
    bounds = tuple(float(value) for value in region.bounds)
    area = float(region.area)
    min_x, min_y, max_x, max_y = bounds
    bounding_box_area = (max_x - min_x) * (max_y - min_y)

    for attempt in range(1, max_attempts + 1):
        source_x = float(rng.uniform(min_x, max_x))
        source_y = float(rng.uniform(min_y, max_y))
        inside_region = bool(shapely.contains_xy(region, source_x, source_y))
        on_point_caustic = any(
            source_x == float(point[0]) and source_y == float(point[1]) for point in excluded_points
        )
        if inside_region and not on_point_caustic:
            return source_x, source_y, area, attempt

    settings = ", ".join(f"{name}={value}" for name, value in geometry_settings.items())
    raise RuntimeError(
        "Unable to sample the strong-lensing region for "
        f"{lens_identifier} after {max_attempts} attempts; bounding-box area="
        f"{bounding_box_area} arcsec^2, polygon area={area} arcsec^2, {settings}."
    )


def _validate_source_position_configuration(
    *,
    fov,
    pixelscale,
    pixelscale_fraction,
    max_fov_expansions,
    fov_expansion_factor,
    pseudo_caustic_points,
    pseudo_caustic_epsilon,
    geometry_tolerance,
    geometry_tolerance_fraction,
    boundary_tolerance,
    boundary_tolerance_fraction,
    max_boundary_refinements,
    max_attempts,
):
    """Validate immutable source-position geometry and sampling settings.

    Parameters
    ----------
    fov : object or None
        Optional float-convertible initial image-plane FOV in arcseconds. A
        registered adapter derives it per realization when ``None``.
    pixelscale : object
        Float-convertible positive configured Jacobian-grid spacing upper bound
        in arcseconds.
    pixelscale_fraction : float or None
        Pre-normalized positive dimensionless fraction of the realized
        characteristic angular scale, or ``None`` for absolute scaling.
    max_fov_expansions : int
        Non-negative maximum number of bounded FOV expansions.
    fov_expansion_factor : object
        Float-convertible finite multiplier strictly greater than one.
    pseudo_caustic_points : int
        Number of unique vertices per pseudo-caustic; at least three.
    pseudo_caustic_epsilon : object
        Float-convertible positive initial singular-loop radius in arcseconds.
    geometry_tolerance : object
        Float-convertible positive curve/topology tolerance in arcseconds,
        strictly smaller than ``pixelscale``.
    geometry_tolerance_fraction : float or None
        Pre-normalized positive dimensionless curve/topology tolerance fraction
        of the realized characteristic angular scale, or ``None``.
    boundary_tolerance : object
        Float-convertible positive certification tolerance in arcseconds, at
        least ``geometry_tolerance``.
    boundary_tolerance_fraction : float or None
        Pre-normalized positive dimensionless certification tolerance fraction
        of the realized characteristic angular scale, or ``None``.
    max_boundary_refinements : int
        Positive maximum number of factor-of-two boundary-refinement steps.
    max_attempts : int
        Positive maximum number of rejection draws per realization.

    Returns
    -------
    normalized : dict[str, float]
        New mapping with exactly the float-valued keys ``pixelscale``,
        ``pseudo_caustic_epsilon``, ``geometry_tolerance``,
        ``boundary_tolerance``, and ``fov_expansion_factor``; the additional
        key ``fov`` is present only when ``fov`` was configured. Integer
        settings are validated, while pre-normalized fraction settings are
        consumed only for static relations and are not returned.

    Raises
    ------
    TypeError
        If a normalized scalar cannot be converted to float.
    ValueError
        If an integer setting is outside its built-in ``int`` type/range
        contract, a scalar is non-finite or outside its range, or the FOV,
        pixel-scale, and tolerance relations are inconsistent.

    Notes
    -----
    All normalized angular settings must be finite and strictly positive;
    ``fov_expansion_factor`` must instead be finite and strictly greater than
    one. ``geometry_tolerance`` must be strictly smaller than configured
    ``pixelscale``, while ``boundary_tolerance`` may equal but not undercut
    ``geometry_tolerance``. A configured absolute ``fov`` must exceed
    ``pixelscale`` only when relative pixel scaling is disabled; the relative
    FOV relation is checked after realization. Configured relative geometry
    tolerance must be smaller than relative pixel scale, and relative boundary
    tolerance may equal but not undercut relative geometry tolerance.
    ``max_fov_expansions`` is non-negative,
    ``pseudo_caustic_points`` is at least three, and both
    ``max_boundary_refinements`` and ``max_attempts`` are positive. Unlike the
    image node's integer settings, these four checks use
    ``isinstance(value, int)``: NumPy integer scalars are rejected, while
    booleans follow Python's ``bool``-is-an-``int`` subclass semantics.
    """
    numeric_settings = {
        "pixelscale": pixelscale,
        "pseudo_caustic_epsilon": pseudo_caustic_epsilon,
        "geometry_tolerance": geometry_tolerance,
        "boundary_tolerance": boundary_tolerance,
        "fov_expansion_factor": fov_expansion_factor,
    }
    if fov is not None:
        numeric_settings["fov"] = fov

    normalized = {}
    for name, value in numeric_settings.items():
        try:
            normalized[name] = float(value)
        except (TypeError, ValueError, OverflowError) as err:
            raise TypeError(f"{name} must be a scalar number.") from err
        if name == "fov_expansion_factor":
            if not np.isfinite(normalized[name]) or normalized[name] <= 1.0:
                raise ValueError("fov_expansion_factor must be finite and greater than one.")
        elif not np.isfinite(normalized[name]) or normalized[name] <= 0.0:
            raise ValueError(f"{name} must be finite and positive.")

    if fov is not None and pixelscale_fraction is None and normalized["pixelscale"] >= normalized["fov"]:
        raise ValueError("pixelscale must be smaller than fov.")
    if normalized["geometry_tolerance"] >= normalized["pixelscale"]:
        raise ValueError("geometry_tolerance must be smaller than pixelscale.")
    if normalized["boundary_tolerance"] < normalized["geometry_tolerance"]:
        raise ValueError("boundary_tolerance must be at least geometry_tolerance.")
    if (
        pixelscale_fraction is not None
        and geometry_tolerance_fraction is not None
        and geometry_tolerance_fraction >= pixelscale_fraction
    ):
        raise ValueError("geometry_tolerance_fraction must be smaller than pixelscale_fraction.")
    if (
        geometry_tolerance_fraction is not None
        and boundary_tolerance_fraction is not None
        and boundary_tolerance_fraction < geometry_tolerance_fraction
    ):
        raise ValueError("boundary_tolerance_fraction must be at least geometry_tolerance_fraction.")
    if not isinstance(max_fov_expansions, int) or max_fov_expansions < 0:
        raise ValueError("max_fov_expansions must be a non-negative integer.")
    if not isinstance(pseudo_caustic_points, int) or pseudo_caustic_points < 3:
        raise ValueError("pseudo_caustic_points must be an integer of at least three.")
    if not isinstance(max_boundary_refinements, int) or max_boundary_refinements < 1:
        raise ValueError("max_boundary_refinements must be a positive integer.")
    if not isinstance(max_attempts, int) or max_attempts < 1:
        raise ValueError("max_attempts must be a positive integer.")
    return normalized


def _validate_source_geometry_support(lens_spec, lens, name="lens"):
    """Validate the source-region support tier of a realized lens tree.

    Parameters
    ----------
    lens_spec : CausticsLensSpec
        Registered atomic or recursive specification corresponding exactly to
        ``lens``.
    lens : object
        Realized Caustics atomic lens or ``SinglePlane`` subtree.
    name : str, optional
        Generated positional component name used in diagnostics. Child index
        ``i`` appends ``_<i>`` recursively, beginning with ``"lens"``.

    Returns
    -------
    None
        Every atomic component belongs to the currently certifiable
        source-position geometry tier.

    Raises
    ------
    NotImplementedError
        If any realized EPL component has slope ``t > 1``. The message names
        the recursive component and advises restricting the prior to
        ``t <= 1`` or using ``CausticsLensImageNode`` until the unbounded
        multi-image source geometry is supported.
    ValueError
        If trusted specification and realized composite child counts differ,
        as reported by strict recursive pairing.

    Notes
    -----
    The general EPL model domain is established elsewhere. This validator adds
    only the source-position node's supported tier, ``t <= 1``; the image node
    continues to support steeper physically valid EPL realizations. The first
    unsupported depth-first component raises and stops traversal.
    """
    if lens_spec.model == "EPL":
        slope = _lens_system._caustics_scalar(lens.t.value)
        if slope > 1.0:
            raise NotImplementedError(
                f"Component {name!r} has physically valid EPL slope t={slope} > 1, "
                "but CausticsSourcePositionNode cannot yet certify its unbounded "
                "multi-image geometry. Restrict the prior to t <= 1 or use "
                "CausticsLensImageNode; support is expected in a future extension."
            )
    for index, (child_spec, child_lens) in enumerate(
        zip(lens_spec.parameters.get("lenses", ()), getattr(lens, "lenses", ()), strict=True)
    ):
        _validate_source_geometry_support(child_spec, child_lens, f"{name}_{index}")

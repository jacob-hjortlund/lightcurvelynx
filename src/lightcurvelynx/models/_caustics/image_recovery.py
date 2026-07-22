import numpy as np

from . import runtime as _runtime

_RECOVERY_SEED_POINTS = 256
_RECOVERY_ROOT_REFINEMENTS = 8


def _recovery_neighborhoods_are_empty(coordinates, recovery_points, radius):
    """Identify recovery neighborhoods without a global image.

    Parameters
    ----------
    coordinates : numpy.ndarray, shape (I, 2)
        Trusted global image coordinates in image-plane arcseconds.
    recovery_points : numpy.ndarray, shape (R, 2)
        Trusted registered recovery locations in image-plane arcseconds.
    radius : float
        Positive neighborhood radius in arcseconds.

    Returns
    -------
    numpy.ndarray, shape (R,)
        Boolean mask that is true where every global image is farther than
        ``radius`` from the corresponding recovery point.

    Notes
    -----
    Production callers establish the shapes and positive radius. With no image
    coordinates every recovery neighborhood is empty; with no recovery points
    the returned mask has length zero. A global image exactly ``radius`` from a
    recovery point occupies that neighborhood; emptiness requires every
    distance to be strictly greater.
    """
    distances = np.linalg.norm(
        coordinates[:, None, :] - recovery_points[None, :, :],
        axis=2,
    )
    return np.all(distances > radius, axis=0)


def _validated_recovery_images(
    lens,
    torch,
    image_x,
    image_y,
    recovery_points,
    beta_x,
    beta_y,
    epsilon,
    neighborhood_radius,
):
    """Certify targeted roots by source residual and recovery-point locality.

    Parameters
    ----------
    lens : object
        Realized Caustics lens implementing ``raytrace(x, y)``.
    torch : module
        PyTorch module used by the realized Caustics lens.
    image_x : torch.Tensor, shape (R,)
        Candidate image-plane x coordinates in arcseconds.
    image_y : torch.Tensor, shape (R,)
        Candidate image-plane y coordinates in arcseconds.
    recovery_points : numpy.ndarray, shape (R, 2)
        Originating registered recovery locations in image-plane arcseconds,
        paired one-to-one with the candidates.
    beta_x : torch.Tensor
        Scalar source-plane x coordinate in arcseconds.
    beta_y : torch.Tensor
        Scalar source-plane y coordinate in arcseconds.
    epsilon : float
        Strict upper bound on the source-plane residual in arcseconds.
    neighborhood_radius : float
        Inclusive upper bound on distance from the originating recovery point in
        image-plane arcseconds.

    Returns
    -------
    numpy.ndarray, shape (K, 2)
        Candidate image coordinates certified by both tests, in image-plane
        arcseconds.

    Notes
    -----
    A root is retained only when its source residual is strictly less than
    ``epsilon`` and its recovery-point distance is no greater than
    ``neighborhood_radius``. Paired shapes, output types, and finiteness are
    trusted Caustics and registered-adapter postconditions. Passing candidates
    retain their input order and are not deduplicated.
    """
    candidate_coordinates = np.column_stack((_runtime._to_numpy(image_x), _runtime._to_numpy(image_y)))

    mapped_x, mapped_y = lens.raytrace(
        torch.as_tensor(candidate_coordinates[:, 0], dtype=torch.float64),
        torch.as_tensor(candidate_coordinates[:, 1], dtype=torch.float64),
    )
    mapped_x = _runtime._to_numpy(mapped_x)
    mapped_y = _runtime._to_numpy(mapped_y)

    source_x = float(_runtime._to_numpy(beta_x))
    source_y = float(_runtime._to_numpy(beta_y))
    residuals = np.hypot(mapped_x - source_x, mapped_y - source_y)
    distances = np.linalg.norm(candidate_coordinates - recovery_points, axis=1)
    valid = (residuals < epsilon) & (distances <= neighborhood_radius)
    return candidate_coordinates[valid]


def _refine_image_seeds(
    lens,
    torch,
    seeds,
    recovery_points,
    beta_x,
    beta_y,
    epsilon,
    neighborhood_radius,
):
    """Refine targeted recovery seeds and certify the resulting roots.

    Parameters
    ----------
    lens : object
        Realized Caustics lens implementing ``raytrace(x, y)``.
    torch : module
        PyTorch module used by the realized Caustics lens.
    seeds : numpy.ndarray, shape (R, 2)
        One image-plane seed per active recovery point, in arcseconds.
    recovery_points : numpy.ndarray, shape (R, 2)
        Corresponding registered recovery locations in image-plane arcseconds.
    beta_x : torch.Tensor
        Scalar source-plane x coordinate in arcseconds.
    beta_y : torch.Tensor
        Scalar source-plane y coordinate in arcseconds.
    epsilon : float
        Strict source-plane residual tolerance in arcseconds.
    neighborhood_radius : float
        Inclusive recovery-neighborhood radius in image-plane arcseconds.

    Returns
    -------
    numpy.ndarray, shape (K, 2)
        Refined roots passing source-residual and recovery-point locality
        certification, in image-plane arcseconds.

    Raises
    ------
    ImportError
        If the Caustics root-refinement implementation is unavailable.

    Notes
    -----
    The one-to-one seed/recovery-point ordering is preserved through exactly
    ``_RECOVERY_ROOT_REFINEMENTS`` root-refinement passes before
    certification. The direct ``caustics.lenses.func`` import is not wrapped;
    its original ``ImportError`` propagates unchanged.
    """
    from caustics.lenses.func import forward_raytrace_rootfind

    roots = torch.as_tensor(seeds, dtype=torch.float64)
    for _ in range(_RECOVERY_ROOT_REFINEMENTS):
        roots = forward_raytrace_rootfind(
            roots[:, 0],
            roots[:, 1],
            beta_x,
            beta_y,
            lens.raytrace,
        )
    return _validated_recovery_images(
        lens,
        torch,
        roots[:, 0],
        roots[:, 1],
        recovery_points,
        beta_x,
        beta_y,
        epsilon,
        neighborhood_radius,
    )


def _is_singular_forward_raytrace_error(error):
    """Classify the exact recognized singular linear-solve failure.

    Parameters
    ----------
    error : BaseException
        Exception raised by a Caustics global or targeted image solve.

    Returns
    -------
    bool
        Whether ``error`` is a ``RuntimeError`` whose case-insensitive text
        contains ``linalg.solve`` and either ``input matrix is singular``
        or ``singular u``.

    Notes
    -----
    This narrow predicate controls inner grid-variant progression and must not
    be generalized to unrelated numerical failures.
    """
    if not isinstance(error, RuntimeError):
        return False
    message = str(error).lower()
    return "linalg.solve" in message and ("input matrix is singular" in message or "singular u" in message)


def _is_retryable_forward_raytrace_error(error):
    """Classify the known empty-candidate failure for outer recovery.

    Parameters
    ----------
    error : BaseException
        Exception raised by a Caustics global or targeted image solve.

    Returns
    -------
    bool
        Whether ``error`` is an ``IndexError`` whose case-insensitive text
        contains ``index 0 is out of bounds``.

    Notes
    -----
    Singular linear-solve failures are classified separately by
    ``_is_singular_forward_raytrace_error``. Only this empty-candidate
    failure exits the inner variant sequence immediately for the bounded outer
    schedule.
    """
    return isinstance(error, IndexError) and "index 0 is out of bounds" in str(error).lower()


def _recovery_image_seeds(lens, recovery_points, *, source_x, source_y, radius):
    """Select one targeted image seed per recovery point.

    Parameters
    ----------
    lens : object
        Realized Caustics lens implementing ``raytrace(x, y)``.
    recovery_points : iterable of tuple of float
        Image-plane recovery locations in arcseconds.
    source_x : float
        Source-plane x position in arcseconds.
    source_y : float
        Source-plane y position in arcseconds.
    radius : float
        Image-plane circle radius around each recovery point in arcseconds.

    Returns
    -------
    numpy.ndarray, shape (R, 2)
        One image-plane seed in arcseconds per recovery point.

    Raises
    ------
    ImportError
        If the optional Caustics runtime dependencies needed to raytrace each
        circle are unavailable.

    Notes
    -----
    Each circle contains ``_RECOVERY_SEED_POINTS`` uniformly spaced locations
    over ``[0, 2 pi)``. The retained location minimizes Euclidean source-plane
    distance to ``(source_x, source_y)`` independently for each recovery point;
    ``numpy.argmin`` resolves a tie by first angular position. With no recovery
    points the result has shape ``(0, 2)`` and no optional runtime import is
    attempted.
    """
    angles = 2.0 * np.pi * np.arange(_RECOVERY_SEED_POINTS) / _RECOVERY_SEED_POINTS
    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    source_position = np.array([source_x, source_y], dtype=float)
    seeds = []
    for recovery_point in recovery_points:
        circle = np.asarray(recovery_point, dtype=float) + radius * directions
        mapped_circle = _runtime._raytrace_curve(lens, circle)
        nearest = np.argmin(np.linalg.norm(mapped_circle - source_position, axis=1))
        seeds.append(circle[nearest])
    return np.asarray(seeds, dtype=float).reshape(-1, 2)

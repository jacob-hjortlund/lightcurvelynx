"""Caustics-backed nodes for strong-lens image configurations."""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from citation_compass import CiteClass
from scipy.optimize import linear_sum_assignment

from lightcurvelynx.base_models import FunctionNode

__all__ = ["CausticsLensImageNode", "CausticsSourcePositionNode"]

_RESERVED_LENS_PARAMETERS = {
    "cosmology",
    "name",
    "z_l",
    "z_s",
}
_SINGULAR_SEED_POINTS = 256
_SINGULAR_ROOT_REFINEMENTS = 8


def _validate_lens_configuration(lens_model, lens_parameters):
    """Validate constructor inputs shared by Caustics-backed lens nodes.

    Parameters
    ----------
    lens_model : str
        Name of a lens class exposed by the top-level ``caustics`` package.
    lens_parameters : Mapping
        Mapping from Caustics constructor parameter names to LightCurveLynx
        parameter setters. Values are not required to be numeric until graph
        sampling realizes them.

    Raises
    ------
    TypeError
        If ``lens_model`` is not a non-empty string or ``lens_parameters`` is
        not a mapping.
    ValueError
        If a lens parameter would collide with a constructor argument managed
        internally by the adapter.
    """
    if not isinstance(lens_model, str) or not lens_model:
        raise TypeError("lens_model must be a non-empty Caustics class name.")
    if not isinstance(lens_parameters, Mapping):
        raise TypeError("lens_parameters must be a mapping.")

    collisions = _RESERVED_LENS_PARAMETERS.intersection(lens_parameters)
    if collisions:
        names = ", ".join(sorted(collisions))
        raise ValueError(f"Reserved lens parameter name(s): {names}.")


def _import_caustics_dependencies():
    """Lazily import the optional Caustics runtime dependencies.

    Returns
    -------
    caustics : module
        Imported Caustics package.
    torch : module
        Imported PyTorch backend used by the installed Caustics package.

    Raises
    ------
    ImportError
        If either Caustics or its PyTorch backend cannot be imported. The
        original import error is retained as the exception cause.
    """
    try:
        import caustics
        import torch
    except ImportError as err:  # pragma: no cover
        raise ImportError(
            "Caustics-backed lens nodes require the optional 'caustics' package. "
            "Install it with `pip install caustics`."
        ) from err
    return caustics, torch


def _to_numpy(tensor):
    """Convert a Caustics backend tensor to a CPU NumPy float array.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor returned by Caustics. It may reside on any Torch device or be
        attached to an autograd graph.

    Returns
    -------
    numpy.ndarray
        Detached CPU array with a floating dtype. No copy is made when the
        tensor's existing NumPy representation already has the requested dtype.
    """
    return tensor.detach().cpu().numpy().astype(float, copy=False)


def _sample_value(value, sample_index, num_samples):
    """Extract one lens realization from a sampled graph input.

    Parameters
    ----------
    value : object or array-like
        A scalar/single-system value when ``num_samples == 1`` or an object
        whose first axis indexes graph samples otherwise.
    sample_index : int
        Zero-based sample index to extract from a multi-sample value.
    num_samples : int
        Number of samples represented by the current ``GraphState``.

    Returns
    -------
    object
        ``value`` unchanged for a single-sample state, otherwise
        ``value[sample_index]``.
    """
    if num_samples == 1:
        return value
    return value[sample_index]


def _pixelscale_to_divisions(fov, pixelscale):
    """Convert an image-plane pixel scale to Caustics grid divisions.

    Parameters
    ----------
    fov : float
        Width of the square image-plane search region in arcseconds.
    pixelscale : float
        Requested nominal image-plane grid spacing in arcseconds.

    Returns
    -------
    divisions : int
        Integer grid resolution accepted by Caustics ``forward_raytrace``.

    Notes
    -----
    Rounding up makes the nominal scale, ``fov / divisions``, no larger than
    the requested ``pixelscale``. This preserves Caustics' public convention
    that ``divisions`` is the number of divisions across the field of view.
    """
    return int(np.ceil(float(fov) / float(pixelscale)))


def _validate_optional_positive_fraction(name, value):
    """Return a normalized positive finite scalar fraction or None."""
    if value is None:
        return None
    if isinstance(value, np.ndarray) and value.ndim != 0:
        raise TypeError(f"{name} must be None or a scalar value convertible to float.")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as err:
        raise TypeError(f"{name} must be None or a scalar value convertible to float.") from err
    if not np.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return normalized


def _global_image_coordinates(image_x, image_y):
    """Return one Caustics global result as finite paired coordinates."""
    coordinates_x = _to_numpy(image_x)
    coordinates_y = _to_numpy(image_y)
    if coordinates_x.ndim != 1 or coordinates_y.ndim != 1 or coordinates_x.shape != coordinates_y.shape:
        raise RuntimeError(
            "Caustics forward_raytrace returned image coordinates with invalid "
            f"shapes {coordinates_x.shape} and {coordinates_y.shape}."
        )
    coordinates = np.column_stack((coordinates_x, coordinates_y))
    if not np.all(np.isfinite(coordinates)):
        raise RuntimeError("Caustics forward_raytrace returned non-finite image coordinates.")
    return coordinates


def _singular_neighborhoods_are_empty(coordinates, singular_points, radius):
    """Return which singular points lack a global image within one grid cell."""
    coordinates = np.asarray(coordinates, dtype=float)
    singular_points = np.asarray(singular_points, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("Global image coordinates must have shape (N, 2).")
    if singular_points.ndim != 2 or singular_points.shape[1] != 2:
        raise ValueError("Singular points must have shape (N, 2).")
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("The singular-neighborhood radius must be positive and finite.")
    if not len(singular_points):
        return np.empty(0, dtype=bool)
    if not len(coordinates):
        return np.ones(len(singular_points), dtype=bool)
    distances = np.linalg.norm(
        coordinates[:, None, :] - singular_points[None, :, :],
        axis=2,
    )
    return np.all(distances > radius, axis=0)


def _validated_singular_images(
    lens,
    torch,
    image_x,
    image_y,
    singular_points,
    beta_x,
    beta_y,
    epsilon,
    neighborhood_radius,
):
    """Return finite source-matching roots inside their singular neighborhoods."""
    candidate_x = _to_numpy(image_x)
    candidate_y = _to_numpy(image_y)
    singular_points = np.asarray(singular_points, dtype=float)
    if candidate_x.ndim != 1 or candidate_y.ndim != 1 or candidate_x.shape != candidate_y.shape:
        raise RuntimeError(
            "Caustics singular root finder returned image coordinates with invalid "
            f"shapes {candidate_x.shape} and {candidate_y.shape}."
        )
    candidate_coordinates = np.column_stack((candidate_x, candidate_y))
    if singular_points.shape != candidate_coordinates.shape:
        raise RuntimeError(
            "Caustics singular root finder returned a different number of roots "
            "and originating singular points."
        )

    finite = np.all(np.isfinite(candidate_coordinates), axis=1) & np.all(np.isfinite(singular_points), axis=1)
    candidate_coordinates = candidate_coordinates[finite]
    singular_points = singular_points[finite]
    if not len(candidate_coordinates):
        return np.empty((0, 2), dtype=float)

    mapped_x, mapped_y = lens.raytrace(
        torch.as_tensor(candidate_coordinates[:, 0], dtype=torch.float64),
        torch.as_tensor(candidate_coordinates[:, 1], dtype=torch.float64),
    )
    mapped_x = _to_numpy(mapped_x)
    mapped_y = _to_numpy(mapped_y)
    if mapped_x.ndim != 1 or mapped_y.ndim != 1 or mapped_x.shape != mapped_y.shape:
        raise RuntimeError(
            "Caustics raytrace returned source coordinates with invalid shapes "
            f"{mapped_x.shape} and {mapped_y.shape}."
        )
    if mapped_x.shape != candidate_coordinates[:, 0].shape:
        raise RuntimeError("Caustics raytrace returned a different number of mapped and image coordinates.")

    source_x = float(_to_numpy(beta_x))
    source_y = float(_to_numpy(beta_y))
    residuals = np.hypot(mapped_x - source_x, mapped_y - source_y)
    distances = np.linalg.norm(candidate_coordinates - singular_points, axis=1)
    valid = (
        np.isfinite(mapped_x)
        & np.isfinite(mapped_y)
        & np.isfinite(residuals)
        & (residuals < epsilon)
        & (distances <= neighborhood_radius)
    )
    return candidate_coordinates[valid]


def _refine_image_seeds(
    lens,
    torch,
    seeds,
    singular_points,
    beta_x,
    beta_y,
    epsilon,
    neighborhood_radius,
):
    """Refine and validate targeted roots inside their singular neighborhoods."""
    from caustics.lenses.func import forward_raytrace_rootfind

    roots = torch.as_tensor(seeds, dtype=torch.float64)
    beta_x = torch.as_tensor(beta_x, dtype=torch.float64)
    beta_y = torch.as_tensor(beta_y, dtype=torch.float64)
    for _ in range(_SINGULAR_ROOT_REFINEMENTS):
        roots = forward_raytrace_rootfind(
            roots[:, 0],
            roots[:, 1],
            beta_x,
            beta_y,
            lens.raytrace,
        )
    return _validated_singular_images(
        lens,
        torch,
        roots[:, 0],
        roots[:, 1],
        singular_points,
        beta_x,
        beta_y,
        epsilon,
        neighborhood_radius,
    )


def _is_singular_forward_raytrace_error(error):
    """Return whether Caustics failed in a recognized singular linear solve."""
    message = str(error).lower()
    return (
        isinstance(error, RuntimeError)
        and "linalg.solve" in message
        and ("input matrix is singular" in message or "singular u" in message)
    )


def _is_retryable_forward_raytrace_error(error):
    """Return whether a Caustics image-root search has a known retryable failure."""
    if isinstance(error, IndexError):
        message = str(error).lower()
        return "index 0 is out of bounds" in message
    return _is_singular_forward_raytrace_error(error)


def _validate_lens_redshifts(values):
    """Validate redshifts from one realized Caustics-node input mapping.

    Parameters
    ----------
    values : Mapping
        Numeric inputs for one lens-system sample. The mapping must contain
        dimensionless ``lens_redshift`` and ``source_redshift`` entries.

    Returns
    -------
    lens_redshift : float
        Finite, non-negative lens redshift.
    source_redshift : float
        Finite source redshift strictly greater than ``lens_redshift``.

    Raises
    ------
    KeyError
        If either required redshift is absent.
    TypeError
        If a redshift cannot be converted to a scalar float.
    ValueError
        If a redshift is non-finite or the lens/source ordering is invalid.
    """
    z_l = float(values["lens_redshift"])
    z_s = float(values["source_redshift"])
    if not np.isfinite(z_l) or not np.isfinite(z_s):
        raise ValueError("Lens and source redshifts must be finite.")
    if z_l < 0.0 or z_s <= z_l:
        raise ValueError(f"Expected 0 <= lens_redshift < source_redshift; got {z_l} and {z_s}.")
    return z_l, z_s


def _construct_caustics_lens(
    *,
    lens_model,
    cosmology,
    values,
    lens_parameter_names,
):
    """Construct one Caustics lens from a single graph realization.

    Parameters
    ----------
    lens_model : str
        Name of the Caustics lens class to instantiate.
    cosmology : caustics.Cosmology
        Fixed cosmology object supplied to the Caustics lens constructor.
    values : Mapping
        Numeric inputs for one lens-system sample. It must contain
        ``lens_redshift``, ``source_redshift``, and one ``lens_<name>`` entry
        for every name in ``lens_parameter_names``.
    lens_parameter_names : iterable of str
        Caustics constructor parameter names without the ``lens_`` graph-input
        prefix.

    Returns
    -------
    lens : object
        Newly constructed Caustics lens for this realization. The object is not
        cached on a LightCurveLynx node.
    torch : module
        Imported PyTorch module, returned so callers can create compatible
        ``float64`` inputs without importing it eagerly.

    Raises
    ------
    ImportError
        If the optional Caustics runtime dependencies are unavailable.
    KeyError
        If a required realized input is missing from ``values``.
    ValueError
        If redshifts are invalid or ``lens_model`` is not exposed by Caustics.
    """
    caustics, torch = _import_caustics_dependencies()
    z_l, z_s = _validate_lens_redshifts(values)

    # TODO: Currently only supports base lens classes in Caustics. Implement a
    # helper for compound lens configurations such as SIE plus external shear.
    try:
        lens_class = getattr(caustics, lens_model)
    except AttributeError as err:
        raise ValueError(f"Unknown Caustics lens model '{lens_model}'.") from err

    dtype = torch.float64
    lens_kwargs = {
        name: torch.as_tensor(values[f"lens_{name}"], dtype=dtype) for name in lens_parameter_names
    }
    lens = lens_class(
        name="lens",
        cosmology=cosmology,
        z_l=torch.as_tensor(z_l, dtype=dtype),
        z_s=torch.as_tensor(z_s, dtype=dtype),
        **lens_kwargs,
    )
    return lens, torch


def _import_contourpy():
    """Lazily import the contour implementation used for critical curves.

    Returns
    -------
    contourpy : module
        Imported ContourPy package.

    Raises
    ------
    ImportError
        If ContourPy is unavailable. The original import error is retained as
        the exception cause.
    """
    try:
        import contourpy
    except ImportError as err:  # pragma: no cover
        raise ImportError(
            "Caustics source-position sampling requires the optional 'contourpy' "
            "package. Install it with `pip install contourpy`."
        ) from err
    return contourpy


def _lens_plane_origin(values):
    """Read the image-plane search origin from one realized input mapping.

    Parameters
    ----------
    values : Mapping
        Numeric inputs for one lens-system sample. ``lens_x0`` and ``lens_y0``
        are interpreted as angular offsets in arcseconds and each defaults to
        zero when absent.

    Returns
    -------
    x0 : float
        Image-plane x origin in arcseconds.
    y0 : float
        Image-plane y origin in arcseconds.

    Raises
    ------
    TypeError
        If either coordinate cannot be converted to a scalar float.
    ValueError
        If either coordinate is non-finite.
    """
    x0 = float(values.get("lens_x0", 0.0))
    y0 = float(values.get("lens_y0", 0.0))
    if not np.isfinite(x0) or not np.isfinite(y0):
        raise ValueError("The lens-plane origin must be finite.")
    return x0, y0


def _raytrace_curve(lens, coordinates):
    """Map one image-plane curve through the Caustics lens equation.

    Parameters
    ----------
    lens : object
        Realized Caustics lens implementing ``raytrace(x, y)``.
    coordinates : array-like, shape (N, 2)
        Image-plane x/y angular offsets in arcseconds.

    Returns
    -------
    numpy.ndarray, shape (N, 2)
        Source-plane x/y angular offsets in arcseconds, stored as finite CPU
        floating-point values.

    Raises
    ------
    ValueError
        If ``coordinates`` does not have shape ``(N, 2)``.
    RuntimeError
        If raytracing changes the expected shape or produces non-finite values.
    """
    _, torch = _import_caustics_dependencies()
    coordinates = np.asarray(coordinates, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("A lens-plane curve must have shape (N, 2).")

    source_x, source_y = lens.raytrace(
        torch.as_tensor(coordinates[:, 0], dtype=torch.float64),
        torch.as_tensor(coordinates[:, 1], dtype=torch.float64),
    )
    source_curve = np.column_stack((_to_numpy(source_x), _to_numpy(source_y)))
    if source_curve.shape != coordinates.shape or not np.all(np.isfinite(source_curve)):
        raise RuntimeError("Caustics returned an invalid source-plane boundary.")
    return source_curve


class _PointSingularityGeometryAdapter:
    """Enumerate pseudo-caustics for a lens with one point singularity.

    This private adapter describes the singular geometry shared by the Caustics
    SIE and SIS implementations. An unsoftened lens (``lens_s == 0``) has one
    singular point at ``(lens_x0, lens_y0)``. Its pseudo-caustic is obtained by
    raytracing successively smaller image-plane circles around that point until
    the source-plane boundary changes by no more than ``geometry_tolerance``.
    A softened lens (``lens_s > 0``) is continuous at its center and therefore
    contributes no pseudo-caustic through this adapter.

    Notes
    -----
    The adapter receives numeric values for a single graph sample. It never
    reads a ``GraphState`` directly and does not retain a realized lens or
    extracted boundary as mutable state.
    """

    _MAX_REFINEMENTS = 32

    @staticmethod
    def singular_points(values):
        """Return singular image-plane locations for one lens realization.

        Parameters
        ----------
        values : Mapping
            Numeric inputs for one lens-system sample. ``lens_s`` is the
            softening radius in arcseconds and defaults to zero. Unsoftened
            lenses must provide ``lens_x0`` and ``lens_y0`` in arcseconds.

        Returns
        -------
        tuple of tuple of float
            Empty for a softened lens, otherwise ``((x0, y0),)`` containing the
            singular image-plane location in arcseconds.

        Raises
        ------
        ValueError
            If the softening radius is negative/non-finite or an unsoftened
            realization omits either center coordinate.
        """
        softening = float(values.get("lens_s", 0.0))
        if not np.isfinite(softening) or softening < 0.0:
            raise ValueError("The lens softening radius must be finite and non-negative.")
        if softening > 0.0:
            return ()
        if "lens_x0" not in values or "lens_y0" not in values:
            raise ValueError("Point-singularity geometry requires lens_parameters entries for 'x0' and 'y0'.")
        return (_lens_plane_origin(values),)

    def singular_image_seeds(self, lens, values, *, source_x, source_y, radius):
        """Return one nearest mapped-circle seed per active point singularity."""
        singular_points = self.singular_points(values)
        if not singular_points:
            return np.empty((0, 2), dtype=float)
        angles = 2.0 * np.pi * np.arange(_SINGULAR_SEED_POINTS) / _SINGULAR_SEED_POINTS
        directions = np.column_stack((np.cos(angles), np.sin(angles)))
        source_position = np.array([source_x, source_y], dtype=float)
        seeds = []
        for singular_point in singular_points:
            circle = np.asarray(singular_point, dtype=float) + radius * directions
            mapped_circle = _raytrace_curve(lens, circle)
            nearest = np.argmin(np.linalg.norm(mapped_circle - source_position, axis=1))
            seeds.append(circle[nearest])
        return np.asarray(seeds, dtype=float)

    @staticmethod
    def expected_num_images(
        source_x,
        source_y,
        *,
        caustic_curves,
        pseudo_caustic_curves,
        geometry_tolerance,
    ):
        """Count every regular SIE/SIS image from typed boundary containment."""
        shapely = _import_shapely()
        true_regions = _boundary_regions(
            caustic_curves,
            geometry_tolerance=geometry_tolerance,
        )
        pseudo_regions = _boundary_regions(
            pseudo_caustic_curves,
            geometry_tolerance=geometry_tolerance,
        )
        count = 1
        count += 2 * sum(bool(shapely.contains_xy(region, source_x, source_y)) for region in true_regions)
        count += sum(bool(shapely.contains_xy(region, source_x, source_y)) for region in pseudo_regions)
        return count

    def pseudo_caustics(
        self,
        lens,
        values,
        *,
        num_points,
        epsilon,
        geometry_tolerance,
    ):
        """Trace every converged pseudo-caustic for one lens realization.

        Parameters
        ----------
        lens : object
            Realized Caustics lens implementing ``raytrace(x, y)``.
        values : Mapping
            Numeric inputs for the same single lens-system realization used to
            construct ``lens``. The adapter reads ``lens_s``, ``lens_x0``, and
            ``lens_y0`` to determine its singular geometry; other realized
            entries may remain in the mapping and are ignored here.
        num_points : int
            Number of unique, evenly spaced vertices on each image-plane loop.
        epsilon : float
            Initial loop radius around each singularity in arcseconds.
        geometry_tolerance : float
            Maximum allowed pointwise source-plane change, in arcseconds,
            between successive loop refinements.

        Returns
        -------
        list of numpy.ndarray
            One closed source-plane boundary per singularity. Each array has
            shape ``(num_points + 1, 2)`` in arcseconds, with the first vertex
            repeated at the end. A softened lens returns an empty list.

        Raises
        ------
        ValueError
            If the realized singularity configuration is invalid.
        RuntimeError
            If raytracing produces an invalid boundary or the shrinking-loop
            sequence does not converge within the bounded refinement count.
        """
        singular_points = self.singular_points(values)
        if not singular_points:
            return []

        angles = 2.0 * np.pi * np.arange(num_points, dtype=float) / num_points
        directions = np.column_stack((np.cos(angles), np.sin(angles)))
        pseudo_caustics = []

        for singular_x, singular_y in singular_points:
            center = np.array([singular_x, singular_y], dtype=float)
            radius = float(epsilon)
            previous_curve = _raytrace_curve(lens, center + radius * directions)
            last_change = np.inf

            for _ in range(self._MAX_REFINEMENTS):
                radius *= 0.5
                current_curve = _raytrace_curve(lens, center + radius * directions)
                last_change = float(np.max(np.linalg.norm(current_curve - previous_curve, axis=1)))
                if last_change <= geometry_tolerance:
                    pseudo_caustics.append(np.concatenate((current_curve, current_curve[:1]), axis=0))
                    break
                previous_curve = current_curve
            else:
                raise RuntimeError(
                    "Pseudo-caustic extraction did not converge after "
                    f"{self._MAX_REFINEMENTS} refinements; final boundary change "
                    f"was {last_change} arcsec."
                )

        return pseudo_caustics


_INITIAL_FOV_PADDING = 1.1


def _positive_lens_parameter(values, name):
    """Return one finite positive realized lens parameter as a float."""
    key = f"lens_{name}"
    try:
        value = float(values[key])
    except KeyError as err:
        raise ValueError(f"Realized lens parameter '{name}' is required for lens geometry.") from err
    except (TypeError, ValueError) as err:
        raise TypeError(f"Realized lens parameter '{name}' must be a scalar number.") from err
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"Realized lens parameter '{name}' must be finite and positive.")
    return value


class _SIEGeometryAdapter(_PointSingularityGeometryAdapter):
    """Complete singular geometry and initial-FOV policy for Caustics SIE."""

    @staticmethod
    def characteristic_angular_scale(values):
        """Return the realized positive Einstein radius in arcseconds."""
        return _positive_lens_parameter(values, "Rein")

    @staticmethod
    def initial_fov(values):
        """Return a padded analytic critical-curve diameter in arcseconds."""
        einstein_radius = _positive_lens_parameter(values, "Rein")
        axis_ratio = _positive_lens_parameter(values, "q")
        if axis_ratio > 1.0:
            raise ValueError("Realized SIE lens parameter 'q' must be no greater than one.")
        return 2.0 * _INITIAL_FOV_PADDING * einstein_radius / np.sqrt(axis_ratio)


class _SISGeometryAdapter(_PointSingularityGeometryAdapter):
    """Complete singular geometry and initial-FOV policy for Caustics SIS."""

    @staticmethod
    def characteristic_angular_scale(values):
        """Return the realized positive Einstein radius in arcseconds."""
        return _positive_lens_parameter(values, "Rein")

    @staticmethod
    def initial_fov(values):
        """Return a padded analytic critical-curve diameter in arcseconds."""
        einstein_radius = _positive_lens_parameter(values, "Rein")
        return 2.0 * _INITIAL_FOV_PADDING * einstein_radius


_LENS_GEOMETRY_ADAPTERS = {
    "SIE": _SIEGeometryAdapter(),
    "SIS": _SISGeometryAdapter(),
}


def _get_lens_geometry_adapter(lens_model):
    """Look up the certified singular-geometry adapter for a lens class.

    Parameters
    ----------
    lens_model : str
        Name of the Caustics lens class used to construct the realized lens.

    Returns
    -------
    object
        Private adapter implementing ``singular_points`` and
        ``pseudo_caustics`` for every singular boundary of that lens class.

    Raises
    ------
    ValueError
        If no complete geometry adapter is registered. Explicit source
        positions may still use such a model through ``CausticsLensImageNode``.
    """
    try:
        return _LENS_GEOMETRY_ADAPTERS[lens_model]
    except KeyError as err:
        supported = ", ".join(sorted(_LENS_GEOMETRY_ADAPTERS))
        raise ValueError(
            "Source-position sampling has no complete pseudo-caustic geometry "
            f"adapter for Caustics lens model '{lens_model}'. Supported models: "
            f"{supported}."
        ) from err


class _CausticFOVError(RuntimeError):
    """Signal that a larger image-plane FOV is required for completeness."""


def _outer_grid_boundary(values):
    """Return every outer-boundary value from a square grid without duplicates."""
    values = np.asarray(values)
    if values.ndim < 2 or values.shape[0] != values.shape[1]:
        raise ValueError("Expected a square grid with at least two dimensions.")
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
    singular_points=(),
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
    singular_points : iterable of tuple of float, optional
        Image-plane singular locations in arcseconds. One grid-spacing radius
        around each point is masked so a discontinuity is not misidentified as
        a true critical curve.

    Returns
    -------
    list of numpy.ndarray
        Closed source-plane caustic curves. Each element has shape ``(N, 2)``
        and contains x/y offsets in arcseconds. Disconnected critical curves
        remain separate list entries.

    Raises
    ------
    ImportError
        If Caustics, Torch, or ContourPy is unavailable.
    TypeError
        If the lens lacks the required protocol or returns a Jacobian with an
        unsupported representation.
    RuntimeError
        If the grid/Jacobian is invalid, a contour is malformed or open, or
        raytracing yields an invalid caustic. The ``_CausticFOVError`` subclass
        is raised when no critical curves are found, a curve reaches the
        boundary, or the boundary Jacobian is not positive definite.
    """
    contourpy = _import_contourpy()
    _, torch = _import_caustics_dependencies()

    for method_name in ("jacobian_lens_equation", "raytrace"):
        if not hasattr(lens, method_name):
            raise TypeError(f"Caustics source-position sampling requires lens method '{method_name}'.")

    center_x, center_y = (float(value) for value in center)
    num_intervals = int(np.ceil(fov / pixelscale))
    if num_intervals % 2:
        num_intervals += 1
    actual_pixelscale = float(fov) / num_intervals
    half_fov = 0.5 * float(fov)
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
    if not isinstance(jacobian, torch.Tensor) or jacobian.shape[-2:] != (2, 2):
        raise TypeError("Caustics jacobian_lens_equation must return a tensor with trailing shape (2, 2).")
    determinant = _to_numpy(torch.linalg.det(jacobian))
    if determinant.shape != (num_intervals + 1, num_intervals + 1):
        raise RuntimeError("Caustics returned a lens-equation Jacobian with an unexpected grid shape.")

    x_coordinates = _to_numpy(x_axis)
    y_coordinates = _to_numpy(y_axis)
    invalid = ~np.isfinite(determinant)
    for singular_x, singular_y in singular_points:
        squared_distance = (x_coordinates[np.newaxis, :] - singular_x) ** 2 + (
            y_coordinates[:, np.newaxis] - singular_y
        ) ** 2
        invalid |= squared_distance <= actual_pixelscale**2

    if np.all(invalid):
        raise RuntimeError("The lens-equation Jacobian grid contains no finite values.")

    symmetric_jacobian = 0.5 * (jacobian + jacobian.transpose(-1, -2))
    eigenvalues = _to_numpy(torch.linalg.eigvalsh(symmetric_jacobian))
    if eigenvalues.shape != (num_intervals + 1, num_intervals + 1, 2):
        raise RuntimeError("Caustics returned unexpected lens-equation Jacobian eigenvalue shapes.")

    boundary_invalid = _outer_grid_boundary(invalid)
    boundary_eigenvalues = _outer_grid_boundary(eigenvalues)
    valid_boundary = ~boundary_invalid
    if not np.any(valid_boundary):
        raise RuntimeError("The lens-equation Jacobian grid boundary contains no finite values.")
    if not np.all(np.isfinite(boundary_eigenvalues[valid_boundary])):
        raise RuntimeError("The lens-equation Jacobian grid boundary contains invalid eigenvalues.")
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

        caustic_curve = _raytrace_curve(lens, critical_curve)
        caustic_gap = float(np.linalg.norm(caustic_curve[0] - caustic_curve[-1]))
        if caustic_gap > geometry_tolerance:
            raise RuntimeError(f"A mapped caustic is open with endpoint gap {caustic_gap} arcsec.")
        caustic_curves.append(caustic_curve)

    return caustic_curves


def _find_all_pseudo_caustics(
    lens,
    *,
    lens_model,
    values,
    num_points,
    epsilon,
    geometry_tolerance,
):
    """Extract all certified pseudo-caustics for one lens realization.

    Parameters
    ----------
    lens : object
        Realized Caustics lens used for source-plane raytracing.
    lens_model : str
        Caustics class name used to select a complete singular-geometry adapter.
    values : Mapping
        Numeric inputs for exactly one lens-system sample, normally produced by
        selecting one index from ``FunctionNode._build_inputs``. Keys retain
        their registered graph-input names: ``lens_redshift``,
        ``source_redshift``, and ``lens_<parameter>`` for each Caustics
        constructor parameter. The selected adapter reads the subset needed to
        enumerate singularities, such as ``lens_s``, ``lens_x0``, and
        ``lens_y0`` for SIE/SIS.
    num_points : int
        Number of unique vertices used to trace each singular boundary.
    epsilon : float
        Initial image-plane offset from a singular boundary in arcseconds.
    geometry_tolerance : float
        Required source-plane convergence tolerance in arcseconds.

    Returns
    -------
    list of numpy.ndarray
        Closed pseudo-caustic boundaries in source-plane arcseconds. Boundary
        arrays may have different vertex counts for different future adapters.

    Raises
    ------
    ValueError
        If ``lens_model`` has no complete geometry adapter or the realized
        singular geometry is invalid.
    RuntimeError
        If boundary raytracing is invalid or fails to converge.
    """
    adapter = _get_lens_geometry_adapter(lens_model)
    return adapter.pseudo_caustics(
        lens,
        values,
        num_points=num_points,
        epsilon=epsilon,
        geometry_tolerance=geometry_tolerance,
    )


def _import_shapely():
    """Lazily import Shapely for source-plane topology operations.

    Returns
    -------
    shapely : module
        Imported Shapely package.

    Raises
    ------
    ImportError
        If Shapely is unavailable. The original import error is retained as the
        exception cause.
    """
    try:
        import shapely
    except ImportError as err:  # pragma: no cover
        raise ImportError(
            "Caustics source-position sampling requires the optional 'shapely' "
            "package. Install it with `pip install shapely`."
        ) from err
    return shapely


@dataclass(frozen=True)
class _BoundaryGeometry:
    caustic_curves: tuple[np.ndarray, ...]
    pseudo_caustic_curves: tuple[np.ndarray, ...]
    critical_curve_fov: float
    pixelscale: float
    pseudo_caustic_points: int


def _close_curve(curve, *, tolerance):
    """Validate, normalize, and close one numerical source-plane boundary.

    Parameters
    ----------
    curve : array-like, shape (N, 2)
        Source-plane x/y coordinates in arcseconds.
    tolerance : float
        Maximum permitted endpoint gap and minimum retained separation between
        cyclic consecutive vertices, in arcseconds.

    Returns
    -------
    numpy.ndarray, shape (M, 2)
        Finite floating-point coordinates with at least three unique vertices
        and an exactly repeated first/last vertex. Consecutive vertices within
        ``tolerance`` of one another are removed, so ``M`` may be smaller than
        ``N``.

    Raises
    ------
    ValueError
        If ``tolerance`` is not finite and positive.
    RuntimeError
        If the coordinates are malformed/non-finite, contain fewer than three
        unique vertices after normalization, or have an endpoint gap larger
        than ``tolerance``.
    """
    tolerance = float(tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("Curve closure tolerance must be finite and positive.")

    coordinates = np.asarray(curve, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise RuntimeError("A caustic boundary must have shape (N, 2).")
    if len(coordinates) < 3 or not np.all(np.isfinite(coordinates)):
        raise RuntimeError("A caustic boundary must contain at least three finite vertices.")

    endpoint_gap = float(np.linalg.norm(coordinates[0] - coordinates[-1]))
    if endpoint_gap > tolerance:
        raise RuntimeError(
            "A caustic boundary is open; endpoint gap "
            f"{endpoint_gap} arcsec exceeds tolerance {tolerance} arcsec."
        )
    if endpoint_gap > 0.0:
        coordinates = np.concatenate((coordinates, coordinates[:1]), axis=0)

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


def _extract_polygonal_geometry(geometry):
    """Retain every polygonal part of a repaired Shapely geometry.

    Parameters
    ----------
    geometry : shapely.Geometry
        Geometry returned by a Shapely construction or validity-repair
        operation.

    Returns
    -------
    shapely.Polygon or shapely.MultiPolygon or shapely.GeometryCollection
        All polygonal components, merged without replacing concavities or holes.
        An empty input returns an empty ``GeometryCollection``.

    Raises
    ------
    RuntimeError
        If a non-empty point or line component remains after validity repair,
        or if merging polygonal components produces a non-polygonal result.

    Notes
    -----
    Rejecting non-polygonal remnants is deliberately conservative. Such parts
    can indicate collapsed or ambiguous input linework, which must not silently
    reduce the inferred strong-lensing cross-section.
    """
    shapely = _import_shapely()
    if geometry.is_empty:
        return shapely.GeometryCollection()
    if geometry.geom_type == "Polygon":
        return geometry
    if geometry.geom_type == "MultiPolygon":
        return geometry

    polygons = []
    non_polygonal_types = []

    def collect_parts(current_geometry):
        if current_geometry.is_empty:
            return
        if current_geometry.geom_type == "Polygon":
            polygons.append(current_geometry)
        elif current_geometry.geom_type in {"MultiPolygon", "GeometryCollection"}:
            for part in current_geometry.geoms:
                collect_parts(part)
        else:
            non_polygonal_types.append(current_geometry.geom_type)

    collect_parts(geometry)
    if non_polygonal_types:
        names = ", ".join(sorted(set(non_polygonal_types)))
        raise RuntimeError(f"Validity repair left non-polygonal caustic geometry component(s): {names}.")
    if not polygons:
        return shapely.GeometryCollection()

    polygonal_geometry = shapely.union_all(polygons)
    if polygonal_geometry.geom_type not in {"Polygon", "MultiPolygon"}:
        raise RuntimeError(
            "Expected polygonal geometry after merging repaired components, got "
            f"{polygonal_geometry.geom_type}."
        )
    return polygonal_geometry


def _boundary_regions(curves, *, geometry_tolerance):
    """Return one repaired positive-area polygonal region per curve."""
    regions = []
    for boundary_index, curve in enumerate(curves):
        coordinates = _close_curve(curve, tolerance=geometry_tolerance)
        region = _extract_polygonal_geometry(
            _import_shapely().make_valid(_import_shapely().Polygon(coordinates))
        )
        area = float(region.area)
        if region.is_empty or not np.isfinite(area) or area <= 0.0:
            raise RuntimeError(
                f"Caustic boundary {boundary_index} did not enclose a finite positive-area polygonal region."
            )
        regions.append(region)
    return tuple(regions)


def _match_boundary_curves(reference_curves, candidate_curves, *, geometry_tolerance):
    """Match one typed boundary set and return ordered candidates and displacement."""
    if len(reference_curves) != len(candidate_curves):
        return tuple(candidate_curves), np.inf, False
    if not reference_curves:
        return (), 0.0, True
    shapely = _import_shapely()
    reference_lines = [
        shapely.LineString(_close_curve(curve, tolerance=geometry_tolerance)) for curve in reference_curves
    ]
    candidate_lines = [
        shapely.LineString(_close_curve(curve, tolerance=geometry_tolerance)) for curve in candidate_curves
    ]
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
    return ordered, displacement, True


def _boundary_topology_signature(geometry, *, geometry_tolerance):
    """Return typed component, ring, and pairwise-relation topology."""
    true_regions = _boundary_regions(
        geometry.caustic_curves,
        geometry_tolerance=geometry_tolerance,
    )
    pseudo_regions = _boundary_regions(
        geometry.pseudo_caustic_curves,
        geometry_tolerance=geometry_tolerance,
    )

    def region_counts(region):
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
    """Match boundary snapshots and compare their displacement and topology."""
    caustic_curves, caustic_displacement, caustic_counts_stable = _match_boundary_curves(
        previous.caustic_curves,
        current.caustic_curves,
        geometry_tolerance=geometry_tolerance,
    )
    pseudo_caustic_curves, pseudo_displacement, pseudo_counts_stable = _match_boundary_curves(
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
    )
    displacement = max(caustic_displacement, pseudo_displacement)
    typed_counts_stable = caustic_counts_stable and pseudo_counts_stable
    topology_stable = typed_counts_stable and _boundary_topology_signature(
        previous,
        geometry_tolerance=geometry_tolerance,
    ) == _boundary_topology_signature(
        current,
        geometry_tolerance=geometry_tolerance,
    )
    return current, displacement, topology_stable


def _source_boundary_clearance(source_x, source_y, geometry, geometry_tolerance):
    """Return source distance to the nearest typed boundary in arcseconds."""
    curves = (*geometry.caustic_curves, *geometry.pseudo_caustic_curves)
    if not curves:
        return np.inf
    shapely = _import_shapely()
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
        True-caustic source-plane boundaries in arcseconds.
    pseudo_caustic_curves : iterable of array-like
        Pseudo-caustic source-plane boundaries in arcseconds.
    geometry_tolerance : float
        Endpoint closure tolerance and Shapely precision-grid spacing in
        arcseconds.

    Returns
    -------
    shapely.Polygon or shapely.MultiPolygon
        Valid, finite, positive-area strong-lensing geometry. Disconnected
        components, concavities, and holes are retained.

    Raises
    ------
    ImportError
        If Shapely is unavailable.
    RuntimeError
        If no boundaries exist, a boundary cannot be interpreted as polygonal,
        or the final union is empty, non-polygonal, non-finite, or has
        non-positive area.

    Notes
    -----
    Each boundary is converted to its own interior before union. This avoids
    accepting bounded faces that are collectively formed by several curves but
    lie inside none of the individual caustic interiors.
    """
    shapely = _import_shapely()
    curves = [*caustic_curves, *pseudo_caustic_curves]
    if not curves:
        raise RuntimeError("No caustic or pseudo-caustic boundaries were found.")

    enclosed_regions = _boundary_regions(
        curves,
        geometry_tolerance=geometry_tolerance,
    )

    region = shapely.union_all(enclosed_regions, grid_size=geometry_tolerance)
    region = shapely.make_valid(region)
    region = _extract_polygonal_geometry(region)
    area = float(region.area)
    if region.is_empty or not np.isfinite(area) or area <= 0.0:
        raise RuntimeError("The strong-lensing region has no finite positive area.")
    if region.geom_type not in {"Polygon", "MultiPolygon"}:
        raise RuntimeError(f"Expected polygonal strong-lensing geometry, got {region.geom_type}.")
    return region


def _sample_position(
    region,
    rng,
    *,
    max_attempts,
    lens_identifier,
    geometry_settings,
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
    RuntimeError
        If the region has invalid bounds/area or no point is accepted within
        ``max_attempts``.
    """
    shapely = _import_shapely()
    bounds = tuple(float(value) for value in region.bounds)
    area = float(region.area)
    if len(bounds) != 4 or not np.all(np.isfinite(bounds)):
        raise RuntimeError(f"Strong-lensing region for {lens_identifier} has invalid bounds.")
    min_x, min_y, max_x, max_y = bounds
    bounding_box_area = (max_x - min_x) * (max_y - min_y)
    if not np.isfinite(area) or area <= 0.0 or not np.isfinite(bounding_box_area) or bounding_box_area <= 0.0:
        raise RuntimeError(
            f"Strong-lensing region for {lens_identifier} has invalid area "
            f"{area} or bounding-box area {bounding_box_area}."
        )

    for attempt in range(1, max_attempts + 1):
        source_x = float(rng.uniform(min_x, max_x))
        source_y = float(rng.uniform(min_y, max_y))
        if bool(shapely.contains_xy(region, source_x, source_y)):
            return source_x, source_y, area, attempt

    settings = ", ".join(f"{name}={value}" for name, value in geometry_settings.items())
    raise RuntimeError(
        "Unable to sample the strong-lensing region for "
        f"{lens_identifier} after {max_attempts} attempts; bounding-box area="
        f"{bounding_box_area} arcsec^2, polygon area={area} arcsec^2, {settings}."
    )


def _validate_source_position_configuration(
    *,
    lens_model,
    lens_parameters,
    fov,
    pixelscale,
    pixelscale_fraction,
    max_fov_expansions,
    pseudo_caustic_points,
    pseudo_caustic_epsilon,
    geometry_tolerance,
    boundary_tolerance,
    max_boundary_refinements,
    max_attempts,
):
    """Validate immutable source-position geometry and sampling settings.

    Parameters correspond to the CausticsSourcePositionNode constructor
    arguments. Angular configuration values are measured in arcseconds.

    Raises
    ------
    TypeError
        If shared lens configuration has the wrong type or an angular setting
        cannot be converted to a scalar number.
    ValueError
        If shared lens configuration is invalid, the lens lacks a complete
        geometry adapter, a count setting is not a valid integer in its allowed
        range, or a numerical setting is non-finite, outside its allowed range,
        or inconsistent with another setting.
    """
    _validate_lens_configuration(lens_model, lens_parameters)
    _get_lens_geometry_adapter(lens_model)

    numeric_settings = {
        "pixelscale": pixelscale,
        "pseudo_caustic_epsilon": pseudo_caustic_epsilon,
        "geometry_tolerance": geometry_tolerance,
        "boundary_tolerance": boundary_tolerance,
    }
    if fov is not None:
        numeric_settings["fov"] = fov

    normalized = {}
    for name, value in numeric_settings.items():
        try:
            normalized[name] = float(value)
        except (TypeError, ValueError) as err:
            raise TypeError(f"{name} must be a scalar number.") from err
        if not np.isfinite(normalized[name]) or normalized[name] <= 0.0:
            raise ValueError(f"{name} must be finite and positive.")

    if fov is not None and pixelscale_fraction is None and normalized["pixelscale"] >= normalized["fov"]:
        raise ValueError("pixelscale must be smaller than fov.")
    if normalized["geometry_tolerance"] >= normalized["pixelscale"]:
        raise ValueError("geometry_tolerance must be smaller than pixelscale.")
    if normalized["boundary_tolerance"] < normalized["geometry_tolerance"]:
        raise ValueError("boundary_tolerance must be at least geometry_tolerance.")
    if (
        not isinstance(max_fov_expansions, int)
        or isinstance(max_fov_expansions, bool)
        or max_fov_expansions < 0
    ):
        raise ValueError("max_fov_expansions must be a non-negative integer.")
    if not isinstance(pseudo_caustic_points, int) or pseudo_caustic_points < 3:
        raise ValueError("pseudo_caustic_points must be an integer of at least three.")
    if (
        not isinstance(max_boundary_refinements, int)
        or isinstance(max_boundary_refinements, bool)
        or max_boundary_refinements < 1
    ):
        raise ValueError("max_boundary_refinements must be a positive integer.")
    if not isinstance(max_attempts, int) or max_attempts < 1:
        raise ValueError("max_attempts must be a positive integer.")


class CausticsSourcePositionNode(FunctionNode, CiteClass):
    """Uniformly sample the complete geometric strong-lensing source region.

    For each realized lens configuration, this node extracts all supported true
    caustics and pseudo-caustics, constructs the union of their source-plane
    interiors, and samples one position uniformly in area. The realized point,
    geometric cross-section, rejection-attempt count, expected mathematical
    image count, and boundary-certification diagnostics are persisted in the
    node's ``GraphState`` entries.

    Parameters
    ----------
    lens_model : str
        Name of a Caustics lens class with a registered complete geometry
        adapter.
    cosmology : caustics.Cosmology
        Fixed cosmology used to construct each realized lens.
    lens_redshift : parameter
        Dimensionless lens-redshift setter.
    source_redshift : parameter
        Dimensionless source-redshift setter.
    lens_parameters : Mapping
        Caustics constructor parameter names mapped to LightCurveLynx setters.
        Every entry is registered separately to preserve graph dependencies.
    fov : float or None, optional
        Initial image-plane critical-curve search width in arcseconds. When
        None, the registered lens geometry adapter derives a starting width
        from each realized lens configuration.
    pixelscale : float, optional
        Maximum configured image-plane Jacobian-grid spacing in arcseconds.
        When ``pixelscale_fraction`` is enabled, this becomes an upper bound on
        the realized per-lens spacing.
    pixelscale_fraction : float or None, optional
        Maximum initial Jacobian-grid spacing as a fraction of the realized
        adapter-provided characteristic angular scale. When enabled, the smaller
        of this relative scale and ``pixelscale`` is used for each lens.
    max_fov_expansions : int, optional
        Maximum number of factor-of-two FOV expansions after the initial
        attempt. Larger values can increase two-dimensional grid cost rapidly.
    pseudo_caustic_points : int, optional
        Unique vertices used for each mapped singular boundary.
    pseudo_caustic_epsilon : float, optional
        Initial image-plane offset from singular boundaries in arcseconds.
    geometry_tolerance : float, optional
        Curve-closure, convergence, and topology precision in arcseconds.
    boundary_tolerance : float, optional
        Maximum matched-boundary displacement required for certification in
        arcseconds. It must be at least ``geometry_tolerance``.
    max_boundary_refinements : int, optional
        Maximum number of factor-of-two resolution refinements used to certify
        boundary displacement and topology.
    max_attempts : int, optional
        Maximum bounding-box rejection draws per lens realization.
    seed : int, optional
        Seed for the node-owned fallback random generator.
    node_label : str, optional
        Human-readable graph node identifier.

    Notes
    -----
    ``strong_lensing_area`` is a geometric source-plane cross-section in square
    arcseconds. It does not include magnification bias, detectability, cadence,
    image resolution, or cross-section weighting of the upstream lens sample.
    Adaptive FOV expansion keeps each realized pixelscale fixed, while boundary
    refinement starts from that realized value. Doubling FOV at a fixed
    pixelscale approximately quadruples the Jacobian-grid point count.

    References
    ----------
    * Caustics - https://github.com/Ciela-Institute/caustics
    * Shapely - https://shapely.readthedocs.io/en/stable/
    """

    _OUTPUTS = [
        "source_x",
        "source_y",
        "strong_lensing_area",
        "sampling_attempts",
        "expected_num_images",
        "critical_curve_fov",
        "boundary_uncertainty",
        "source_boundary_clearance",
        "boundary_refinements",
    ]

    def __init__(
        self,
        lens_model,
        *,
        cosmology,
        lens_redshift,
        source_redshift,
        lens_parameters,
        fov=None,
        pixelscale=0.01,
        pixelscale_fraction=None,
        max_fov_expansions=5,
        fov_expansion_factor=1.25,
        pseudo_caustic_points=2_048,
        pseudo_caustic_epsilon=1.0e-5,
        geometry_tolerance=1.0e-6,
        boundary_tolerance=1.0e-4,
        max_boundary_refinements=10,
        max_attempts=1_000,
        seed=None,
        node_label=None,
    ):
        pixelscale_fraction = _validate_optional_positive_fraction(
            "pixelscale_fraction",
            pixelscale_fraction,
        )
        _validate_source_position_configuration(
            lens_model=lens_model,
            lens_parameters=lens_parameters,
            fov=fov,
            pixelscale=pixelscale,
            pixelscale_fraction=pixelscale_fraction,
            max_fov_expansions=max_fov_expansions,
            pseudo_caustic_points=pseudo_caustic_points,
            pseudo_caustic_epsilon=pseudo_caustic_epsilon,
            geometry_tolerance=geometry_tolerance,
            boundary_tolerance=boundary_tolerance,
            max_boundary_refinements=max_boundary_refinements,
            max_attempts=max_attempts,
        )

        self.lens_model = lens_model
        self.cosmology = cosmology
        self.fov = None if fov is None else float(fov)
        self.pixelscale = float(pixelscale)
        self.pixelscale_fraction = pixelscale_fraction
        self.max_fov_expansions = int(max_fov_expansions)
        self.fov_expansion_factor = float(fov_expansion_factor)
        self.pseudo_caustic_points = int(pseudo_caustic_points)
        self.pseudo_caustic_epsilon = float(pseudo_caustic_epsilon)
        self.geometry_tolerance = float(geometry_tolerance)
        self.boundary_tolerance = float(boundary_tolerance)
        self.max_boundary_refinements = int(max_boundary_refinements)
        self.max_attempts = int(max_attempts)
        self._lens_parameter_names = tuple(lens_parameters)
        self._rng = np.random.default_rng(seed)

        node_inputs = {
            "lens_redshift": lens_redshift,
            "source_redshift": source_redshift,
        }
        for name, setter in lens_parameters.items():
            node_inputs[f"lens_{name}"] = setter

        super().__init__(
            self._non_func,
            node_label=node_label,
            outputs=self._OUTPUTS,
            **node_inputs,
        )

    def set_seed(self, seed):
        """Reset the node-owned fallback random generator.

        Parameters
        ----------
        seed : int or None
            Seed accepted by ``numpy.random.default_rng``.
        """
        self._rng = np.random.default_rng(seed)

    def _realized_pixelscale_for_one_lens(self, geometry_adapter, values):
        """Return this realization's initial Jacobian-grid spacing in arcseconds."""
        if self.pixelscale_fraction is None:
            return self.pixelscale
        characteristic_scale = geometry_adapter.characteristic_angular_scale(values)
        return min(
            self.pixelscale,
            self.pixelscale_fraction * characteristic_scale,
        )

    def _initial_fov_for_one_lens(self, geometry_adapter, values, *, pixelscale):
        """Return the explicit or adapter-derived starting FOV in arcseconds."""
        if self.fov is not None:
            initial_fov = self.fov
        else:
            initial_fov = geometry_adapter.initial_fov(values)
        if (self.fov is None or self.pixelscale_fraction is not None) and initial_fov <= pixelscale:
            raise ValueError(
                f"Initial fov {initial_fov} arcsec must be larger than "
                f"pixelscale={pixelscale} arcsec for lens model '{self.lens_model}'."
            )
        return initial_fov

    def _find_all_caustics_for_one_lens(
        self,
        lens,
        geometry_adapter,
        values,
        *,
        sample_index,
        pixelscale,
        initial_fov=None,
    ):
        """Extract complete caustics with bounded sample-local FOV expansion."""
        if initial_fov is None:
            initial_fov = self._initial_fov_for_one_lens(
                geometry_adapter,
                values,
                pixelscale=pixelscale,
            )
        current_fov = initial_fov
        singular_points = geometry_adapter.singular_points(values)

        for expansion_count in range(self.max_fov_expansions + 1):
            try:
                caustic_curves = _find_all_caustics(
                    lens,
                    center=_lens_plane_origin(values),
                    fov=current_fov,
                    pixelscale=pixelscale,
                    geometry_tolerance=self.geometry_tolerance,
                    singular_points=singular_points,
                )
                return tuple(caustic_curves), current_fov
            except _CausticFOVError as err:
                if expansion_count == self.max_fov_expansions:
                    raise RuntimeError(
                        "Critical-curve extraction exhausted adaptive FOV "
                        f"expansion for {self.lens_model} sample {sample_index} "
                        f"at node '{self.node_string}'; initial fov={initial_fov} "
                        f"arcsec, final fov={current_fov} arcsec, "
                        f"pixelscale={pixelscale} arcsec, "
                        f"max_fov_expansions={self.max_fov_expansions}. "
                        "Increase max_fov_expansions, provide a larger fov, "
                        "or reassess pixelscale."
                    ) from err
                current_fov *= self.fov_expansion_factor

    def _boundary_geometry_for_one_lens(
        self,
        lens,
        geometry_adapter,
        values,
        *,
        sample_index,
        pixelscale,
        pseudo_caustic_points,
        initial_fov=None,
    ):
        """Extract one immutable typed-boundary snapshot for a realized lens."""
        caustic_curves, critical_curve_fov = self._find_all_caustics_for_one_lens(
            lens,
            geometry_adapter,
            values,
            sample_index=sample_index,
            pixelscale=pixelscale,
            initial_fov=initial_fov,
        )
        pseudo_caustic_curves = _find_all_pseudo_caustics(
            lens,
            lens_model=self.lens_model,
            values=values,
            num_points=pseudo_caustic_points,
            epsilon=self.pseudo_caustic_epsilon,
            geometry_tolerance=self.geometry_tolerance,
        )
        return _BoundaryGeometry(
            caustic_curves=tuple(caustic_curves),
            pseudo_caustic_curves=tuple(pseudo_caustic_curves),
            critical_curve_fov=float(critical_curve_fov),
            pixelscale=float(pixelscale),
            pseudo_caustic_points=int(pseudo_caustic_points),
        )

    def _certified_boundary_geometry_for_one_lens(
        self,
        lens,
        geometry_adapter,
        values,
        *,
        sample_index,
        pixelscale,
    ):
        """Refine typed boundaries until displacement and topology converge."""
        previous = None
        last_previous = None
        last_uncertainty = np.inf
        last_topology_stable = False
        for refinement in range(self.max_boundary_refinements + 1):
            current = self._boundary_geometry_for_one_lens(
                lens,
                geometry_adapter,
                values,
                sample_index=sample_index,
                pixelscale=pixelscale / (2**refinement),
                pseudo_caustic_points=self.pseudo_caustic_points * (2**refinement),
                initial_fov=None if previous is None else previous.critical_curve_fov,
            )
            if previous is not None:
                last_previous = previous
                current, last_uncertainty, last_topology_stable = _compare_boundary_geometry(
                    previous,
                    current,
                    geometry_tolerance=self.geometry_tolerance,
                )
                if last_topology_stable and last_uncertainty <= self.boundary_tolerance:
                    return previous, current, last_uncertainty, refinement
            previous = current
        raise RuntimeError(
            "Boundary certification exhausted refinement for "
            f"{self.lens_model} sample {sample_index} at node '{self.node_string}'; "
            f"last displacement={last_uncertainty} arcsec, "
            f"topology_stable={last_topology_stable}, "
            f"boundary_tolerance={self.boundary_tolerance} arcsec, "
            f"max_boundary_refinements={self.max_boundary_refinements}, "
            f"previous_pixelscale={last_previous.pixelscale} arcsec, "
            f"previous_pseudo_caustic_points={last_previous.pseudo_caustic_points}, "
            f"previous_critical_curve_fov={last_previous.critical_curve_fov} arcsec, "
            f"current_pixelscale={current.pixelscale} arcsec, "
            f"current_pseudo_caustic_points={current.pseudo_caustic_points}, "
            f"current_critical_curve_fov={current.critical_curve_fov} arcsec."
        )

    def _region_for_one_lens(self, values, *, sample_index):
        """Certify typed boundaries and construct one strong-lensing region.

        Parameters
        ----------
        values : Mapping
            Numeric inputs for exactly one graph sample, including redshifts and
            every registered ``lens_<parameter>`` entry.
        sample_index : int
            Zero-based graph sample index included in adaptive-FOV exhaustion
            diagnostics.

        Returns
        -------
        tuple
            Geometry adapter, penultimate and final boundary snapshots,
            boundary uncertainty in arcseconds, refinement count, complete
            supported source-plane strong-lensing region, and realized initial
            Jacobian-grid spacing in arcseconds.
        """
        lens, _ = _construct_caustics_lens(
            lens_model=self.lens_model,
            cosmology=self.cosmology,
            values=values,
            lens_parameter_names=self._lens_parameter_names,
        )
        geometry_adapter = _get_lens_geometry_adapter(self.lens_model)
        realized_pixelscale = self._realized_pixelscale_for_one_lens(
            geometry_adapter,
            values,
        )
        previous_geometry, geometry, uncertainty, refinements = (
            self._certified_boundary_geometry_for_one_lens(
                lens,
                geometry_adapter,
                values,
                sample_index=sample_index,
                pixelscale=realized_pixelscale,
            )
        )
        region = _build_strong_lensing_region(
            geometry.caustic_curves,
            geometry.pseudo_caustic_curves,
            geometry_tolerance=self.geometry_tolerance,
        )
        return (
            geometry_adapter,
            previous_geometry,
            geometry,
            uncertainty,
            refinements,
            region,
            realized_pixelscale,
        )

    def compute(self, graph_state, rng_info=None, **kwargs):
        """Sample one uniform strong-lensing source position per graph sample.

        A fixed number of sub-seeds is drawn from ``rng_info`` (or the node-owned
        fallback generator) before any rejection sampling. Variable rejection
        counts for one lens therefore cannot perturb later lens samples.

        Parameters
        ----------
        graph_state : GraphState
            State containing the realized node inputs and receiving the nine
            computed outputs.
        rng_info : numpy.random.Generator, optional
            Caller-owned random generator. When omitted, the node-owned generator
            configured by ``seed`` or ``set_seed`` is used.
        **kwargs : dict, optional
            Explicit overrides for registered node inputs.

        Returns
        -------
        list
            ``source_x``, ``source_y``, ``strong_lensing_area``, and
            certification diagnostics as scalars for one sample or sample-first
            NumPy arrays for multiple samples.
        """
        input_values = self._build_inputs(graph_state, **kwargs)
        num_samples = graph_state.num_samples
        rng = self._rng if rng_info is None else rng_info
        sample_seeds = rng.integers(
            0,
            2**63,
            size=num_samples,
            dtype=np.uint64,
        )

        source_x = np.empty(num_samples, dtype=float)
        source_y = np.empty(num_samples, dtype=float)
        areas = np.empty(num_samples, dtype=float)
        attempts = np.empty(num_samples, dtype=int)
        expected_num_images = np.empty(num_samples, dtype=int)
        critical_curve_fov = np.empty(num_samples, dtype=float)
        boundary_uncertainty = np.empty(num_samples, dtype=float)
        source_boundary_clearance = np.empty(num_samples, dtype=float)
        boundary_refinements = np.empty(num_samples, dtype=int)
        for sample_index, sample_seed in enumerate(sample_seeds):
            values = {
                name: _sample_value(value, sample_index, num_samples) for name, value in input_values.items()
            }
            (
                geometry_adapter,
                previous_geometry,
                geometry,
                uncertainty,
                refinements,
                region,
                realized_pixelscale,
            ) = self._region_for_one_lens(
                values,
                sample_index=sample_index,
            )
            sample_rng = np.random.default_rng(sample_seed)
            geometry_settings = {
                "fov": self.fov,
                "configured_pixelscale": self.pixelscale,
                "pixelscale_fraction": self.pixelscale_fraction,
                "realized_pixelscale": realized_pixelscale,
                "max_fov_expansions": self.max_fov_expansions,
                "pseudo_caustic_points": self.pseudo_caustic_points,
                "pseudo_caustic_epsilon": self.pseudo_caustic_epsilon,
                "geometry_tolerance": self.geometry_tolerance,
                "boundary_tolerance": self.boundary_tolerance,
                "max_boundary_refinements": self.max_boundary_refinements,
            }
            (
                source_x[sample_index],
                source_y[sample_index],
                areas[sample_index],
                attempts[sample_index],
            ) = _sample_position(
                region,
                sample_rng,
                max_attempts=self.max_attempts,
                lens_identifier=(f"{self.lens_model} sample {sample_index} at node '{self.node_string}'"),
                geometry_settings=geometry_settings,
            )

            previous_count = geometry_adapter.expected_num_images(
                source_x[sample_index],
                source_y[sample_index],
                caustic_curves=previous_geometry.caustic_curves,
                pseudo_caustic_curves=previous_geometry.pseudo_caustic_curves,
                geometry_tolerance=self.geometry_tolerance,
            )
            final_count = geometry_adapter.expected_num_images(
                source_x[sample_index],
                source_y[sample_index],
                caustic_curves=geometry.caustic_curves,
                pseudo_caustic_curves=geometry.pseudo_caustic_curves,
                geometry_tolerance=self.geometry_tolerance,
            )
            clearance = _source_boundary_clearance(
                source_x[sample_index],
                source_y[sample_index],
                geometry,
                self.geometry_tolerance,
            )
            if previous_count != final_count or clearance <= uncertainty:
                raise RuntimeError(
                    "Sampled source-position certification failed for "
                    f"{self.lens_model} sample {sample_index} at node '{self.node_string}'; "
                    f"penultimate expected_num_images={previous_count}, "
                    f"final expected_num_images={final_count}, "
                    f"source_boundary_clearance={clearance} arcsec, "
                    f"boundary_uncertainty={uncertainty} arcsec, "
                    f"penultimate pixelscale={previous_geometry.pixelscale} arcsec, "
                    "penultimate pseudo_caustic_points="
                    f"{previous_geometry.pseudo_caustic_points}, "
                    f"final pixelscale={geometry.pixelscale} arcsec, "
                    f"final pseudo_caustic_points={geometry.pseudo_caustic_points}, "
                    f"boundary_refinements={refinements}."
                )

            expected_num_images[sample_index] = final_count
            critical_curve_fov[sample_index] = geometry.critical_curve_fov
            boundary_uncertainty[sample_index] = uncertainty
            source_boundary_clearance[sample_index] = clearance
            boundary_refinements[sample_index] = refinements

        if num_samples == 1:
            results = [
                source_x[0],
                source_y[0],
                areas[0],
                attempts[0],
                expected_num_images[0],
                critical_curve_fov[0],
                boundary_uncertainty[0],
                source_boundary_clearance[0],
                boundary_refinements[0],
            ]
        else:
            results = [
                source_x,
                source_y,
                areas,
                attempts,
                expected_num_images,
                critical_curve_fov,
                boundary_uncertainty,
                source_boundary_clearance,
                boundary_refinements,
            ]

        self._save_results(results, graph_state)
        return results


class CausticsLensImageNode(FunctionNode, CiteClass):
    """Compute point-source macro-images with the optional Caustics package.

    Notes
    -----
    ``pixelscale_fraction`` and ``epsilon_fraction`` optionally scale their
    corresponding numerical settings to each realized lens's characteristic
    angular scale. The configured absolute values remain upper bounds. Lens models
    without a registered geometry adapter retain the absolute settings.

    References
    ----------
    * Caustics - https://github.com/Ciela-Institute/caustics
    """

    _OUTPUTS = [
        "num_images",
        "image_x",
        "image_y",
        "macro_magnifications",
        "time_delays",
        "image_count_deficit",
        "solver_fov",
        "solver_pixelscale",
        "solver_attempts",
        "solver_fov_expansions",
        "solver_pixelscale_refinements",
    ]

    def __init__(
        self,
        lens_model,
        *,
        cosmology,
        lens_redshift,
        source_redshift,
        source_x,
        source_y,
        lens_parameters,
        max_images,
        min_images=2,
        expected_num_images=None,
        fov=5.0,
        fov_multiplier=1.0,
        pixelscale=0.05,
        pixelscale_fraction=None,
        epsilon=1.0e-3,
        epsilon_fraction=None,
        max_depth=25,
        max_fov_expansions=5,
        fov_expansion_factor=1.25,
        max_pixelscale_refinements=5,
        pixelscale_refinement_factor=0.5,
        node_label=None,
    ):
        pixelscale_fraction = _validate_optional_positive_fraction(
            "pixelscale_fraction",
            pixelscale_fraction,
        )
        epsilon_fraction = _validate_optional_positive_fraction(
            "epsilon_fraction",
            epsilon_fraction,
        )
        _validate_lens_configuration(lens_model, lens_parameters)
        integer_types = (int, np.integer)
        boolean_types = (bool, np.bool_)
        if (
            isinstance(max_images, boolean_types)
            or not isinstance(max_images, integer_types)
            or max_images < 2
        ):
            raise ValueError("max_images must be an integer greater than one.")
        if (
            isinstance(min_images, boolean_types)
            or not isinstance(min_images, integer_types)
            or not 1 <= min_images <= max_images
        ):
            raise ValueError("min_images must be between one and max_images.")
        if not np.isfinite(pixelscale) or pixelscale <= 0.0:
            raise ValueError("Invalid forward-raytrace solver configuration.")
        if not np.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError("Invalid forward-raytrace solver configuration.")
        if isinstance(max_depth, boolean_types) or not isinstance(max_depth, integer_types) or max_depth < 1:
            raise ValueError("Invalid forward-raytrace solver configuration.")
        if not np.isfinite(fov_multiplier) or fov_multiplier <= 0.0:
            raise ValueError("fov_multiplier must be positive and finite.")
        for name, limit in (
            ("max_fov_expansions", max_fov_expansions),
            ("max_pixelscale_refinements", max_pixelscale_refinements),
        ):
            if isinstance(limit, boolean_types) or not isinstance(limit, integer_types) or limit < 0:
                raise ValueError(f"{name} must be a non-negative integer.")

        self.lens_model = lens_model
        self.cosmology = cosmology
        self.max_images = int(max_images)
        self.min_images = int(min_images)
        self.fov_multiplier = float(fov_multiplier)
        self.pixelscale = float(pixelscale)
        self.pixelscale_fraction = pixelscale_fraction
        self.epsilon = float(epsilon)
        self.epsilon_fraction = epsilon_fraction
        self.max_depth = int(max_depth)
        self.max_fov_expansions = int(max_fov_expansions)
        self.fov_expansion_factor = float(fov_expansion_factor)
        self.max_pixelscale_refinements = int(max_pixelscale_refinements)
        self.pixelscale_refinement_factor = float(pixelscale_refinement_factor)
        self._lens_parameter_names = tuple(lens_parameters)

        # Register every lens parameter independently so AttributeIndicatorNode
        # dependencies inside the mapping remain visible to the graph.
        node_inputs = {
            "lens_redshift": lens_redshift,
            "source_redshift": source_redshift,
            "source_x": source_x,
            "source_y": source_y,
            "fov": fov,
            "expected_num_images": expected_num_images,
        }
        for name, setter in lens_parameters.items():
            node_inputs[f"lens_{name}"] = setter

        super().__init__(
            self._non_func,
            node_label=node_label,
            outputs=self._OUTPUTS,
            **node_inputs,
        )

    def _realized_angular_settings(self, geometry_adapter, values):
        """Return this realization's pixel-scale upper bound and epsilon."""
        realized_pixelscale = self.pixelscale
        realized_epsilon = self.epsilon
        if geometry_adapter is None or (self.pixelscale_fraction is None and self.epsilon_fraction is None):
            return realized_pixelscale, realized_epsilon

        characteristic_scale = geometry_adapter.characteristic_angular_scale(values)
        if self.pixelscale_fraction is not None:
            realized_pixelscale = min(
                realized_pixelscale,
                self.pixelscale_fraction * characteristic_scale,
            )
        if self.epsilon_fraction is not None:
            realized_epsilon = min(
                realized_epsilon,
                self.epsilon_fraction * characteristic_scale,
            )
        return realized_pixelscale, realized_epsilon

    def _forward_raytrace_images(
        self,
        lens,
        torch,
        beta_x,
        beta_y,
        *,
        center_x,
        center_y,
        current_fov,
        divisions,
        epsilon,
    ):
        """Return one independent global result from one Caustics invocation."""
        image_x, image_y = lens.forward_raytrace(
            beta_x,
            beta_y,
            epsilon=epsilon,
            x0=torch.as_tensor(center_x, dtype=torch.float64),
            y0=torch.as_tensor(center_y, dtype=torch.float64),
            fov=current_fov,
            divisions=divisions,
            max_depth=self.max_depth,
        )
        return _global_image_coordinates(image_x, image_y)

    def _solve_one(self, values):
        """Solve and validate the active macro-images for one lens system.

        Parameters
        ----------
        values : Mapping
            Numeric inputs for exactly one graph sample. Required entries are
            ``lens_redshift``, ``source_redshift``, source-plane ``source_x``
            and ``source_y`` in arcseconds, and ``lens_<parameter>`` for every
            configured Caustics constructor parameter.

        Returns
        -------
        image_x : numpy.ndarray, shape (I,)
            Active image-plane x positions in arcseconds.
        image_y : numpy.ndarray, shape (I,)
            Active image-plane y positions in arcseconds.
        macro_magnifications : numpy.ndarray, shape (I,)
            Finite, absolute dimensionless macro-magnifications.
        time_delays : numpy.ndarray, shape (I,)
            Observer-frame relative delays in days, normalized to start at zero.
        diagnostics : dict
            Scalar recovery diagnostics for the accepted global-search result.

        Notes
        -----
        All four arrays use the same deterministic ordering: increasing delay,
        then image x, then image y. Padding to ``max_images`` is performed by
        ``compute`` after this method returns.

        Raises
        ------
        ImportError
            If optional Caustics runtime dependencies are unavailable.
        TypeError
            If the selected lens lacks a required point-image method.
        ValueError
            If redshifts or the selected lens model are invalid.
        RuntimeError
            If root residuals, image counts, or active solver outputs fail
            validation.
        """
        try:
            realized_fov = float(values["fov"])
        except (TypeError, ValueError) as err:
            raise TypeError("fov must realize to a scalar numeric value.") from err
        if not np.isfinite(realized_fov) or realized_fov <= 0.0:
            raise ValueError("fov must realize to a positive finite value.")
        initial_fov = realized_fov * self.fov_multiplier
        geometry_adapter = _LENS_GEOMETRY_ADAPTERS.get(self.lens_model)
        realized_pixelscale, realized_epsilon = self._realized_angular_settings(
            geometry_adapter,
            values,
        )
        if initial_fov <= realized_pixelscale:
            raise ValueError(
                f"Initial solver fov={initial_fov} arcsec must be larger "
                f"than pixelscale={realized_pixelscale} arcsec."
            )

        expected_num_images = values["expected_num_images"]
        if expected_num_images is not None:
            if (
                isinstance(expected_num_images, (bool, np.bool_))
                or not isinstance(expected_num_images, (int, np.integer))
                or not self.min_images <= expected_num_images <= self.max_images
            ):
                raise ValueError(
                    "expected_num_images must be None or a non-Boolean integer "
                    "between min_images and max_images."
                )
            expected_num_images = int(expected_num_images)

        lens, torch = _construct_caustics_lens(
            lens_model=self.lens_model,
            cosmology=self.cosmology,
            values=values,
            lens_parameter_names=self._lens_parameter_names,
        )

        for method_name in (
            "forward_raytrace",
            "raytrace",
            "magnification",
            "time_delay",
        ):
            if not hasattr(lens, method_name):
                raise TypeError(
                    f"Caustics lens model '{self.lens_model}' does not implement "
                    f"required method '{method_name}'."
                )

        beta_x = torch.as_tensor(values["source_x"], dtype=torch.float64)
        beta_y = torch.as_tensor(values["source_y"], dtype=torch.float64)
        center_x, center_y = _lens_plane_origin(values)
        target_count = self.min_images if expected_num_images is None else expected_num_images
        retryable_errors = []
        solver_attempts = 0

        current_fov = initial_fov
        current_pixelscale = realized_pixelscale
        current_grid_pixelscale = None
        current_grid_variant = None
        fov_expansions = 0
        pixelscale_refinements = 0

        def recovery_context(recovered_count, recovery_stage):
            latest_retryable_error = retryable_errors[-1] if retryable_errors else None
            return (
                f"initial_fov={initial_fov}, current_fov={current_fov}, "
                f"configured_pixelscale={self.pixelscale}, "
                f"pixelscale_fraction={self.pixelscale_fraction}, "
                f"initial_pixelscale={realized_pixelscale}, "
                f"current_pixelscale={current_pixelscale}, "
                f"current_grid_pixelscale={current_grid_pixelscale}, "
                f"grid_variant={current_grid_variant}, "
                f"configured_epsilon={self.epsilon}, "
                f"epsilon_fraction={self.epsilon_fraction}, "
                f"realized_epsilon={realized_epsilon}, "
                f"recovery_stage={recovery_stage}, "
                f"solver_fov_expansions={fov_expansions}, "
                f"solver_pixelscale_refinements={pixelscale_refinements}, "
                f"solver_attempts={solver_attempts}, "
                f"expected_num_images={expected_num_images}, max_images={self.max_images}, "
                f"recovered_num_images={recovered_count}, "
                f"latest_retryable_error={latest_retryable_error}"
            )

        def result_is_complete(coordinates, *, recovery_stage):
            recovered_count = len(coordinates)
            if expected_num_images is not None and recovered_count > expected_num_images:
                raise RuntimeError(
                    f"Found {recovered_count} images, exceeding "
                    f"expected_num_images={expected_num_images}; "
                    f"{recovery_context(recovered_count, recovery_stage)}."
                )
            if recovered_count > self.max_images:
                raise RuntimeError(
                    f"Found {recovered_count} images, exceeding max_images={self.max_images}; "
                    f"{recovery_context(recovered_count, recovery_stage)}."
                )
            if expected_num_images is not None:
                return recovered_count == expected_num_images
            return recovered_count >= self.min_images

        def attempt(current_fov, current_pixelscale, *, recovery_stage):
            nonlocal solver_attempts, current_grid_pixelscale, current_grid_variant
            base_divisions = _pixelscale_to_divisions(current_fov, current_pixelscale)
            base_spacing = current_fov / base_divisions
            grid_variants = (
                ("base", center_x, center_y, base_divisions),
                ("divisions_plus_one", center_x, center_y, base_divisions + 1),
                (
                    "half_cell_shift",
                    center_x + 0.5 * base_spacing,
                    center_y + 0.5 * base_spacing,
                    base_divisions,
                ),
            )

            for (
                grid_variant,
                search_center_x,
                search_center_y,
                divisions,
            ) in grid_variants:
                current_grid_variant = grid_variant
                current_grid_pixelscale = current_fov / divisions
                solver_attempts += 1
                try:
                    coordinates = self._forward_raytrace_images(
                        lens,
                        torch,
                        beta_x,
                        beta_y,
                        center_x=search_center_x,
                        center_y=search_center_y,
                        current_fov=current_fov,
                        divisions=divisions,
                        epsilon=realized_epsilon,
                    )
                except Exception as error:
                    if _is_singular_forward_raytrace_error(error):
                        retryable_errors.append(f"{type(error).__name__}: {error}")
                        continue
                    if _is_retryable_forward_raytrace_error(error):
                        retryable_errors.append(f"{type(error).__name__}: {error}")
                        return None, False
                    raise
                break
            else:
                return None, False

            complete = result_is_complete(
                coordinates,
                recovery_stage=recovery_stage,
            )
            if complete or not len(coordinates) or geometry_adapter is None:
                return coordinates, complete

            singular_points = np.asarray(
                geometry_adapter.singular_points(values),
                dtype=float,
            ).reshape(-1, 2)
            empty_neighborhoods = _singular_neighborhoods_are_empty(
                coordinates,
                singular_points,
                current_grid_pixelscale,
            )
            if not np.any(empty_neighborhoods):
                return coordinates, False

            seeds = geometry_adapter.singular_image_seeds(
                lens,
                values,
                source_x=values["source_x"],
                source_y=values["source_y"],
                radius=min(realized_epsilon, current_grid_pixelscale),
            )
            seeds = np.asarray(seeds, dtype=float)
            seeds = seeds[empty_neighborhoods]
            unresolved_points = singular_points[empty_neighborhoods]

            solver_attempts += 1
            try:
                singular_coordinates = _refine_image_seeds(
                    lens,
                    torch,
                    seeds,
                    unresolved_points,
                    beta_x,
                    beta_y,
                    realized_epsilon,
                    current_grid_pixelscale,
                )
            except Exception as error:
                if not _is_retryable_forward_raytrace_error(error):
                    raise
                retryable_errors.append(f"{type(error).__name__}: {error}")
                return coordinates, False

            if len(singular_coordinates):
                coordinates = np.vstack((coordinates, singular_coordinates))
            return coordinates, result_is_complete(
                coordinates,
                recovery_stage="singular_seed",
            )

        coordinates, complete = attempt(
            current_fov,
            current_pixelscale,
            recovery_stage="initial_global",
        )

        for expansion in range(1, self.max_fov_expansions + 1):
            if complete:
                break
            fov_expansions = expansion
            current_fov *= self.fov_expansion_factor
            coordinates, complete = attempt(
                current_fov,
                current_pixelscale,
                recovery_stage="fov_expansion",
            )

        for refinement in range(1, self.max_pixelscale_refinements + 1):
            if complete:
                break
            pixelscale_refinements = refinement
            current_pixelscale *= self.pixelscale_refinement_factor
            coordinates, complete = attempt(
                current_fov,
                current_pixelscale,
                recovery_stage="pixelscale_refinement",
            )

        if coordinates is None:
            coordinates = np.empty((0, 2), dtype=float)

        num_images = len(coordinates)
        if not complete and num_images < self.min_images:
            raise RuntimeError(
                "Caustics image recovery exhausted; "
                f"target_count={target_count}; "
                f"{recovery_context(num_images, 'exhausted')}."
            )

        image_x_tensor = torch.as_tensor(coordinates[:, 0], dtype=torch.float64)
        image_y_tensor = torch.as_tensor(coordinates[:, 1], dtype=torch.float64)
        magnifications = torch.abs(lens.magnification(image_x_tensor, image_y_tensor))
        time_delays = lens.time_delay(image_x_tensor, image_y_tensor)

        image_x = coordinates[:, 0]
        image_y = coordinates[:, 1]
        magnifications = _to_numpy(magnifications)
        time_delays = _to_numpy(time_delays)

        if not all(
            np.all(np.isfinite(values_array))
            for values_array in (
                image_x,
                image_y,
                magnifications,
                time_delays,
            )
        ):
            raise RuntimeError("Caustics returned non-finite active image values.")

        time_delays = time_delays - np.min(time_delays)
        # np.lexsort uses the final key as the primary key: delay, then x, then y.
        order = np.lexsort((image_y, image_x, time_delays))
        return (
            image_x[order],
            image_y[order],
            magnifications[order],
            time_delays[order],
            {
                "image_count_deficit": (
                    -1 if expected_num_images is None else expected_num_images - num_images
                ),
                "solver_fov": current_fov,
                "solver_pixelscale": current_grid_pixelscale,
                "solver_attempts": solver_attempts,
                "solver_fov_expansions": fov_expansions,
                "solver_pixelscale_refinements": pixelscale_refinements,
            },
        )

    def compute(self, graph_state, rng_info=None, **kwargs):
        """Solve every sampled lens and save fixed-width numeric outputs."""
        del rng_info  # The solver is deterministic for realized input parameters.
        input_values = self._build_inputs(graph_state, **kwargs)
        num_samples = graph_state.num_samples

        counts = np.empty(num_samples, dtype=int)
        image_x = np.full((num_samples, self.max_images), np.nan)
        image_y = np.full((num_samples, self.max_images), np.nan)
        magnifications = np.zeros((num_samples, self.max_images), dtype=float)
        time_delays = np.full((num_samples, self.max_images), np.nan)
        image_count_deficit = np.empty(num_samples, dtype=int)
        solver_fov = np.empty(num_samples, dtype=float)
        solver_pixelscale = np.empty(num_samples, dtype=float)
        solver_attempts = np.empty(num_samples, dtype=int)
        solver_fov_expansions = np.empty(num_samples, dtype=int)
        solver_pixelscale_refinements = np.empty(num_samples, dtype=int)

        for sample_index in range(num_samples):
            current_values = {
                name: _sample_value(value, sample_index, num_samples) for name, value in input_values.items()
            }
            current_x, current_y, current_mu, current_delay, diagnostics = self._solve_one(current_values)
            count = len(current_x)
            counts[sample_index] = count
            image_x[sample_index, :count] = current_x
            image_y[sample_index, :count] = current_y
            magnifications[sample_index, :count] = current_mu
            time_delays[sample_index, :count] = current_delay
            image_count_deficit[sample_index] = diagnostics["image_count_deficit"]
            solver_fov[sample_index] = diagnostics["solver_fov"]
            solver_pixelscale[sample_index] = diagnostics["solver_pixelscale"]
            solver_attempts[sample_index] = diagnostics["solver_attempts"]
            solver_fov_expansions[sample_index] = diagnostics["solver_fov_expansions"]
            solver_pixelscale_refinements[sample_index] = diagnostics["solver_pixelscale_refinements"]

        if num_samples == 1:
            results = [
                counts[0],
                image_x[0],
                image_y[0],
                magnifications[0],
                time_delays[0],
                image_count_deficit[0],
                solver_fov[0],
                solver_pixelscale[0],
                solver_attempts[0],
                solver_fov_expansions[0],
                solver_pixelscale_refinements[0],
            ]
        else:
            results = [
                counts,
                image_x,
                image_y,
                magnifications,
                time_delays,
                image_count_deficit,
                solver_fov,
                solver_pixelscale,
                solver_attempts,
                solver_fov_expansions,
                solver_pixelscale_refinements,
            ]

        self._save_results(results, graph_state)
        return results

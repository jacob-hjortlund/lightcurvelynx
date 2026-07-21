"""Caustics-backed lens specifications and strong-lensing graph nodes.

``CausticsLensSpec`` mirrors one explicitly registered Caustics constructor.
Its parameters become graph inputs, except that a ``SinglePlane`` spec's
``lenses`` sequence contains nested specifications. The root spec owns its
cosmology and lens redshift (``z_l``); the consuming node supplies the source
redshift as ``z_s`` when each sampled lens is realized.
"""

import inspect
from collections.abc import Mapping
from dataclasses import dataclass
from inspect import Parameter
from math import hypot
from types import MappingProxyType

import numpy as np
from citation_compass import CiteClass
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from lightcurvelynx.base_models import FunctionNode

__all__ = [
    "CausticsLensImageNode",
    "CausticsLensSpec",
    "CausticsSourcePositionNode",
]

_INHERITED_LENS_PARAMETERS = {"cosmology", "z_l"}


def _is_required_parameter(parameter):
    """Return whether an inspected constructor parameter has no default."""
    return parameter.default is Parameter.empty and parameter.kind not in (
        Parameter.VAR_POSITIONAL,
        Parameter.VAR_KEYWORD,
    )


def _get_model_class(model):
    """Return one explicitly registered top-level Caustics lens class."""
    if model not in _LENS_MODEL_REGISTRY:
        raise ValueError(f"Requested model {model!r} is not supported currently.")
    caustics = _import_caustics()
    return getattr(caustics, model)


@dataclass(frozen=True, init=False)
class CausticsLensSpec:
    """Describe one registered Caustics lens constructor.

    Parameters
    ----------
    model : str
        Explicit registry key, including ``SinglePlane`` for a composition.
    parameters : Mapping[str, object]
        Constructor arguments. Values remain graph dependencies. For
        ``SinglePlane``, ``lenses`` is a sequence of nested lens specs rather
        than a graph input.

    Attributes
    ----------
    model : str
        Registered constructor name.
    parameters : Mapping[str, object]
        Read-only snapshot of supplied constructor arguments.

    Notes
    -----
    Validation is deliberately structural: the model must be registered,
    required constructor arguments must be available here or inherited from a
    parent plane, and explicit parameter names must occur in the constructor
    signature. Realized parameter values are passed to Caustics without local
    physical-domain validation.
    """

    model: str
    parameters: Mapping[str, object]

    def __init__(self, model, parameters):
        if not isinstance(parameters, Mapping):
            raise TypeError("parameters must be a mapping.")
        parameter_snapshot = dict(parameters)
        if any(not isinstance(name, str) for name in parameter_snapshot):
            raise TypeError("parameters keys must be strings.")
        if "z_s" in parameter_snapshot:
            raise ValueError("z_s is supplied by the consuming node as source_redshift.")

        model_class = _get_model_class(model)
        signature = inspect.signature(model_class)
        supported = {
            name
            for name, parameter in signature.parameters.items()
            if parameter.kind not in (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD)
        }
        if model == "ExternalShear":
            parametrization = parameter_snapshot.get("parametrization", "cartesian")
            if isinstance(parametrization, str) and parametrization == "angular":
                supported.difference_update(("gamma_1", "gamma_2"))
                supported.update(("gamma", "phi"))
            elif not isinstance(parametrization, str):
                supported.update(("gamma", "phi"))
        for name, parameter in signature.parameters.items():
            if (
                _is_required_parameter(parameter)
                and name not in parameter_snapshot
                and name not in _INHERITED_LENS_PARAMETERS
            ):
                raise ValueError(f"{name} is required by the chosen {model} lens model.")
        unsupported = set(parameter_snapshot).difference(supported)
        if unsupported:
            names = "\n".join(sorted(unsupported))
            raise ValueError(
                f"parameters contains keys not supported by the chosen {model} lens model.\n"
                f"Offending keys are:\n{names}"
            )

        if model == "SinglePlane":
            lenses = tuple(parameter_snapshot["lenses"])
            if any(not isinstance(lens, CausticsLensSpec) for lens in lenses):
                raise TypeError("SinglePlane lenses must contain only CausticsLensSpec objects.")
            parameter_snapshot["lenses"] = lenses

        object.__setattr__(self, "model", model)
        object.__setattr__(self, "parameters", MappingProxyType(parameter_snapshot))


_RECOVERY_SEED_POINTS = 256
_RECOVERY_ROOT_REFINEMENTS = 8
_INITIAL_FOV_PADDING = 1.1


def _import_caustics():
    """Lazily import the optional Caustics package."""
    try:
        import caustics
    except ImportError as err:  # pragma: no cover
        raise ImportError(
            "Caustics-backed lens nodes require the optional 'caustics' package. "
            "Install it with `pip install caustics`."
        ) from err
    return caustics


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
    caustics = _import_caustics()
    try:
        import torch

        torch.set_default_dtype(torch.float64)
    except ImportError as err:  # pragma: no cover
        raise ImportError(
            "Caustics-backed lens nodes require the optional 'caustics' package. "
            "Install it with `pip install caustics`."
        ) from err
    return caustics, torch


def _to_numpy(tensor):
    """Convert a Caustics backend tensor to a detached CPU NumPy array.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor returned by Caustics. It may reside on any Torch device or be
        attached to an autograd graph.

    Returns
    -------
    numpy.ndarray
        Detached CPU array with the same shape and dtype as the tensor's NumPy
        representation. No additional dtype conversion or copy is requested.
    """
    return tensor.detach().cpu().numpy()


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


def _validate_optional_positive_fraction(name, value):
    """Normalize an optional positive dimensionless fraction.

    Parameters
    ----------
    name : str
        Public argument name used in contextual error messages.
    value : object or None
        Ordinary float-convertible scalar, including a zero-dimensional NumPy
        array, or ``None`` to disable relative scaling.

    Returns
    -------
    float or None
        Finite positive dimensionless fraction, or ``None`` unchanged.

    Raises
    ------
    TypeError
        If ``value`` is a non-scalar NumPy array or cannot be converted to a
        scalar float.
    ValueError
        If the normalized fraction is non-finite or not strictly positive.
    """
    if value is None:
        return None
    if isinstance(value, np.ndarray) and value.ndim != 0:
        raise TypeError(f"{name} must be None or a scalar value convertible to float.")
    try:
        normalized = float(value)
    except (TypeError, ValueError, OverflowError) as err:
        raise TypeError(f"{name} must be None or a scalar value convertible to float.") from err
    if not np.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return normalized


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
    the returned mask has length zero.
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
    trusted Caustics and registered-adapter postconditions.
    """
    candidate_coordinates = np.column_stack((_to_numpy(image_x), _to_numpy(image_y)))

    mapped_x, mapped_y = lens.raytrace(
        torch.as_tensor(candidate_coordinates[:, 0], dtype=torch.float64),
        torch.as_tensor(candidate_coordinates[:, 1], dtype=torch.float64),
    )
    mapped_x = _to_numpy(mapped_x)
    mapped_y = _to_numpy(mapped_y)

    source_x = float(_to_numpy(beta_x))
    source_y = float(_to_numpy(beta_y))
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
    certification.
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
        or ``singular U``.

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
        Source-plane x/y angular offsets in arcseconds on the CPU.

    Raises
    ------
    ImportError
        If the optional Caustics runtime dependencies are unavailable.

    Notes
    -----
    The production caller supplies a trusted ``(N, 2)`` curve. Caustics'
    paired output shapes, types, and finiteness are trusted; this helper only
    crosses the Torch-to-NumPy representation boundary.
    """
    _, torch = _import_caustics_dependencies()
    coordinates = np.asarray(coordinates, dtype=float)

    source_x, source_y = lens.raytrace(
        torch.as_tensor(coordinates[:, 0], dtype=torch.float64),
        torch.as_tensor(coordinates[:, 1], dtype=torch.float64),
    )
    return np.column_stack((_to_numpy(source_x), _to_numpy(source_y)))


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
    """
    angles = 2.0 * np.pi * np.arange(_RECOVERY_SEED_POINTS) / _RECOVERY_SEED_POINTS
    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    source_position = np.array([source_x, source_y], dtype=float)
    seeds = []
    for recovery_point in recovery_points:
        circle = np.asarray(recovery_point, dtype=float) + radius * directions
        mapped_circle = _raytrace_curve(lens, circle)
        nearest = np.argmin(np.linalg.norm(mapped_circle - source_position, axis=1))
        seeds.append(circle[nearest])
    return np.asarray(seeds, dtype=float).reshape(-1, 2)


_PSEUDO_CAUSTIC_SEPARATION_FRACTION = 0.25


@dataclass(frozen=True)
class _PseudoCausticGenerator:
    """Describe one pseudo-caustic loop generator.

    The center is an image-plane (x, y) position in arcseconds.
    max_initial_radius is an optional image-plane radius cap in arcseconds for
    the first loop traced around that center.
    """

    center: tuple[float, float]
    max_initial_radius: float | None = None


class _GeometryAdapter:
    """Expose independent geometry capabilities for one realized lens system.

    search_center gives the image-plane center in arcseconds. initial_fov gives
    a full image-plane search extent in arcseconds, whereas resolution_scale
    gives a characteristic angular resolution in arcseconds. Jacobian mask
    points identify image-plane locations in arcseconds omitted from contour
    calculations. Root recovery points identify image-plane locations in
    arcseconds that receive targeted image searches. Pseudo-caustic generators
    identify centers and optional initial-radius caps in image-plane arcseconds
    for loops mapped into source-plane boundaries.
    """

    def search_center(self, values):
        """Return the image-plane search center in arcseconds."""
        raise NotImplementedError

    def initial_fov(self, values):
        """Return the full initial image-plane search extent in arcseconds."""
        raise NotImplementedError

    def resolution_scale(self, values):
        """Return the characteristic image-plane resolution in arcseconds."""
        raise NotImplementedError

    def jacobian_mask_points(self, values):
        """Return image-plane points in arcseconds masked from the Jacobian."""
        raise NotImplementedError

    def root_recovery_points(self, values):
        """Return image-plane points in arcseconds used for targeted recovery."""
        raise NotImplementedError

    def pseudo_caustic_generators(self, values):
        """Return image-plane loop generators for pseudo-caustic boundaries."""
        raise NotImplementedError

    def axisymmetry_center(self, values):
        """Return the exact image-plane symmetry center or None."""
        raise NotImplementedError

    def preserves_axisymmetry(self, values):
        """Return whether an affine adapter preserves rotational symmetry."""
        raise NotImplementedError

    @staticmethod
    def winding_number(curve, x, y):
        """Return the signed winding number around a source-plane point."""
        point = np.array([x, y], dtype=float)
        vectors = np.asarray(curve, dtype=float) - point
        following = np.roll(vectors, -1, axis=0)
        angles = np.arctan2(
            vectors[:, 0] * following[:, 1] - vectors[:, 1] * following[:, 0],
            np.sum(vectors * following, axis=1),
        )
        return int(round(float(np.sum(angles)) / (2.0 * np.pi)))

    def reference_num_images(self, values):
        """Return the regular-image count outside all typed boundaries."""
        raise NotImplementedError

    def expected_num_images(
        self,
        source_x,
        source_y,
        *,
        values,
        caustic_curves,
        pseudo_caustic_curves,
    ):
        """Return the signed-boundary regular-image count."""
        count = self.reference_num_images(values)
        count += 2 * sum(self.winding_number(curve, source_x, source_y) for curve in caustic_curves)
        count += sum(self.winding_number(curve, source_x, source_y) for curve in pseudo_caustic_curves)
        return count


@dataclass(frozen=True)
class _PointSingularityGeometryAdapter(_GeometryAdapter):
    """Realized finite geometry with independently enabled point capabilities."""

    center: tuple[float, float]
    resolution: float
    extent: float
    mask_center: bool
    recover_center: bool
    generate_pseudo_caustic: bool
    axisymmetric: bool = False

    def search_center(self, values):
        """Return the stored image-plane center in arcseconds."""
        return self.center

    def initial_fov(self, values):
        """Return the stored full image-plane extent in arcseconds."""
        return self.extent

    def resolution_scale(self, values):
        """Return the stored image-plane resolution scale in arcseconds."""
        return self.resolution

    def jacobian_mask_points(self, values):
        """Return the center when it must be masked from the Jacobian."""
        return (self.center,) if self.mask_center else ()

    def root_recovery_points(self, values):
        """Return the center when it requires targeted root recovery."""
        return (self.center,) if self.recover_center else ()

    def pseudo_caustic_generators(self, values):
        """Return the center when it generates a pseudo-caustic boundary."""
        if self.generate_pseudo_caustic:
            return (_PseudoCausticGenerator(self.center),)
        return ()

    def axisymmetry_center(self, values):
        """Return the stored center only for an exactly axisymmetric lens."""
        return self.center if self.axisymmetric else None

    def preserves_axisymmetry(self, values):
        """Return false because this is a non-affine adapter."""
        return False

    def reference_num_images(self, values):
        """Return the one-image reference count for an atomic lens."""
        return 1


@dataclass(frozen=True)
class _SmoothCuspGeometryAdapter(_GeometryAdapter):
    """Realized finite geometry for a cusp without a pseudo-caustic."""

    center: tuple[float, float]
    resolution: float
    extent: float
    mask_center: bool
    recover_center: bool
    axisymmetric: bool = False

    def search_center(self, values):
        """Return the stored image-plane center in arcseconds."""
        return self.center

    def initial_fov(self, values):
        """Return the stored full image-plane extent in arcseconds."""
        return self.extent

    def resolution_scale(self, values):
        """Return the stored image-plane resolution scale in arcseconds."""
        return self.resolution

    def jacobian_mask_points(self, values):
        """Return the center when it must be masked from the Jacobian."""
        return (self.center,) if self.mask_center else ()

    def root_recovery_points(self, values):
        """Return the center when it requires targeted root recovery."""
        return (self.center,) if self.recover_center else ()

    def pseudo_caustic_generators(self, values):
        """Return no pseudo-caustic generators for a smooth cusp."""
        return ()

    def axisymmetry_center(self, values):
        """Return the stored center only for an exactly axisymmetric lens."""
        return self.center if self.axisymmetric else None

    def preserves_axisymmetry(self, values):
        """Return false because this is a non-affine adapter."""
        return False

    def reference_num_images(self, values):
        """Return the one-image reference count for an atomic lens."""
        return 1


@dataclass(frozen=True)
class _AffinePerturbationGeometryAdapter(_GeometryAdapter):
    """Geometry capabilities for an affine lens-plane perturbation."""

    axisymmetry_preserving: bool = False

    def search_center(self, values):
        """Return the realized affine center in image-plane arcseconds."""
        return float(values["x0"]), float(values["y0"])

    def initial_fov(self, values):
        """Return no independent search extent for an affine perturbation."""
        return None

    def resolution_scale(self, values):
        """Return no independent resolution for an affine perturbation."""
        return None

    def jacobian_mask_points(self, values):
        """Return no Jacobian mask points for an affine perturbation."""
        return ()

    def root_recovery_points(self, values):
        """Return no targeted root-recovery points for an affine perturbation."""
        return ()

    def pseudo_caustic_generators(self, values):
        """Return no pseudo-caustic generators for an affine perturbation."""
        return ()

    def axisymmetry_center(self, values):
        """Return no independent symmetry center."""
        return None

    def preserves_axisymmetry(self, values):
        """Return whether this perturbation preserves rotational symmetry."""
        return self.axisymmetry_preserving

    def reference_num_images(self, values):
        """Return zero excess over the single-plane reference image."""
        return 1


@dataclass(frozen=True)
class _GeometryComponent:
    """Associate one named component with its realized geometry adapter."""

    name: str
    adapter: _GeometryAdapter
    affine: bool


@dataclass(frozen=True)
class _SinglePlaneGeometryAdapter(_GeometryAdapter):
    """Aggregate ordered atomic capabilities across one lens plane."""

    components: tuple[_GeometryComponent, ...]

    def axisymmetry_center(self, values):
        """Return the sole non-affine center when all children preserve symmetry."""
        non_affine = tuple(component for component in self.components if not component.affine)
        if len(non_affine) != 1:
            return None
        component = non_affine[0]
        center = component.adapter.axisymmetry_center(values[component.name])
        if center is None:
            return None
        if not all(
            component.adapter.preserves_axisymmetry(values[component.name])
            for component in self.components
            if component.affine
        ):
            return None
        return center

    def preserves_axisymmetry(self, values):
        """Return whether this plane is entirely symmetry-preserving affine."""
        return all(
            component.affine and component.adapter.preserves_axisymmetry(values[component.name])
            for component in self.components
        )

    def reference_num_images(self, values):
        """Return one image plus every component's reference-image excess."""
        return 1 + sum(
            component.adapter.reference_num_images(values[component.name]) - 1
            for component in self.components
        )

    def jacobian_mask_points(self, values):
        """Concatenate component Jacobian mask points in supplied order."""
        return tuple(
            point
            for component in self.components
            for point in component.adapter.jacobian_mask_points(values[component.name])
        )

    def root_recovery_points(self, values):
        """Concatenate component root-recovery points in supplied order."""
        return tuple(
            point
            for component in self.components
            for point in component.adapter.root_recovery_points(values[component.name])
        )

    def resolution_scale(self, values):
        """Return the smallest positive non-affine resolution in arcseconds."""
        return min(
            resolution
            for component in self.components
            if not component.affine
            if (resolution := component.adapter.resolution_scale(values[component.name])) > 0.0
        )

    def _search_envelope(self, values):
        bounds = []
        for component in self.components:
            if component.affine:
                continue
            component_values = values[component.name]
            center_x, center_y = component.adapter.search_center(component_values)
            extent = component.adapter.initial_fov(component_values)
            half_extent = 0.5 * extent
            bounds.append(
                (
                    center_x - half_extent,
                    center_x + half_extent,
                    center_y - half_extent,
                    center_y + half_extent,
                )
            )
        return (
            min(bound[0] for bound in bounds),
            max(bound[1] for bound in bounds),
            min(bound[2] for bound in bounds),
            max(bound[3] for bound in bounds),
        )

    def search_center(self, values):
        """Return the midpoint of the non-affine envelope in arcseconds."""
        min_x, max_x, min_y, max_y = self._search_envelope(values)
        return 0.5 * (min_x + max_x), 0.5 * (min_y + max_y)

    def initial_fov(self, values):
        """Return the larger full width of the non-affine envelope."""
        min_x, max_x, min_y, max_y = self._search_envelope(values)
        return max(max_x - min_x, max_y - min_y)

    def pseudo_caustic_generators(self, values):
        """Return ordered generators capped by their nearest distinct peer."""
        owner_centers = []
        generators = []
        for component in self.components:
            if component.affine:
                continue
            component_values = values[component.name]
            component_generators = component.adapter.pseudo_caustic_generators(component_values)
            if component_generators:
                owner_centers.extend(generator.center for generator in component_generators)
            else:
                owner_centers.append(component.adapter.search_center(component_values))
            generators.extend(component_generators)

        capped_generators = []
        for generator in generators:
            separations = tuple(
                distance
                for center in owner_centers
                if (
                    distance := hypot(
                        generator.center[0] - center[0],
                        generator.center[1] - center[1],
                    )
                )
                > 0.0
            )
            max_initial_radius = generator.max_initial_radius
            if separations:
                peer_cap = _PSEUDO_CAUSTIC_SEPARATION_FRACTION * min(separations)
                if max_initial_radius is None:
                    max_initial_radius = peer_cap
                else:
                    max_initial_radius = min(max_initial_radius, peer_cap)
            capped_generators.append(
                _PseudoCausticGenerator(
                    center=generator.center,
                    max_initial_radius=max_initial_radius,
                )
            )
        return tuple(capped_generators)


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
        if generator.max_initial_radius is None:
            radius = epsilon
        else:
            radius = min(epsilon, generator.max_initial_radius)
        previous_curve = _raytrace_curve(lens, center + radius * directions)
        last_change = np.inf

        for _ in range(32):
            radius *= 0.5
            current_curve = _raytrace_curve(lens, center + radius * directions)
            last_change = float(np.max(np.linalg.norm(current_curve - previous_curve, axis=1)))
            if last_change <= geometry_tolerance:
                pseudo_caustics.append(np.concatenate((current_curve, current_curve[:1]), axis=0))
                break
            previous_curve = current_curve
        else:
            raise RuntimeError(
                "Pseudo-caustic extraction did not converge after 32 refinements; "
                f"final boundary change was {last_change} arcsec."
            )

    return tuple(pseudo_caustics)


def _caustics_scalar(value):
    """Convert one scalar Caustics tensor to an immutable Python float."""
    return float(np.asarray(_to_numpy(value)).item())


def _sis_geometry_adapter(lens, values):
    center = (
        _caustics_scalar(lens.x0.value),
        _caustics_scalar(lens.y0.value),
    )
    einstein_radius = _caustics_scalar(lens.Rein.value)
    unsoftened = float(lens.s) == 0.0
    return _PointSingularityGeometryAdapter(
        center=center,
        resolution=einstein_radius,
        extent=2.0 * _INITIAL_FOV_PADDING * einstein_radius,
        mask_center=unsoftened,
        recover_center=unsoftened,
        generate_pseudo_caustic=unsoftened,
        axisymmetric=True,
    )


def _sie_geometry_adapter(lens, values):
    center = (
        _caustics_scalar(lens.x0.value),
        _caustics_scalar(lens.y0.value),
    )
    einstein_radius = _caustics_scalar(lens.Rein.value)
    axis_ratio = _caustics_scalar(lens.q.value)
    unsoftened = float(lens.s) == 0.0
    return _PointSingularityGeometryAdapter(
        center=center,
        resolution=einstein_radius,
        extent=(2.0 * _INITIAL_FOV_PADDING * einstein_radius / np.sqrt(axis_ratio)),
        mask_center=unsoftened,
        recover_center=unsoftened,
        generate_pseudo_caustic=unsoftened,
        axisymmetric=axis_ratio == 1.0,
    )


def _epl_geometry_adapter(lens, values):
    center = (
        _caustics_scalar(lens.x0.value),
        _caustics_scalar(lens.y0.value),
    )
    einstein_radius = _caustics_scalar(lens.Rein.value)
    axis_ratio = _caustics_scalar(lens.q.value)
    slope = _caustics_scalar(lens.t.value)
    return _PointSingularityGeometryAdapter(
        center=center,
        resolution=einstein_radius,
        extent=(2.0 * _INITIAL_FOV_PADDING * einstein_radius / np.sqrt(axis_ratio)),
        mask_center=slope <= 1.0,
        recover_center=slope <= 1.0,
        generate_pseudo_caustic=slope == 1.0,
        axisymmetric=axis_ratio == 1.0,
    )


def _nfw_geometry_adapter(lens, values):
    from caustics.constants import rad_to_arcsec

    center = (
        _caustics_scalar(lens.x0.value),
        _caustics_scalar(lens.y0.value),
    )
    scale_radius_mpc = _caustics_scalar(lens.get_scale_radius())
    distance_mpc = _caustics_scalar(lens.cosmology.angular_diameter_distance(lens.z_l.value))
    angular_scale = float(scale_radius_mpc / distance_mpc * rad_to_arcsec)
    unsoftened = float(lens.s) == 0.0
    return _SmoothCuspGeometryAdapter(
        center=center,
        resolution=angular_scale,
        extent=2.0 * _INITIAL_FOV_PADDING * angular_scale,
        mask_center=unsoftened,
        recover_center=unsoftened,
        axisymmetric=True,
    )


def _tnfw_geometry_adapter(lens, values):
    center = (
        _caustics_scalar(lens.x0.value),
        _caustics_scalar(lens.y0.value),
    )
    scale_radius = _caustics_scalar(lens.Rs.value)
    truncation = _caustics_scalar(lens.tau.value)
    unsoftened = float(lens.s) == 0.0
    return _PointSingularityGeometryAdapter(
        center=center,
        resolution=scale_radius,
        extent=(2.0 * _INITIAL_FOV_PADDING * truncation * scale_radius),
        mask_center=unsoftened,
        recover_center=unsoftened,
        generate_pseudo_caustic=not unsoftened,
        axisymmetric=True,
    )


def _pseudo_jaffe_geometry_adapter(lens, values):
    center = (
        _caustics_scalar(lens.x0.value),
        _caustics_scalar(lens.y0.value),
    )
    core_radius = _caustics_scalar(lens.Rc.value)
    scale_radius = _caustics_scalar(lens.Rs.value)
    return _SmoothCuspGeometryAdapter(
        center=center,
        resolution=core_radius,
        extent=2.0 * _INITIAL_FOV_PADDING * scale_radius,
        mask_center=True,
        recover_center=True,
        axisymmetric=True,
    )


def _external_shear_geometry_adapter(lens, values):
    gamma_1 = _caustics_scalar(lens.gamma_1.value)
    gamma_2 = _caustics_scalar(lens.gamma_2.value)
    return _AffinePerturbationGeometryAdapter(
        axisymmetry_preserving=gamma_1 == 0.0 and gamma_2 == 0.0,
    )


def _mass_sheet_geometry_adapter(lens, values):
    return _AffinePerturbationGeometryAdapter(axisymmetry_preserving=True)


@dataclass(frozen=True)
class _RegisteredLensModel:
    affine: bool
    geometry_factory: object


_LENS_MODEL_REGISTRY = {
    "SIS": _RegisteredLensModel(affine=False, geometry_factory=_sis_geometry_adapter),
    "SIE": _RegisteredLensModel(affine=False, geometry_factory=_sie_geometry_adapter),
    "EPL": _RegisteredLensModel(affine=False, geometry_factory=_epl_geometry_adapter),
    "NFW": _RegisteredLensModel(affine=False, geometry_factory=_nfw_geometry_adapter),
    "TNFW": _RegisteredLensModel(affine=False, geometry_factory=_tnfw_geometry_adapter),
    "PseudoJaffe": _RegisteredLensModel(
        affine=False,
        geometry_factory=_pseudo_jaffe_geometry_adapter,
    ),
    "ExternalShear": _RegisteredLensModel(
        affine=True,
        geometry_factory=_external_shear_geometry_adapter,
    ),
    "MassSheet": _RegisteredLensModel(
        affine=True,
        geometry_factory=_mass_sheet_geometry_adapter,
    ),
    "SinglePlane": _RegisteredLensModel(
        affine=False,
        geometry_factory=_SinglePlaneGeometryAdapter,
    ),
}


def _validate_root_lens_spec(lens):
    missing = _INHERITED_LENS_PARAMETERS.difference(lens.parameters)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"The root lens specification requires: {names}.")

    def validate_children(spec):
        for child in spec.parameters.get("lenses", ()):
            inherited = _INHERITED_LENS_PARAMETERS.intersection(child.parameters)
            if inherited:
                names = ", ".join(sorted(inherited))
                raise ValueError(f"Nested lens specifications inherit rather than define: {names}.")
            validate_children(child)

    validate_children(lens)
    if _lens_spec_is_affine(lens):
        raise ValueError("Caustics strong-lensing nodes require at least one non-affine lens model.")


def _lens_spec_is_affine(lens):
    """Return the effective affinity of one recursive lens specification."""
    if lens.model == "SinglePlane":
        is_affine = all(_lens_spec_is_affine(child) for child in lens.parameters["lenses"])
        if is_affine:
            error_str = "Composite lens model only contains affine lens components:"
            for child in lens.parameters["lenses"]:
                error_str += f"\n{child.model}"
            raise ValueError(error_str)
        else:
            return False
    return _LENS_MODEL_REGISTRY[lens.model].affine


def _lens_graph_inputs(lens, prefix="lens"):
    """Flatten one recursive specification into graph input names and setters."""
    graph_inputs = []
    for name, setter in lens.parameters.items():
        if name == "lenses":
            for index, child in enumerate(setter):
                graph_inputs.extend(_lens_graph_inputs(child, f"{prefix}_{index}"))
        else:
            graph_inputs.append((f"{prefix}_{name}", setter))
    return tuple(graph_inputs)


def _construct_lens_tree(
    lens_spec,
    values,
    *,
    caustics,
    torch,
    prefix,
    name,
    cosmology=None,
    z_l=None,
    z_s=None,
    root=False,
):
    """Recursively construct fresh Caustics objects from one graph sample."""
    lens_values = {
        parameter_name: values[f"{prefix}_{parameter_name}"]
        for parameter_name in lens_spec.parameters
        if parameter_name != "lenses"
    }
    if root:
        cosmology = lens_values.pop("cosmology")
        z_l = torch.as_tensor(lens_values.pop("z_l"), dtype=torch.float64)

    lens_name = lens_values.pop("name", name)
    lens_class = getattr(caustics, lens_spec.model)

    if lens_spec.model == "SinglePlane":

        children = tuple(
            _construct_lens_tree(
                child_spec,
                values,
                caustics=caustics,
                torch=torch,
                prefix=f"{prefix}_{index}",
                name=f"{name}_{index}",
                cosmology=cosmology,
            )
            for index, child_spec in enumerate(lens_spec.parameters.get("lenses", ()))
        )

        lens = lens_class(
            cosmology=cosmology,
            lenses=children,
            name=lens_name,
            z_l=z_l,
            z_s=z_s,
            **lens_values,
        )
    else:
        lens = lens_class(
            cosmology=cosmology,
            name=lens_name,
            z_l=z_l,
            z_s=z_s,
            **lens_values,
        )

    return lens


def _realize_lens_geometry(lens_spec, lens, values, *, torch, prefix, name, root=False):
    """Staticize one realized tree and recursively construct its geometry."""
    lens_values = {
        parameter_name: values[f"{prefix}_{parameter_name}"]
        for parameter_name in lens_spec.parameters
        if parameter_name != "lenses"
    }
    if root:
        lens_values.pop("cosmology")
        lens_values.pop("z_l")
    lens_values.pop("name", None)

    registry = _LENS_MODEL_REGISTRY[lens_spec.model]
    if lens_spec.model != "SinglePlane":
        for parameter_name in lens_values:
            parameter = getattr(lens, parameter_name, None)
            if hasattr(parameter, "value") and hasattr(parameter, "to"):
                parameter.to(dtype=torch.float64)
        lens.to_static()
        adapter_values = MappingProxyType(lens_values)
        adapter = registry.geometry_factory(lens, adapter_values)
        non_affine_components = ()
        if not registry.affine:
            non_affine_components = ((name, adapter.search_center(adapter_values)),)
        return adapter, adapter_values, registry.affine, non_affine_components

    components = []
    adapter_values = {}
    non_affine_components = []
    for index, (child_spec, child_lens) in enumerate(
        zip(lens_spec.parameters["lenses"], lens.lenses, strict=True)
    ):
        child_name = f"{name}_{index}"
        child_adapter, child_values, child_affine, child_non_affine = _realize_lens_geometry(
            child_spec,
            child_lens,
            values,
            torch=torch,
            prefix=f"{prefix}_{index}",
            name=child_name,
        )
        components.append(
            _GeometryComponent(
                name=child_name,
                adapter=child_adapter,
                affine=child_affine,
            )
        )
        adapter_values[child_name] = child_values
        non_affine_components.extend(child_non_affine)
    for first_index, (first_name, first_center) in enumerate(non_affine_components):
        for second_name, second_center in non_affine_components[first_index + 1 :]:
            if first_center == second_center:
                raise ValueError(
                    f"Non-affine lens components {first_name!r} and {second_name!r} "
                    "have exactly coincident centers."
                )
    adapter_values = MappingProxyType(adapter_values)
    return (
        registry.geometry_factory(tuple(components)),
        adapter_values,
        all(component.affine for component in components),
        tuple(non_affine_components),
    )


def _build_lens_system(lens_spec, *, values):
    """Realize one fresh registered lens tree for a graph sample."""
    caustics, torch = _import_caustics_dependencies()
    lens = _construct_lens_tree(
        lens_spec,
        values,
        caustics=caustics,
        torch=torch,
        prefix="lens",
        name="lens",
        z_s=torch.as_tensor(values["source_redshift"], dtype=torch.float64),
        root=True,
    )
    geometry_adapter, adapter_values, _, _ = _realize_lens_geometry(
        lens_spec,
        lens,
        values,
        torch=torch,
        prefix="lens",
        name="lens",
        root=True,
    )
    return lens, geometry_adapter, adapter_values


class _CausticFOVError(RuntimeError):
    """Signal that a larger image-plane FOV is required for completeness."""


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
        Closed source-plane caustic curves. Each element has shape ``(N, 2)``
        and contains x/y offsets in arcseconds. Disconnected critical curves
        remain separate list entries.

    Raises
    ------
    ImportError
        If Caustics, Torch, or ContourPy is unavailable.
    RuntimeError
        If singularity masking leaves no valid grid or boundary sample, or
        ContourPy returns a malformed, non-finite, or open curve, or a mapped
        caustic is not closed within ``geometry_tolerance``.
    _CausticFOVError
        If the outer boundary has not reached the positive-definite mapping
        region, no critical curve is found, or a curve reaches the image-plane
        boundary.

    Notes
    -----
    The interval count is rounded up to an even value, so the actual grid
    spacing is no larger than the requested ``pixelscale``. Caustics
    Jacobian types, shapes, and finiteness are trusted except at registered
    Jacobian mask points: non-finite determinant samples and a one-spacing
    neighborhood around each registered point are deliberately masked as part
    of critical-curve extraction. The post-mask grid and outer-boundary
    non-emptiness checks remain algorithmic completeness guards. ContourPy
    output is independently validated.
    """
    contourpy = _import_contourpy()
    _, torch = _import_caustics_dependencies()

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
    determinant = _to_numpy(torch.linalg.det(jacobian))

    x_coordinates = _to_numpy(x_axis)
    y_coordinates = _to_numpy(y_axis)
    invalid = ~np.isfinite(determinant)
    for mask_x, mask_y in jacobian_mask_points:
        squared_distance = (x_coordinates[np.newaxis, :] - mask_x) ** 2 + (
            y_coordinates[:, np.newaxis] - mask_y
        ) ** 2
        invalid |= squared_distance <= actual_pixelscale**2

    if np.all(invalid):
        raise RuntimeError("The lens-equation Jacobian grid contains no finite values.")

    symmetric_jacobian = 0.5 * (jacobian + jacobian.transpose(-1, -2))
    eigenvalues = _to_numpy(torch.linalg.eigvalsh(symmetric_jacobian))

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

        caustic_curve = _raytrace_curve(lens, critical_curve)
        caustic_gap = float(np.linalg.norm(caustic_curve[0] - caustic_curve[-1]))
        if caustic_gap > geometry_tolerance:
            raise RuntimeError(f"A mapped caustic is open with endpoint gap {caustic_gap} arcsec.")
        caustic_curves.append(caustic_curve)

    return caustic_curves


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
    immutable. NumPy arrays contained by the curve tuples remain mutable.
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
    previous_curves: tuple[np.ndarray, ...]
    current_curves: tuple[np.ndarray, ...]
    previous_points: tuple[np.ndarray, ...]
    current_points: tuple[np.ndarray, ...]


def _curve_center_and_diameter(curve):
    points = np.asarray(curve, dtype=float)
    if np.array_equal(points[0], points[-1]):
        points = points[:-1]
    lower = np.min(points, axis=0)
    upper = np.max(points, axis=0)
    return 0.5 * (lower + upper), float(np.linalg.norm(upper - lower))


def _raw_curve_hausdorff_distance(first_curve, second_curve):
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
    """Match raw mapped true-caustic curves without constructing geometry."""
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
    """Partition two matched contractions from regular true boundaries."""
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
        Floating-point coordinates with at least three unique vertices and an
        exactly repeated first/last vertex. Consecutive vertices within
        ``tolerance`` are removed, so ``M`` may be smaller than ``N``.

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
        Number of unique equally spaced points to return.

    Returns
    -------
    numpy.ndarray, shape (num_points, 2)
        Source-plane coordinates sampled without repeating the endpoint.
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
    """Return whether a matched curve retains its producer direction."""
    reference_points = _resample_closed_curve(reference)
    candidate_points = _resample_closed_curve(candidate)

    def minimum_cyclic_distance(points):
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
        Repaired finite positive-area region corresponding to each input curve,
        in input order.

    Raises
    ------
    ImportError
        If the optional Shapely dependency is unavailable.
    RuntimeError
        If curve normalization collapses a boundary or a repaired region is
        empty, non-finite, or has non-positive area.

    Notes
    -----
    Precision-grid snapping is not performed here; the tolerance controls only
    curve normalization. Final union snapping belongs to
    ``_build_strong_lensing_region``.
    """
    shapely = _import_shapely()
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
    """
    if len(reference_curves) != len(candidate_curves):
        return tuple(candidate_curves), np.inf, False, False
    if not reference_curves:
        return (), 0.0, True, True
    shapely = _import_shapely()
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
        touches)`` for one increasing index pair.

    Raises
    ------
    ImportError
        If the optional Shapely dependency is unavailable.
    RuntimeError
        If a boundary cannot be normalized into positive-area geometry.
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

    Notes
    -----
    Matching never crosses true/pseudo boundary types. Reordered arrays remain
    realization-local and are not cached on a node or across snapshots.
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
        Certified snapshot containing at least one true or pseudo boundary.
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
    """
    curves = (*geometry.caustic_curves, *geometry.pseudo_caustic_curves)
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
        components, concavities, and holes are retained.

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
    the final area check is retained. Per-boundary construction preserves
    concavities and holes and avoids accepting bounded faces collectively
    formed by several curves but inside none of their individual interiors.
    """
    shapely = _import_shapely()
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
    RuntimeError
        If no point is accepted within ``max_attempts``, with lens identity,
        geometry settings, bounding-box area, and polygon area in the message.

    Notes
    -----
    Production callers provide a validated finite positive-area polygonal
    region. Sampling uses uniform draws over its bounding box followed by
    strict interior rejection. The returned attempt count is one-based, and
    the supplied sample-local generator isolates variable rejection counts
    from other graph samples.
    """
    shapely = _import_shapely()
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
    boundary_tolerance,
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
    boundary_tolerance : object
        Float-convertible positive certification tolerance in arcseconds, at
        least ``geometry_tolerance``.
    max_boundary_refinements : int
        Positive maximum number of boundary-refinement comparisons.
    max_attempts : int
        Positive maximum number of rejection draws per realization.

    Returns
    -------
    normalized : dict[str, float]
        Finite normalized values for ``pixelscale``,
        ``pseudo_caustic_epsilon``, ``geometry_tolerance``,
        ``boundary_tolerance``, and ``fov_expansion_factor``, plus ``fov``
        when configured.

    Raises
    ------
    TypeError
        If a normalized scalar cannot be converted to float.
    ValueError
        If an integer setting is outside its ordinary type/range contract, a
        scalar is non-finite or outside its range, or the FOV, pixel-scale, and
        tolerance relations are inconsistent.
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
    """Reject realized geometries that source-region sampling cannot certify."""
    if lens_spec.model == "EPL":
        slope = _caustics_scalar(lens.t.value)
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


class CausticsSourcePositionNode(FunctionNode, CiteClass):
    """Uniformly sample the complete geometric strong-lensing source region.

    For each realized atomic or single-plane lens configuration, this node
    maps total-lens critical curves into true caustics and maps component-owned
    pseudo-caustics through the total lens. It structurally repairs the regular
    source-plane interiors, samples one position uniformly from their union,
    and certifies the regular-image count with signed boundary winding. The
    realized point, geometric cross-section, rejection-attempt count, image
    count, and boundary diagnostics are persisted in the node's ``GraphState``
    entries.

    Parameters
    ----------
    lens : CausticsLensSpec
        Registered atomic or recursive lens specification. The root parameters
        include ``cosmology`` and the lens-redshift setter ``z_l``.
    source_redshift : parameter
        Dimensionless source-redshift setter.
    fov : float or None, optional
        Initial image-plane critical-curve search width in arcseconds. When
        None, the realized total-lens geometry adapter derives a starting width
        from the component envelope.
    pixelscale : float, optional
        Maximum configured image-plane Jacobian-grid spacing in arcseconds.
        When ``pixelscale_fraction`` is enabled, this becomes an upper bound on
        the realized per-lens spacing.
    pixelscale_fraction : float or None, optional
        Maximum initial Jacobian-grid spacing as a fraction of the realized
        adapter-provided characteristic angular scale. For a composite, the
        adapter derives this scale from all non-affine components. When enabled,
        the smaller of this relative scale and ``pixelscale`` is used.
    max_fov_expansions : int, optional
        Maximum number of FOV expansions after the initial attempt.
    fov_expansion_factor : float, optional
        Finite multiplier greater than one applied at each FOV expansion.
        Larger values can increase two-dimensional grid cost rapidly.
    pseudo_caustic_points : int, optional
        Unique vertices used for each mapped pseudo-caustic boundary.
    pseudo_caustic_epsilon : float, optional
        Initial image-plane loop radius for pseudo-caustics in arcseconds.
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
    seed : object, optional
        Seed accepted by ``numpy.random.default_rng`` for the node-owned
        fallback generator.
    node_label : str, optional
        Human-readable graph node identifier.

    Attributes
    ----------
    source_x : AttributeIndicator
        Graph output for sampled source-plane x position in arcseconds.
    source_y : AttributeIndicator
        Graph output for sampled source-plane y position in arcseconds.
    strong_lensing_area : AttributeIndicator
        Graph output for geometric source-plane area in square arcseconds.
    sampling_attempts : AttributeIndicator
        Graph output for the one-based rejection-draw count.
    expected_num_images : AttributeIndicator
        Graph output for the certified regular-image count.
    critical_curve_fov : AttributeIndicator
        Graph output for the successful image-plane critical-curve FOV in
        arcseconds.
    boundary_uncertainty : AttributeIndicator
        Graph output for maximum matched regular-boundary displacement in
        arcseconds.
    source_boundary_clearance : AttributeIndicator
        Graph output for nearest regular typed-boundary distance in arcseconds.
    boundary_refinements : AttributeIndicator
        Graph output for the completed boundary-refinement count.

    Notes
    -----
    ``strong_lensing_area`` is a geometric source-plane cross-section in square
    arcseconds. It does not include magnification bias, detectability, cadence,
    image resolution, or cross-section weighting of the upstream lens sample.
    The configured ``pixelscale`` is an absolute upper bound. An optional
    fraction produces a realized initial upper bound per lens; each boundary
    refinement requests half the previous scale, while
    ``critical_curve_fov`` records the FOV that succeeded for that snapshot
    and ``boundary_uncertainty`` compares consecutive snapshots. Adaptive FOV
    recovery keeps the requested scale fixed and multiplies its FOV by
    ``fov_expansion_factor``.

    One shared lens system and its total geometry adapter are realized per graph
    sample. The adapter supplies composite search center, extent, resolution,
    Jacobian masks, pseudo-caustic generators, and signed image counting from
    the same realized values. True critical curves come from the total-lens
    Jacobian; every component-owned pseudo-caustic loop is also mapped through
    that total lens. For a composite, the adapter derives the search center,
    extent, and resolution from its realized children.

    Only an exact axisymmetry capability enables three-snapshot contraction
    certification of a true caustic as a source-plane point center. Certified
    points are kept separate from regular true and pseudo-caustic curves: they
    have no radius or uncertainty and do not participate in structural repair,
    signed winding, source-region area, boundary uncertainty, or clearance.
    Proposal sampling rejects only exact coordinate equality with a certified
    point. EPL slopes satisfy the common physical domain ``0 < t < 2``, but
    source-region certification currently requires ``t <= 1``; steeper EPL
    components remain available to ``CausticsLensImageNode``.

    Structural repair converts every final regular-boundary interior into valid
    polygonal geometry before union, and signed winding over those regular
    curves provides the certified image count. Results are saved in
    ``GraphState``; a caller RNG takes precedence over the seeded fallback, and
    sample-local sub-seeds isolate variable rejection counts. Persisted
    coordinates and diagnostics make downstream use deterministic from the
    sampled state.

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
        lens,
        *,
        source_redshift,
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
        """Configure geometric source-position sampling.

        Parameters
        ----------
        lens : CausticsLensSpec
            Registered atomic or recursive lens specification. The root owns
            ``cosmology`` and the dimensionless lens-redshift setter ``z_l``.
        source_redshift : object
            Graph setter for dimensionless source redshift.
        fov : float-convertible scalar or None, optional
            Configured initial image-plane critical-curve FOV in arcseconds, or
            ``None`` for adapter-derived per-lens FOV.
        pixelscale : float-convertible scalar, optional
            Positive configured Jacobian-grid spacing upper bound in
            arcseconds.
        pixelscale_fraction : float-convertible scalar or None, optional
            Positive dimensionless fraction of realized characteristic scale;
            a zero-dimensional NumPy array is accepted.
        max_fov_expansions : int, optional
            Non-negative bounded FOV expansion count. Source-node integer
            settings use the built-in ``int`` contract.
        fov_expansion_factor : float-convertible scalar, optional
            Finite FOV multiplier strictly greater than one.
        pseudo_caustic_points : int, optional
            Number of unique pseudo-caustic vertices, at least three.
        pseudo_caustic_epsilon : float-convertible scalar, optional
            Positive initial pseudo-caustic loop radius in arcseconds.
        geometry_tolerance : float-convertible scalar, optional
            Positive curve and topology tolerance in arcseconds, smaller than
            ``pixelscale``.
        boundary_tolerance : float-convertible scalar, optional
            Positive matched-boundary tolerance in arcseconds, at least
            ``geometry_tolerance``.
        max_boundary_refinements : int, optional
            Positive maximum number of boundary-refinement comparisons.
        max_attempts : int, optional
            Positive maximum rejection-draw count per realization.
        seed : object, optional
            Seed accepted by ``numpy.random.default_rng`` for the fallback
            generator.
        node_label : str or None, optional
            Human-readable graph node identifier.

        Returns
        -------
        None
            The configured function node registers its inputs and outputs.

        Raises
        ------
        TypeError
            If ``lens``, fractions, or normalized scalar settings have invalid
            types.
        ValueError
            If the lens specification or a numerical setting is outside its
            range, or static FOV/scale/tolerance relations fail.

        Notes
        -----
        ``source_redshift`` is registered directly. Lens parameters use
        ``lens_<field>`` at the root and positional path segments inside nested
        planes. All nine public outputs are registered in ``_OUTPUTS`` order.
        The seeded node-owned generator is used only when ``compute`` receives
        no caller generator.
        """
        if not isinstance(lens, CausticsLensSpec):
            raise TypeError("lens must be a CausticsLensSpec.")
        _validate_root_lens_spec(lens)
        lens_graph_inputs = _lens_graph_inputs(lens)

        pixelscale_fraction = _validate_optional_positive_fraction(
            "pixelscale_fraction",
            pixelscale_fraction,
        )
        normalized = _validate_source_position_configuration(
            fov=fov,
            pixelscale=pixelscale,
            pixelscale_fraction=pixelscale_fraction,
            max_fov_expansions=max_fov_expansions,
            fov_expansion_factor=fov_expansion_factor,
            pseudo_caustic_points=pseudo_caustic_points,
            pseudo_caustic_epsilon=pseudo_caustic_epsilon,
            geometry_tolerance=geometry_tolerance,
            boundary_tolerance=boundary_tolerance,
            max_boundary_refinements=max_boundary_refinements,
            max_attempts=max_attempts,
        )

        self.lens = lens
        self.fov = normalized.get("fov")
        self.pixelscale = normalized["pixelscale"]
        self.pixelscale_fraction = pixelscale_fraction
        self.max_fov_expansions = int(max_fov_expansions)
        self.fov_expansion_factor = normalized["fov_expansion_factor"]
        self.pseudo_caustic_points = int(pseudo_caustic_points)
        self.pseudo_caustic_epsilon = normalized["pseudo_caustic_epsilon"]
        self.geometry_tolerance = normalized["geometry_tolerance"]
        self.boundary_tolerance = normalized["boundary_tolerance"]
        self.max_boundary_refinements = int(max_boundary_refinements)
        self.max_attempts = int(max_attempts)
        self._rng = np.random.default_rng(seed)

        node_inputs = {"source_redshift": source_redshift}
        for name, setter in lens_graph_inputs:
            node_inputs[name] = setter

        super().__init__(
            self._non_func,
            node_label=node_label,
            outputs=self._OUTPUTS,
            **node_inputs,
        )

    def _lens_description(self):
        """Return a deterministic description of the root lens."""
        return f"lens model {self.lens.model!r}"

    def _lens_identifier(self, sample_index):
        """Return the lens specification, graph sample, and node identifier."""
        return f"{self._lens_description()} sample {sample_index} at node '{self.node_string}'"

    def _realized_pixelscale_for_one_lens(self, geometry_adapter, values):
        """Realize the initial requested grid-spacing upper bound.

        Parameters
        ----------
        geometry_adapter : _GeometryAdapter
            Realized total-lens adapter for this lens system.
        values : Mapping[str, object]
            Realized inputs for one lens system.

        Returns
        -------
        float
            Initial requested Jacobian-grid spacing upper bound in arcseconds.

        Notes
        -----
        Without a fraction this is the configured absolute scale. Otherwise it
        is the smaller of that value and
        ``pixelscale_fraction * resolution_scale(values)``; the
        adapter's positive finite scale postcondition is trusted.
        """
        if self.pixelscale_fraction is None:
            return self.pixelscale
        characteristic_scale = geometry_adapter.resolution_scale(values)
        return min(
            self.pixelscale,
            self.pixelscale_fraction * characteristic_scale,
        )

    def _initial_fov_for_one_lens(self, geometry_adapter, values, *, pixelscale):
        """Realize the starting critical-curve search FOV.

        Parameters
        ----------
        geometry_adapter : _GeometryAdapter
            Realized total-lens adapter for this lens system.
        values : Mapping[str, object]
            Realized inputs for one lens system.
        pixelscale : float
            Realized initial requested grid-spacing upper bound in arcseconds.

        Returns
        -------
        float
            Explicit configured or adapter-derived initial image-plane FOV in
            arcseconds.

        Raises
        ------
        ValueError
            If a dynamically adapter-derived or relative-scale configuration
            realizes to an FOV no larger than ``pixelscale``.
        """
        if self.fov is not None:
            initial_fov = self.fov
        else:
            initial_fov = geometry_adapter.initial_fov(values)
        if (self.fov is None or self.pixelscale_fraction is not None) and initial_fov <= pixelscale:
            raise ValueError(
                f"Initial fov {initial_fov} arcsec must be larger than "
                f"pixelscale={pixelscale} arcsec for {self._lens_description()}."
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
        """Extract complete caustics with bounded sample-local FOV recovery.

        Parameters
        ----------
        lens : object
            Realized Caustics lens.
        geometry_adapter : _GeometryAdapter
            Realized total-lens adapter for this lens system.
        values : Mapping[str, object]
            Realized inputs for one lens system.
        sample_index : int
            Zero-based graph sample index used in diagnostics.
        pixelscale : float
            Requested maximum Jacobian-grid spacing in arcseconds, held fixed
            throughout FOV recovery.
        initial_fov : float or None, optional
            Initial image-plane FOV in arcseconds, or ``None`` to realize it
            from configured/adapter policy.

        Returns
        -------
        caustic_curves : tuple of numpy.ndarray
            Separate closed source-plane caustic curves, each with shape
            ``(P, 2)`` in arcseconds.
        critical_curve_fov : float
            Image-plane FOV in arcseconds that produced complete curves.

        Raises
        ------
        RuntimeError
            If ``_CausticFOVError`` persists through the configured bounded
            expansion schedule.

        Notes
        -----
        Each retry multiplies the current FOV by
        ``fov_expansion_factor`` while retaining the same requested
        ``pixelscale``.
        """
        if initial_fov is None:
            initial_fov = self._initial_fov_for_one_lens(
                geometry_adapter,
                values,
                pixelscale=pixelscale,
            )
        current_fov = initial_fov
        search_center = geometry_adapter.search_center(values)
        jacobian_mask_points = geometry_adapter.jacobian_mask_points(values)

        for expansion_count in range(self.max_fov_expansions + 1):
            try:
                caustic_curves = _find_all_caustics(
                    lens,
                    center=search_center,
                    fov=current_fov,
                    pixelscale=pixelscale,
                    geometry_tolerance=self.geometry_tolerance,
                    jacobian_mask_points=jacobian_mask_points,
                )
                return tuple(caustic_curves), current_fov
            except _CausticFOVError as err:
                if expansion_count == self.max_fov_expansions:
                    raise RuntimeError(
                        "Critical-curve extraction exhausted adaptive FOV "
                        f"expansion for {self._lens_identifier(sample_index)}; "
                        f"initial fov={initial_fov} "
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
        """Extract one typed-boundary snapshot for a realized lens.

        Parameters
        ----------
        lens : object
            Realized Caustics lens.
        geometry_adapter : _GeometryAdapter
            Realized total-lens adapter for this lens system.
        values : Mapping[str, object]
            Realized inputs for one lens system.
        sample_index : int
            Zero-based graph sample index used in diagnostics.
        pixelscale : float
            Requested critical-curve grid-spacing upper bound in arcseconds.
        pseudo_caustic_points : int
            Number of unique image-plane loop vertices per pseudo-caustic.
        initial_fov : float or None, optional
            Initial image-plane search FOV in arcseconds, or ``None`` for
            configured/adapter policy.

        Returns
        -------
        _BoundaryGeometry
            Snapshot containing separate total-lens true and mapped pseudo
            boundaries, the successful FOV, requested scale upper bound, and
            pseudo-caustic resolution.

        Notes
        -----
        The stored ``pixelscale`` is the requested snapshot upper bound, not
        the actual even-grid spacing. The passed adapter is used directly and
        is not looked up again.
        """
        caustic_curves, critical_curve_fov = self._find_all_caustics_for_one_lens(
            lens,
            geometry_adapter,
            values,
            sample_index=sample_index,
            pixelscale=pixelscale,
            initial_fov=initial_fov,
        )
        pseudo_caustic_curves = _trace_pseudo_caustics(
            lens,
            geometry_adapter,
            values,
            num_points=pseudo_caustic_points,
            epsilon=self.pseudo_caustic_epsilon,
            geometry_tolerance=self.geometry_tolerance,
        )
        return _BoundaryGeometry(
            caustic_curves=tuple(caustic_curves),
            pseudo_caustic_curves=tuple(pseudo_caustic_curves),
            critical_curve_fov=critical_curve_fov,
            pixelscale=pixelscale,
            pseudo_caustic_points=pseudo_caustic_points,
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
        """Refine typed boundaries until displacement and topology converge.

        Parameters
        ----------
        lens : object
            Realized Caustics lens.
        geometry_adapter : _GeometryAdapter
            Realized total-lens adapter for this lens system.
        values : Mapping[str, object]
            Realized inputs for one lens system.
        sample_index : int
            Zero-based graph sample index used in diagnostics.
        pixelscale : float
            Realized initial requested grid-spacing upper bound in arcseconds.

        Returns
        -------
        previous_geometry : _BoundaryGeometry
            Penultimate certified comparison snapshot.
        geometry : _BoundaryGeometry
            Final converged snapshot, reordered to the penultimate boundary
            assignment.
        boundary_uncertainty : float
            Maximum matched true-or-pseudo boundary displacement in
            arcseconds.
        boundary_refinements : int
            Number of completed factor-of-two refinement steps.

        Raises
        ------
        RuntimeError
            If displacement and typed topology do not converge within
            ``max_boundary_refinements``.

        Notes
        -----
        Each refinement halves the requested grid scale and doubles the number
        of pseudo-caustic vertices. Non-axisymmetric systems compare two raw
        snapshots. Exactly axisymmetric systems first partition contracting
        point caustics across three raw snapshots, then apply the unchanged
        regular-boundary topology and displacement policy. Certified point
        centers must also retain their count and move no farther than
        ``boundary_tolerance``, but they do not enter boundary uncertainty.
        """
        axisymmetry_center = geometry_adapter.axisymmetry_center(values)
        older = None
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
            if axisymmetry_center is None:
                if previous is not None:
                    current, last_uncertainty, last_topology_stable = _compare_boundary_geometry(
                        previous,
                        current,
                        geometry_tolerance=self.geometry_tolerance,
                    )
                    if last_topology_stable and last_uncertainty <= self.boundary_tolerance:
                        return previous, current, last_uncertainty, refinement
                previous = current
                continue

            if older is not None and previous is not None:
                partition = _partition_axisymmetric_point_caustics(
                    older.caustic_curves,
                    previous.caustic_curves,
                    current.caustic_curves,
                    boundary_tolerance=self.boundary_tolerance,
                )
                if partition is None:
                    last_uncertainty = np.inf
                    last_topology_stable = False
                else:
                    comparison_previous = _BoundaryGeometry(
                        caustic_curves=partition.previous_curves,
                        pseudo_caustic_curves=previous.pseudo_caustic_curves,
                        critical_curve_fov=previous.critical_curve_fov,
                        pixelscale=previous.pixelscale,
                        pseudo_caustic_points=previous.pseudo_caustic_points,
                        point_caustics=partition.previous_points,
                    )
                    comparison_current = _BoundaryGeometry(
                        caustic_curves=partition.current_curves,
                        pseudo_caustic_curves=current.pseudo_caustic_curves,
                        critical_curve_fov=current.critical_curve_fov,
                        pixelscale=current.pixelscale,
                        pseudo_caustic_points=current.pseudo_caustic_points,
                        point_caustics=partition.current_points,
                    )
                    (
                        comparison_current,
                        last_uncertainty,
                        last_topology_stable,
                    ) = _compare_boundary_geometry(
                        comparison_previous,
                        comparison_current,
                        geometry_tolerance=self.geometry_tolerance,
                    )
                    point_centers_stable = len(comparison_previous.point_caustics) == len(
                        comparison_current.point_caustics
                    ) and all(
                        np.linalg.norm(previous_point - current_point) <= self.boundary_tolerance
                        for previous_point, current_point in zip(
                            comparison_previous.point_caustics,
                            comparison_current.point_caustics,
                            strict=True,
                        )
                    )
                    last_topology_stable = last_topology_stable and point_centers_stable
                    if last_topology_stable and last_uncertainty <= self.boundary_tolerance:
                        return (
                            comparison_previous,
                            comparison_current,
                            last_uncertainty,
                            refinement,
                        )
            older = previous
            previous = current
        raise RuntimeError(
            "Boundary certification exhausted refinement for "
            f"{self._lens_identifier(sample_index)}; "
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
            Inputs for exactly one graph sample, including source redshift and
            every flattened lens-specification parameter.
        sample_index : int
            Zero-based graph sample index included in adaptive-FOV exhaustion
            diagnostics.

        Returns
        -------
        geometry_adapter : _GeometryAdapter
            Realized aggregate geometry adapter.
        adapter_values : Mapping
            Recursive values consumed by the geometry adapter.
        previous_geometry : _BoundaryGeometry
            Penultimate boundary snapshot.
        geometry : _BoundaryGeometry
            Final certified boundary snapshot.
        boundary_uncertainty : float
            Maximum matched-boundary displacement in arcseconds.
        boundary_refinements : int
            Completed boundary-refinement count.
        region : shapely.Polygon or shapely.MultiPolygon
            Complete supported strong-lensing source region.
        realized_pixelscale : float
            Realized initial requested grid-spacing upper bound in arcseconds.

        Raises
        ------
        ImportError
            If an optional Caustics, ContourPy, or Shapely dependency is
            unavailable.
        ValueError
            If realized lens, adapter, or FOV/scale inputs violate their owned
            domains.
        RuntimeError
            If critical-curve or boundary certification exhausts, or topology
            construction collapses.

        Notes
        -----
        This method builds exactly one shared lens system for the entire
        realization and threads its total lens, adapter, and values through
        certification without caching sample-local objects on the node.
        """
        lens, geometry_adapter, adapter_values = _build_lens_system(
            self.lens,
            values=values,
        )
        _validate_source_geometry_support(self.lens, lens)
        realized_pixelscale = self._realized_pixelscale_for_one_lens(
            geometry_adapter,
            adapter_values,
        )
        previous_geometry, geometry, uncertainty, refinements = (
            self._certified_boundary_geometry_for_one_lens(
                lens,
                geometry_adapter,
                adapter_values,
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
            adapter_values,
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
            configured by ``seed`` is used.
        **kwargs : dict, optional
            Explicit overrides keyed by registered node input name.

        Returns
        -------
        results : list
            Nine values in this exact order:

            1. ``source_x``, source-plane arcseconds;
            2. ``source_y``, source-plane arcseconds;
            3. ``strong_lensing_area``, square arcseconds;
            4. ``sampling_attempts``, one-based rejection-draw count;
            5. ``expected_num_images``, regular-image count;
            6. ``critical_curve_fov``, successful image-plane FOV in
               arcseconds;
            7. ``boundary_uncertainty``, matched-boundary displacement in
               arcseconds;
            8. ``source_boundary_clearance``, nearest typed-boundary
               distance in arcseconds;
            9. ``boundary_refinements``, completed refinement count.

            Each value is a scalar when ``graph_state.num_samples == 1`` and
            a NumPy array with shape ``(S,)`` for ``S`` graph samples
            otherwise.

        Raises
        ------
        RuntimeError
            If bounded critical-curve extraction, boundary certification, or
            rejection sampling exhausts, or the sampled source fails
            penultimate/final count and clearance certification.

        Notes
        -----
        All nine results are persisted to this node's ``GraphState``
        entries. The caller generator takes precedence over the fallback
        generator. Exactly ``S`` unsigned 64-bit sub-seeds are drawn before
        any per-sample rejection, then one independent generator is created per
        sample so variable rejection counts cannot perturb later samples.
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
                adapter_values,
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
                lens_identifier=self._lens_identifier(sample_index),
                geometry_settings=geometry_settings,
                excluded_points=geometry.point_caustics,
            )

            previous_count = geometry_adapter.expected_num_images(
                source_x[sample_index],
                source_y[sample_index],
                values=adapter_values,
                caustic_curves=previous_geometry.caustic_curves,
                pseudo_caustic_curves=previous_geometry.pseudo_caustic_curves,
            )
            final_count = geometry_adapter.expected_num_images(
                source_x[sample_index],
                source_y[sample_index],
                values=adapter_values,
                caustic_curves=geometry.caustic_curves,
                pseudo_caustic_curves=geometry.pseudo_caustic_curves,
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
                    f"{self._lens_identifier(sample_index)}; "
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

    Parameters
    ----------
    lens : CausticsLensSpec
        Registered atomic or recursive lens specification. The root parameters
        include ``cosmology`` and the lens-redshift setter ``z_l``.
    source_redshift : object
        Graph setter for dimensionless source redshift.
    source_x : object
        Graph setter realizing to a finite source-plane x coordinate in
        arcseconds.
    source_y : object
        Graph setter realizing to a finite source-plane y coordinate in
        arcseconds.
    max_images : int or numpy.integer
        Fixed output width and maximum accepted image count; at least two.
    min_images : int or numpy.integer, optional
        Minimum count accepted after bounded recovery, from one through
        ``max_images``. A value of one permits an isolated image to complete
        the configured contract when no expectation is supplied.
    expected_num_images : object or None, optional
        Graph setter realizing to ``None`` or an integer between
        ``min_images`` and ``max_images``.
    fov : object, optional
        Graph setter realizing to ``None`` or a positive finite image-plane
        FOV in arcseconds. ``None`` uses the realized total-lens adapter's
        component-envelope extent.
    fov_multiplier : float-convertible scalar, optional
        Finite positive dimensionless multiplier applied to each realized
        ``fov``.
    pixelscale : float-convertible scalar, optional
        Finite positive configured grid-spacing upper bound in arcseconds.
    pixelscale_fraction : float-convertible scalar or None, optional
        Finite positive dimensionless fraction of a registered adapter's
        realized characteristic scale.
    epsilon : float-convertible scalar, optional
        Finite positive configured Caustics residual tolerance in arcseconds.
    epsilon_fraction : float-convertible scalar or None, optional
        Finite positive dimensionless fraction of a registered adapter's
        realized characteristic scale.
    max_depth : int or numpy.integer, optional
        Positive Caustics global-search tree depth.
    max_fov_expansions : int or numpy.integer, optional
        Non-negative outer FOV expansion count.
    fov_expansion_factor : float-convertible scalar, optional
        Finite FOV multiplier strictly greater than one.
    max_pixelscale_refinements : int or numpy.integer, optional
        Non-negative outer requested-scale refinement count.
    pixelscale_refinement_factor : float-convertible scalar, optional
        Finite requested-scale multiplier strictly between zero and one.
    node_label : str or None, optional
        Human-readable graph node identifier.

    Attributes
    ----------
    num_images : AttributeIndicator
        Graph output for active image count.
    image_x : AttributeIndicator
        Graph output for image-plane x positions in arcseconds, NaN-padded.
    image_y : AttributeIndicator
        Graph output for image-plane y positions in arcseconds, NaN-padded.
    macro_magnifications : AttributeIndicator
        Graph output for absolute dimensionless magnifications, zero-padded.
    time_delays : AttributeIndicator
        Graph output for observer-frame relative delays in days, NaN-padded.
    image_count_deficit : AttributeIndicator
        Graph output for expected minus recovered count, or ``-1`` when no
        expectation exists.
    solver_fov : AttributeIndicator
        Graph output for final accepted or bounded-deficit FOV in arcseconds.
    solver_pixelscale : AttributeIndicator
        Graph output for actual accepted spacing in arcseconds.
    solver_attempts : AttributeIndicator
        Graph output counting every global Caustics invocation and every
        executed targeted-recovery batch.
    solver_fov_expansions : AttributeIndicator
        Graph output counting outer FOV expansion steps only.
    solver_pixelscale_refinements : AttributeIndicator
        Graph output counting outer requested-scale refinement steps only.

    Notes
    -----
    ``pixelscale_fraction`` and ``epsilon_fraction`` optionally scale their
    corresponding numerical settings to each realized lens's characteristic
    angular scale. Configured absolute values remain upper bounds, realized
    values are fixed for one lens, and each attempt's requested pixelscale is
    an upper bound on the actual accepted spacing
    ``current_fov / divisions``.

    One shared registered lens system and its total geometry adapter are
    realized per graph sample. For composites, the adapter derives the
    numerical search center, starting extent, and characteristic resolution
    from all realized non-affine components. Explicit recovery points gate
    targeted searches; every active point is seeded against the total lens,
    and certified supplemental roots are retained without deduplication.

    The physical component centers are distinct from the numerical grid
    center. A half-cell recovery shift changes only the numerical search grid
    and never translates returned physical image coordinates. No realized
    adapter or lens is cached on the node. Magnifications and time delays are
    evaluated once on the realized total lens after count recovery finishes.

    FOV expansions complete before requested-scale refinements. The
    divisions-plus-one variant changes actual spacing, while the half-cell
    numerical-center shift retains the base division count and base spacing.
    Both variants add a global solver attempt, and neither changes the outer
    expansion/refinement counters. Fixed-width GraphState outputs use the
    padding and sentinel conventions documented above.

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
        lens,
        *,
        source_redshift,
        source_x,
        source_y,
        max_images,
        min_images=2,
        expected_num_images=None,
        fov=None,
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
        """Configure deterministic point-image solving and bounded recovery.

        Parameters
        ----------
        lens : CausticsLensSpec
            Registered atomic or recursive lens specification. The root owns
            ``cosmology`` and the dimensionless lens-redshift setter ``z_l``.
        source_redshift : object
            Graph setter for dimensionless source redshift.
        source_x : object
            Graph setter for source-plane x position in arcseconds.
        source_y : object
            Graph setter for source-plane y position in arcseconds.
        max_images : int or numpy.integer
            Fixed output width and maximum active count; at least two.
        min_images : int or numpy.integer, optional
            Minimum acceptable count, from one through ``max_images``.
        expected_num_images : object or None, optional
            Graph setter for an optional realized expected count.
        fov : object, optional
            Graph setter realizing to ``None`` for adapter-derived extent or a
            positive finite image-plane FOV in arcseconds.
        fov_multiplier : float-convertible scalar, optional
            Finite positive multiplier converting realized ``fov`` to the
            initial solver FOV.
        pixelscale : float-convertible scalar, optional
            Finite positive configured grid-spacing upper bound in arcseconds.
        pixelscale_fraction : float-convertible scalar or None, optional
            Finite positive dimensionless relative scale; a zero-dimensional
            NumPy array is accepted.
        epsilon : float-convertible scalar, optional
            Finite positive configured residual tolerance in arcseconds.
        epsilon_fraction : float-convertible scalar or None, optional
            Finite positive dimensionless relative tolerance; a
            zero-dimensional NumPy array is accepted.
        max_depth : int or numpy.integer, optional
            Positive Caustics global-search depth.
        max_fov_expansions : int or numpy.integer, optional
            Non-negative outer FOV expansion limit.
        fov_expansion_factor : float-convertible scalar, optional
            Finite FOV multiplier strictly greater than one.
        max_pixelscale_refinements : int or numpy.integer, optional
            Non-negative outer requested-scale refinement limit.
        pixelscale_refinement_factor : float-convertible scalar, optional
            Finite scale multiplier strictly between zero and one.
        node_label : str or None, optional
            Human-readable graph node identifier.

        Returns
        -------
        None
            The configured function node registers its inputs and outputs.

        Raises
        ------
        TypeError
            If ``lens``, fractions, or normalized scalar settings have invalid
            types.
        ValueError
            If the lens specification or counts, depths, recovery limits,
            factors, or positive scalar settings violate their ranges or
            relations.

        Notes
        -----
        Source coordinates, source redshift, FOV, and optional expected count
        are registered directly. Lens parameters use ``lens_<field>`` at the
        root and positional path segments inside nested planes. All eleven
        public outputs are registered in ``_OUTPUTS`` order.
        """
        if not isinstance(lens, CausticsLensSpec):
            raise TypeError("lens must be a CausticsLensSpec.")
        _validate_root_lens_spec(lens)
        lens_graph_inputs = _lens_graph_inputs(lens)

        pixelscale_fraction = _validate_optional_positive_fraction(
            "pixelscale_fraction",
            pixelscale_fraction,
        )
        epsilon_fraction = _validate_optional_positive_fraction(
            "epsilon_fraction",
            epsilon_fraction,
        )
        integer_types = (int, np.integer)
        if not isinstance(max_images, integer_types) or max_images < 2:
            raise ValueError("max_images must be an integer greater than one.")
        if not isinstance(min_images, integer_types) or not 1 <= min_images <= max_images:
            raise ValueError("min_images must be between one and max_images.")
        scalar_settings = {
            "fov_multiplier": fov_multiplier,
            "pixelscale": pixelscale,
            "epsilon": epsilon,
            "fov_expansion_factor": fov_expansion_factor,
            "pixelscale_refinement_factor": pixelscale_refinement_factor,
        }
        normalized_scalars = {}
        for name, value in scalar_settings.items():
            try:
                normalized_scalars[name] = float(value)
            except (TypeError, ValueError, OverflowError) as err:
                raise TypeError(f"{name} must be a scalar number.") from err
        for name in ("fov_multiplier", "pixelscale", "epsilon"):
            if not np.isfinite(normalized_scalars[name]) or normalized_scalars[name] <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        if (
            not np.isfinite(normalized_scalars["fov_expansion_factor"])
            or normalized_scalars["fov_expansion_factor"] <= 1.0
        ):
            raise ValueError("fov_expansion_factor must be finite and greater than one.")
        if (
            not np.isfinite(normalized_scalars["pixelscale_refinement_factor"])
            or not 0.0 < normalized_scalars["pixelscale_refinement_factor"] < 1.0
        ):
            raise ValueError("pixelscale_refinement_factor must be finite and strictly between zero and one.")
        if not isinstance(max_depth, integer_types) or max_depth < 1:
            raise ValueError("max_depth must be a positive integer.")
        for name, limit in (
            ("max_fov_expansions", max_fov_expansions),
            ("max_pixelscale_refinements", max_pixelscale_refinements),
        ):
            if not isinstance(limit, integer_types) or limit < 0:
                raise ValueError(f"{name} must be a non-negative integer.")

        self.lens = lens
        self.max_images = int(max_images)
        self.min_images = int(min_images)
        self.fov_multiplier = normalized_scalars["fov_multiplier"]
        self.pixelscale = normalized_scalars["pixelscale"]
        self.pixelscale_fraction = pixelscale_fraction
        self.epsilon = normalized_scalars["epsilon"]
        self.epsilon_fraction = epsilon_fraction
        self.max_depth = int(max_depth)
        self.max_fov_expansions = int(max_fov_expansions)
        self.fov_expansion_factor = normalized_scalars["fov_expansion_factor"]
        self.max_pixelscale_refinements = int(max_pixelscale_refinements)
        self.pixelscale_refinement_factor = normalized_scalars["pixelscale_refinement_factor"]

        node_inputs = {
            "source_redshift": source_redshift,
            "source_x": source_x,
            "source_y": source_y,
            "fov": fov,
            "expected_num_images": expected_num_images,
        }
        for name, setter in lens_graph_inputs:
            node_inputs[name] = setter

        super().__init__(
            self._non_func,
            node_label=node_label,
            outputs=self._OUTPUTS,
            **node_inputs,
        )

    def _realized_angular_settings(self, geometry_adapter, values):
        """Realize per-lens grid-scale and residual-tolerance settings.

        Parameters
        ----------
        geometry_adapter : _GeometryAdapter
            Realized total-lens adapter held stable for this realization.
        values : Mapping[str, object]
            Adapter-owned realized inputs for one lens system.

        Returns
        -------
        realized_pixelscale : float
            Initial requested grid-spacing upper bound in arcseconds.
        realized_epsilon : float
            Fixed Caustics residual tolerance in arcseconds.

        Notes
        -----
        Configured absolute values remain upper bounds. When either fraction is
        enabled, one trusted characteristic scale is obtained and each enabled
        setting becomes the smaller of its absolute and relative value. For a
        composite, the adapter's resolution is derived from every non-affine
        component.
        """
        realized_pixelscale = self.pixelscale
        realized_epsilon = self.epsilon
        if self.pixelscale_fraction is None and self.epsilon_fraction is None:
            return realized_pixelscale, realized_epsilon

        characteristic_scale = geometry_adapter.resolution_scale(values)
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
        """Execute exactly one global Caustics image search.

        Parameters
        ----------
        lens : object
            Realized Caustics lens implementing ``forward_raytrace``.
        torch : module
            PyTorch module used by the realized lens.
        beta_x : torch.Tensor
            Scalar source-plane x coordinate in arcseconds.
        beta_y : torch.Tensor
            Scalar source-plane y coordinate in arcseconds.
        center_x : float
            Numerical search-grid x center in image-plane arcseconds.
        center_y : float
            Numerical search-grid y center in image-plane arcseconds.
        current_fov : float
            Current square image-plane FOV in arcseconds.
        divisions : int
            Number of equal grid divisions per axis.
        epsilon : float
            Fixed realized Caustics residual tolerance in arcseconds.

        Returns
        -------
        numpy.ndarray, shape (I, 2)
            Global image coordinates in the physical image-plane coordinate
            system, in arcseconds.

        Notes
        -----
        This helper is the architectural boundary for exactly one Caustics
        invocation. A shifted numerical grid center changes only the search
        grid; returned coordinates are never translated. Paired output shapes,
        types, and finiteness are trusted Caustics postconditions.
        """
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
        return np.column_stack((_to_numpy(image_x), _to_numpy(image_y)))

    def _solve_one(self, values):
        """Solve active macro-images for one realized lens system.

        Parameters
        ----------
        values : Mapping
            Inputs for exactly one graph sample. Required entries are
            dimensionless ``source_redshift``; finite scalar source-plane
            ``source_x`` and ``source_y`` in arcseconds; realized ``fov``,
            which may be ``None`` or a positive finite value in arcseconds;
            required ``expected_num_images``, whose value may be ``None`` or
            an ordinary integer in the configured range; and every flattened
            lens-specification graph input.

        Returns
        -------
        image_x : numpy.ndarray, shape (I,)
            Active image-plane x positions in arcseconds.
        image_y : numpy.ndarray, shape (I,)
            Active image-plane y positions in arcseconds.
        macro_magnifications : numpy.ndarray, shape (I,)
            Absolute dimensionless macro-magnifications.
        time_delays : numpy.ndarray, shape (I,)
            Observer-frame relative delays in days, normalized to start at zero.
        diagnostics : dict
            Six scalar entries. ``image_count_deficit`` is expected minus
            recovered count, or ``-1`` without an expectation;
            ``solver_fov`` is the final accepted or bounded-deficit FOV in
            arcseconds; ``solver_pixelscale`` is actual accepted grid spacing
            in arcseconds; ``solver_attempts`` counts global calls and
            executed targeted batches; and ``solver_fov_expansions`` and
            ``solver_pixelscale_refinements`` count outer steps only.

        Notes
        -----
        All four arrays use the same deterministic ordering: increasing delay,
        then image x, then image y. Padding to ``max_images`` is performed by
        ``compute``.

        A ``None`` FOV uses the realized adapter's total-lens extent; an
        explicit realized FOV overrides that extent. Either value is multiplied
        by ``fov_multiplier`` to form ``initial_fov``. Realized pixel scale and
        epsilon are fixed once from the total adapter. Every global call and
        residual certification uses that fixed epsilon; targeted seed radius
        is ``min(realized_epsilon, actual_grid_spacing)``, while recovery-point
        neighborhood occupancy and root locality use the actual grid spacing.
        Each outer attempt is independent, so its global coordinates replace
        rather than merge with any earlier attempt. Supplemental roots are
        local to a successful, nonempty, deficient attempt and run only when
        the adapter supplies explicit recovery points.

        The divisions-plus-one variant changes actual spacing; the half-cell
        numerical-center shift retains base divisions and base spacing. Both
        add global attempts, and neither changes outer recovery counters. The
        numerical shift leaves physical lens values and returned coordinates
        unchanged. All bounded FOV expansions run before requested-scale
        refinements. After retryable exhaustion, ``None`` coordinates are
        normalized to an empty array; recovery below ``min_images`` raises,
        while a bounded deficit at or above that minimum may be returned.

        Raises
        ------
        ImportError
            If optional Caustics runtime dependencies are unavailable.
        KeyError
            If a required realized input is absent.
        TypeError
            If an owned realized scalar cannot be normalized or a redshift is
            not scalar.
        ValueError
            If source coordinates, FOV, expected count, redshifts, or the
            realized lens specification violate their owned domains.
        RuntimeError
            If a recovered count exceeds its expectation or ``max_images``,
            or bounded recovery exhausts below ``min_images``.
        """
        source_coordinates = {}
        for name in ("source_x", "source_y"):
            try:
                source_coordinates[name] = float(values[name])
            except (TypeError, ValueError, OverflowError) as err:
                raise TypeError(f"{name} must realize to a scalar numeric value in arcseconds.") from err
            if not np.isfinite(source_coordinates[name]):
                raise ValueError(f"{name} must realize to a finite value in arcseconds.")
        source_x = source_coordinates["source_x"]
        source_y = source_coordinates["source_y"]

        expected_num_images = values["expected_num_images"]
        if expected_num_images is not None:
            if (
                not isinstance(expected_num_images, (int, np.integer))
                or not self.min_images <= expected_num_images <= self.max_images
            ):
                raise ValueError(
                    "expected_num_images must be None or an integer between min_images and max_images."
                )
            expected_num_images = int(expected_num_images)

        lens, geometry_adapter, adapter_values = _build_lens_system(
            self.lens,
            values=values,
        )
        _, torch = _import_caustics_dependencies()

        realized_fov = values["fov"]
        if realized_fov is None:
            realized_fov = geometry_adapter.initial_fov(adapter_values)
        else:
            try:
                realized_fov = float(realized_fov)
            except (TypeError, ValueError, OverflowError) as err:
                raise TypeError("fov must realize to None or a scalar numeric value.") from err
            if not np.isfinite(realized_fov) or realized_fov <= 0.0:
                raise ValueError("fov must realize to None or a positive finite value.")
        initial_fov = realized_fov * self.fov_multiplier
        realized_pixelscale, realized_epsilon = self._realized_angular_settings(
            geometry_adapter,
            adapter_values,
        )
        if initial_fov <= realized_pixelscale:
            raise ValueError(
                f"Initial solver fov={initial_fov} arcsec must be larger "
                f"than pixelscale={realized_pixelscale} arcsec."
            )

        beta_x = torch.as_tensor(source_x, dtype=torch.float64)
        beta_y = torch.as_tensor(source_y, dtype=torch.float64)
        center_x, center_y = geometry_adapter.search_center(adapter_values)
        target_count = self.min_images if expected_num_images is None else expected_num_images
        latest_retryable_error = None
        solver_attempts = 0

        current_fov = initial_fov
        current_pixelscale = realized_pixelscale
        current_grid_pixelscale = None
        current_grid_variant = None
        fov_expansions = 0
        pixelscale_refinements = 0

        def recovery_context(recovered_count, recovery_stage):
            """Format the current realization's mutable recovery state.

            Parameters
            ----------
            recovered_count : int
                Number of image coordinates currently available.
            recovery_stage : str
                Label for the outer or targeted recovery stage being reported.

            Returns
            -------
            str
                Diagnostic text containing the product ``initial_fov``,
                current FOV, configured and realized scale/tolerance values,
                current requested and actual grid scales, grid variant, outer
                counters, solver-attempt count, expected/maximum/recovered
                counts, and the latest retryable exception.

            Notes
            -----
            This closure reads enclosing mutable recovery state but performs no
            solver call and changes no counter. ``initial_fov`` already
            includes the configured ``fov_multiplier``; the raw realized FOV
            and multiplier are not separate diagnostic fields.
            """
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
            """Apply expected, maximum, and minimum image-count policy.

            Parameters
            ----------
            coordinates : numpy.ndarray, shape (I, 2)
                Current physical image-plane coordinates in arcseconds.
            recovery_stage : str
                Stage label included in count-overflow diagnostics.

            Returns
            -------
            bool
                With an expectation, whether the recovered count equals it;
                otherwise, whether the count is at least ``min_images``.

            Raises
            ------
            RuntimeError
                If the recovered count exceeds ``expected_num_images`` when
                present or exceeds ``max_images``.
            """
            recovered_count = len(coordinates)
            if expected_num_images is not None and recovered_count > expected_num_images:
                raise RuntimeError(
                    "Caustics image recovery exceeded expected_num_images; "
                    f"{recovery_context(recovered_count, recovery_stage)}."
                )
            if recovered_count > self.max_images:
                raise RuntimeError(
                    "Caustics image recovery exceeded max_images; "
                    f"{recovery_context(recovered_count, recovery_stage)}."
                )
            if expected_num_images is not None:
                return recovered_count == expected_num_images
            return recovered_count >= self.min_images

        def attempt(current_fov, current_pixelscale, *, recovery_stage):
            """Run one independent grid attempt and optional targeted recovery.

            Parameters
            ----------
            current_fov : float
                Current square image-plane FOV in arcseconds.
            current_pixelscale : float
                Requested per-attempt grid-spacing upper bound in arcseconds.
            recovery_stage : str
                Outer schedule stage used in diagnostics.

            Returns
            -------
            coordinates : numpy.ndarray or None
                Physical image-plane coordinates with shape ``(I, 2)`` in
                arcseconds, or ``None`` after retryable global exhaustion.
            complete : bool
                Whether the returned coordinates satisfy the active image-count
                target.

            Raises
            ------
            RuntimeError
                If a recovered count exceeds the expected or maximum count.

            Notes
            -----
            The global variant order is the base grid, a divisions-plus-one
            parity grid, then a half-cell numerical-center shift at the base
            division count. Only the exact singular linear-solve classifier
            advances that inner sequence. Three singular failures exhaust
            naturally through the loop ``else``; the known empty-candidate
            failure immediately returns ``(None, False)`` for retry only by
            the surrounding bounded outer schedule. Unrelated exceptions
            propagate unchanged.

            Every actual global Caustics invocation increments
            ``solver_attempts``. Each executed recovery-seed batch adds one
            more attempt, while its internal root-refinement passes do not.
            Actual spacing is updated to ``current_fov / divisions`` for the
            invoked variant. The divisions-plus-one variant changes that
            spacing; the half-cell numerical-center shift uses base divisions
            and therefore retains base spacing. Both add a global attempt, and
            neither changes the outer expansion/refinement counters.

            Targeted recovery runs only after a successful, nonempty, deficient
            global result with explicit adapter recovery points. It selects one
            seed for each empty recovery neighborhood over the total lens. Its
            radius is the smaller of fixed realized epsilon and actual grid
            spacing. Residual certification uses fixed epsilon, while
            neighborhood occupancy and root locality use actual grid spacing.
            Retryable targeted failures leave the global result deficient.
            Targeted roots and global coordinates are local to this attempt;
            every later outer call replaces them.

            Numerical-center shifts never modify physical lens values or
            translate returned image coordinates. If all outer calls remain
            retryable, the enclosing solver's reachable
            ``coordinates is None`` branch converts the result to shape
            ``(0, 2)``. The enclosing schedule completes every FOV expansion
            before beginning requested-scale refinement.
            """
            nonlocal solver_attempts, current_grid_pixelscale, current_grid_variant
            nonlocal latest_retryable_error
            base_divisions = int(np.ceil(current_fov / current_pixelscale))
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
                    is_singular_error = _is_singular_forward_raytrace_error(error)
                    is_empty_candidate_error = _is_retryable_forward_raytrace_error(error)
                    if is_singular_error:
                        latest_retryable_error = f"{type(error).__name__}: {error}"
                        continue
                    if is_empty_candidate_error:
                        latest_retryable_error = f"{type(error).__name__}: {error}"
                        return None, False
                    raise
                break
            else:
                return None, False

            complete = result_is_complete(
                coordinates,
                recovery_stage=recovery_stage,
            )

            recovery_points = np.asarray(
                geometry_adapter.root_recovery_points(adapter_values),
                dtype=float,
            ).reshape(-1, 2)
            if complete or not len(coordinates) or not len(recovery_points):
                return coordinates, complete

            empty_neighborhoods = _recovery_neighborhoods_are_empty(
                coordinates,
                recovery_points,
                current_grid_pixelscale,
            )
            if not np.any(empty_neighborhoods):
                return coordinates, False

            seeds = _recovery_image_seeds(
                lens,
                recovery_points,
                source_x=source_x,
                source_y=source_y,
                radius=min(realized_epsilon, current_grid_pixelscale),
            )
            seeds = seeds[empty_neighborhoods]
            unresolved_points = recovery_points[empty_neighborhoods]

            solver_attempts += 1
            try:
                recovery_coordinates = _refine_image_seeds(
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
                is_singular_error = _is_singular_forward_raytrace_error(error)
                is_empty_candidate_error = _is_retryable_forward_raytrace_error(error)
                if not (is_singular_error or is_empty_candidate_error):
                    raise
                latest_retryable_error = f"{type(error).__name__}: {error}"
                return coordinates, False

            if len(recovery_coordinates):
                coordinates = np.vstack((coordinates, recovery_coordinates))
            return coordinates, result_is_complete(
                coordinates,
                recovery_stage="recovery_seed",
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
        """Solve every sampled lens and save fixed-width numeric outputs.

        Parameters
        ----------
        graph_state : GraphState
            State containing realized node inputs and receiving all eleven
            computed outputs.
        rng_info : object, optional
            Ignored. Image solving is deterministic for realized inputs.
        **kwargs : dict, optional
            Explicit overrides keyed by registered node input name.

        Returns
        -------
        results : list
            Eleven values in this exact order:

            1. ``num_images``, active image count;
            2. ``image_x``, image-plane arcseconds, NaN-padded;
            3. ``image_y``, image-plane arcseconds, NaN-padded;
            4. ``macro_magnifications``, absolute dimensionless values,
               zero-padded;
            5. ``time_delays``, observer-frame days relative to zero,
               NaN-padded;
            6. ``image_count_deficit``, expected minus recovered count or
               ``-1`` without an expectation;
            7. ``solver_fov``, final accepted or bounded-deficit FOV in
               arcseconds;
            8. ``solver_pixelscale``, actual accepted grid spacing in
               arcseconds;
            9. ``solver_attempts``, global calls plus executed recovery-seed
               batches;
            10. ``solver_fov_expansions``, outer FOV steps only;
            11. ``solver_pixelscale_refinements``, outer requested-scale
                steps only.

            For one sample, count and diagnostic outputs are scalars and each
            fixed-width image output has shape ``(M,)``, where
            ``M = max_images``. For ``S > 1``, count and diagnostic outputs
            have shape ``(S,)``, and image outputs have shape ``(S, M)``.

        Notes
        -----
        The method is deterministic and does not consume ``rng_info``. It
        saves all eleven values to this node's ``GraphState`` entries in
        ``_OUTPUTS`` order.
        """
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

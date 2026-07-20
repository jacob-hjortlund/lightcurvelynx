"""Caustics-backed nodes for strong-lens image configurations."""

import inspect
import keyword
from collections.abc import Mapping
from dataclasses import dataclass
from math import hypot
from numbers import Real
from types import MappingProxyType

import numpy as np
from citation_compass import CiteClass
from scipy.optimize import linear_sum_assignment

from lightcurvelynx.base_models import FunctionNode

__all__ = [
    "CausticsLensImageNode",
    "CausticsLensSpec",
    "CausticsSinglePlaneSpec",
    "CausticsSourcePositionNode",
]

_RESERVED_LENS_PARAMETERS = {
    "cosmology",
    "name",
    "z_l",
    "z_s",
}
_RESERVED_LENS_SPEC_FIELDS = _RESERVED_LENS_PARAMETERS | {"lenses"}
_MISSING = object()


@dataclass(frozen=True)
class _LensFieldSchema:
    default: object = _MISSING
    kind: str = "finite"


@dataclass(frozen=True)
class _LensModelSchema:
    model: str
    fields: Mapping[str, _LensFieldSchema]
    always_required: tuple[str, ...]
    selector_branches: Mapping[str, Mapping[object, tuple[str, ...]]]
    mutually_exclusive_fields: Mapping[str, tuple[str, ...]]
    affine: bool = False


@dataclass(frozen=True)
class _RegisteredLensModel:
    schema: _LensModelSchema
    lens_class: object
    geometry_factory: object


_LENS_MODEL_SCHEMAS = MappingProxyType(
    {
        "SIS": _LensModelSchema(
            model="SIS",
            fields=MappingProxyType(
                {
                    "x0": _LensFieldSchema(),
                    "y0": _LensFieldSchema(),
                    "Rein": _LensFieldSchema(kind="positive"),
                    "s": _LensFieldSchema(default=0.0, kind="nonnegative"),
                }
            ),
            always_required=("x0", "y0", "Rein"),
            selector_branches=MappingProxyType({}),
            mutually_exclusive_fields=MappingProxyType({}),
        ),
        "SIE": _LensModelSchema(
            model="SIE",
            fields=MappingProxyType(
                {
                    "x0": _LensFieldSchema(),
                    "y0": _LensFieldSchema(),
                    "q": _LensFieldSchema(kind="q"),
                    "phi": _LensFieldSchema(),
                    "Rein": _LensFieldSchema(kind="positive"),
                    "parametrization": _LensFieldSchema(default="Rein", kind="selector"),
                    "sigma_v": _LensFieldSchema(kind="positive"),
                    "angle_system": _LensFieldSchema(default="q_phi", kind="selector"),
                    "e1": _LensFieldSchema(kind="ellipticity_pair"),
                    "e2": _LensFieldSchema(kind="ellipticity_pair"),
                    "c1": _LensFieldSchema(),
                    "c2": _LensFieldSchema(),
                    "s": _LensFieldSchema(default=0.0, kind="nonnegative"),
                }
            ),
            always_required=("x0", "y0"),
            selector_branches=MappingProxyType(
                {
                    "parametrization": MappingProxyType(
                        {
                            "Rein": ("Rein",),
                            "velocity_dispersion": ("sigma_v",),
                        }
                    ),
                    "angle_system": MappingProxyType(
                        {
                            "q_phi": ("q", "phi"),
                            "e1_e2": ("e1", "e2"),
                            "c1_c2": ("c1", "c2"),
                        }
                    ),
                }
            ),
            mutually_exclusive_fields=MappingProxyType(
                {
                    "parametrization": ("Rein", "sigma_v"),
                    "angle_system": ("q", "phi", "e1", "e2", "c1", "c2"),
                }
            ),
        ),
        "EPL": _LensModelSchema(
            model="EPL",
            fields=MappingProxyType(
                {
                    "x0": _LensFieldSchema(),
                    "y0": _LensFieldSchema(),
                    "q": _LensFieldSchema(kind="q"),
                    "phi": _LensFieldSchema(),
                    "e1": _LensFieldSchema(kind="ellipticity_pair"),
                    "e2": _LensFieldSchema(kind="ellipticity_pair"),
                    "c1": _LensFieldSchema(),
                    "c2": _LensFieldSchema(),
                    "Rein": _LensFieldSchema(kind="positive"),
                    "t": _LensFieldSchema(kind="t"),
                    "angle_system": _LensFieldSchema(default="q_phi", kind="selector"),
                    "s": _LensFieldSchema(default=0.0, kind="nonnegative"),
                    "n_iter": _LensFieldSchema(default=18, kind="positive_int"),
                    "chunk_size": _LensFieldSchema(default=None, kind="chunk_size"),
                }
            ),
            always_required=("x0", "y0", "Rein", "t"),
            selector_branches=MappingProxyType(
                {
                    "angle_system": MappingProxyType(
                        {
                            "q_phi": ("q", "phi"),
                            "e1_e2": ("e1", "e2"),
                            "c1_c2": ("c1", "c2"),
                        }
                    )
                }
            ),
            mutually_exclusive_fields=MappingProxyType(
                {"angle_system": ("q", "phi", "e1", "e2", "c1", "c2")}
            ),
        ),
        "NFW": _LensModelSchema(
            model="NFW",
            fields=MappingProxyType(
                {
                    "x0": _LensFieldSchema(),
                    "y0": _LensFieldSchema(),
                    "mass": _LensFieldSchema(kind="positive"),
                    "c": _LensFieldSchema(kind="positive"),
                    "s": _LensFieldSchema(default=0.0, kind="nonnegative"),
                }
            ),
            always_required=("x0", "y0", "mass", "c"),
            selector_branches=MappingProxyType({}),
            mutually_exclusive_fields=MappingProxyType({}),
        ),
        "TNFW": _LensModelSchema(
            model="TNFW",
            fields=MappingProxyType(
                {
                    "x0": _LensFieldSchema(),
                    "y0": _LensFieldSchema(),
                    "mass": _LensFieldSchema(kind="positive"),
                    "Rs": _LensFieldSchema(kind="positive"),
                    "tau": _LensFieldSchema(kind="positive"),
                    "s": _LensFieldSchema(default=0.0, kind="nonnegative"),
                    "interpret_m_total_mass": _LensFieldSchema(default=True, kind="bool"),
                }
            ),
            always_required=("x0", "y0", "mass", "Rs", "tau"),
            selector_branches=MappingProxyType({}),
            mutually_exclusive_fields=MappingProxyType({}),
        ),
        "PseudoJaffe": _LensModelSchema(
            model="PseudoJaffe",
            fields=MappingProxyType(
                {
                    "x0": _LensFieldSchema(),
                    "y0": _LensFieldSchema(),
                    "mass": _LensFieldSchema(kind="positive"),
                    "Rc": _LensFieldSchema(kind="positive"),
                    "Rs": _LensFieldSchema(kind="positive"),
                    "s": _LensFieldSchema(default=0.0, kind="nonnegative"),
                }
            ),
            always_required=("x0", "y0", "mass", "Rc", "Rs"),
            selector_branches=MappingProxyType({}),
            mutually_exclusive_fields=MappingProxyType({}),
        ),
        "ExternalShear": _LensModelSchema(
            model="ExternalShear",
            fields=MappingProxyType(
                {
                    "x0": _LensFieldSchema(),
                    "y0": _LensFieldSchema(),
                    "gamma_1": _LensFieldSchema(),
                    "gamma_2": _LensFieldSchema(),
                    "parametrization": _LensFieldSchema(default="cartesian", kind="selector"),
                    "gamma": _LensFieldSchema(kind="nonnegative"),
                    "phi": _LensFieldSchema(),
                    "s": _LensFieldSchema(default=0.0, kind="nonnegative"),
                }
            ),
            always_required=("x0", "y0"),
            selector_branches=MappingProxyType(
                {
                    "parametrization": MappingProxyType(
                        {
                            "cartesian": ("gamma_1", "gamma_2"),
                            "angular": ("gamma", "phi"),
                        }
                    )
                }
            ),
            mutually_exclusive_fields=MappingProxyType(
                {"parametrization": ("gamma_1", "gamma_2", "gamma", "phi")}
            ),
            affine=True,
        ),
        "MassSheet": _LensModelSchema(
            model="MassSheet",
            fields=MappingProxyType(
                {
                    "x0": _LensFieldSchema(),
                    "y0": _LensFieldSchema(),
                    "kappa": _LensFieldSchema(),
                }
            ),
            always_required=("x0", "y0", "kappa"),
            selector_branches=MappingProxyType({}),
            mutually_exclusive_fields=MappingProxyType({}),
            affine=True,
        ),
    }
)


def _snapshot_string_mapping(name, mapping):
    if not isinstance(mapping, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    snapshot = dict(mapping)
    if any(not isinstance(key, str) for key in snapshot):
        raise TypeError(f"{name} keys must be strings.")
    return snapshot


def _get_lens_model_schema(model):
    if not isinstance(model, str) or not model:
        raise TypeError("model must be a non-empty string.")
    try:
        return _LENS_MODEL_REGISTRY[model].schema
    except KeyError as err:
        raise ValueError(f"Unregistered Caustics lens model {model!r}.") from err


def _validate_option_literal(name, value):
    if value is None or isinstance(value, (str, bool, np.bool_, Real)):
        return
    if isinstance(value, tuple):
        for index, item in enumerate(value):
            _validate_option_literal(f"{name}[{index}]", item)
        return
    raise TypeError(f"{name} must be a recursively immutable literal value.")


def _validate_finite(name, value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar excluding booleans.")
    if not -np.inf < value < np.inf:
        raise ValueError(f"{name} must be finite.")
    return value


def _validate_field_value(schema, name, value):
    field = schema.fields[name]
    if field.kind in {"finite", "positive", "nonnegative", "q", "t", "ellipticity_pair"}:
        normalized = _validate_finite(name, value)
        if field.kind == "positive" and normalized <= 0:
            raise ValueError(f"{name} must be positive.")
        if field.kind == "nonnegative" and normalized < 0:
            raise ValueError(f"{name} must be nonnegative.")
        if field.kind == "q" and not 0 < normalized <= 1:
            raise ValueError(f"{name} must satisfy 0 < {name} <= 1.")
        if field.kind == "t" and not 0 < normalized < 2:
            raise ValueError(f"{name} must satisfy 0 < {name} < 2.")
        return normalized
    if field.kind == "positive_int":
        if isinstance(value, (bool, np.bool_)) or (
            type(value) is not int and not isinstance(value, np.integer)
        ):
            raise TypeError(f"{name} must be an integer excluding booleans.")
        if value < 1:
            raise ValueError(f"{name} must be at least 1.")
        return value
    if field.kind == "chunk_size":
        if value is None:
            return None
        if isinstance(value, (bool, np.bool_)) or (
            type(value) is not int and not isinstance(value, np.integer)
        ):
            raise TypeError(f"{name} must be None or an integer excluding booleans.")
        if value != 1:
            raise ValueError(f"{name} must be None or the integer 1.")
        return value
    if field.kind == "bool":
        if not isinstance(value, (bool, np.bool_)):
            raise TypeError(f"{name} must be a boolean.")
        return bool(value)
    if field.kind == "selector":
        branches = schema.selector_branches[name]
        if not isinstance(value, str):
            raise TypeError(f"{name} must be a string selector.")
        if value not in branches:
            choices = ", ".join(repr(choice) for choice in branches)
            raise ValueError(f"{name} must be one of {choices}.")
        return value


def _validate_ellipticity_pair(schema, values):
    pair = tuple(name for name, field in schema.fields.items() if field.kind == "ellipticity_pair")
    if pair and all(name in values for name in pair):
        first, second = pair
        first_value = values[first]
        second_value = values[second]
        if abs(first_value) >= 1 or abs(second_value) >= 1 or hypot(first_value, second_value) >= 1:
            raise ValueError(f"{first} and {second} must satisfy hypot({first}, {second}) < 1.")


def _validate_pseudo_jaffe_radii(schema, values):
    if schema.model == "PseudoJaffe" and "Rc" in values and "Rs" in values and values["Rc"] >= values["Rs"]:
        raise ValueError("PseudoJaffe requires Rc < Rs.")


def _validate_selector_fields(schema, values):
    for selector, branches in schema.selector_branches.items():
        selected = values[selector]
        active_fields = branches[selected]
        missing = tuple(name for name in active_fields if name not in values)
        if missing:
            names = ", ".join(missing)
            raise ValueError(f"{schema.model} selector {selector}={selected!r} requires: {names}.")
        inactive_fields = set(schema.mutually_exclusive_fields[selector]).difference(active_fields)
        supplied_inactive = tuple(name for name in inactive_fields if name in values)
        if supplied_inactive:
            names = ", ".join(sorted(supplied_inactive))
            raise ValueError(f"{schema.model} selector {selector}={selected!r} does not accept: {names}.")


def _validate_realized_fields(schema, values):
    unknown = tuple(name for name in values if name not in schema.fields)
    if unknown:
        names = ", ".join(unknown)
        raise ValueError(f"Unknown {schema.model} field(s): {names}.")
    missing = tuple(name for name in schema.always_required if name not in values)
    if missing:
        names = ", ".join(missing)
        raise ValueError(f"Missing required {schema.model} field(s): {names}.")
    for name in schema.fields:
        if name in values:
            values[name] = _validate_field_value(schema, name, values[name])
    _validate_selector_fields(schema, values)
    _validate_ellipticity_pair(schema, values)
    _validate_pseudo_jaffe_radii(schema, values)


def _validate_spec_fields(schema, parameters, options):
    supplied = set(parameters).union(options)
    reserved = _RESERVED_LENS_SPEC_FIELDS.intersection(supplied)
    if reserved:
        names = ", ".join(sorted(reserved))
        raise ValueError(f"Reserved lens field name(s): {names}.")
    unknown = supplied.difference(schema.fields)
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown {schema.model} field(s): {names}.")
    missing = set(schema.always_required).difference(supplied)
    if missing:
        names = ", ".join(name for name in schema.always_required if name in missing)
        raise ValueError(f"Missing required {schema.model} field(s): {names}.")

    for name, value in options.items():
        _validate_field_value(schema, name, value)

    fixed_options = {name: value for name, value in options.items() if name not in parameters}
    _validate_ellipticity_pair(schema, fixed_options)
    _validate_pseudo_jaffe_radii(schema, fixed_options)

    for selector, branches in schema.selector_branches.items():
        if selector in parameters:
            continue
        selected = options.get(selector, schema.fields[selector].default)
        selected = _validate_field_value(schema, selector, selected)
        active_fields = branches[selected]
        missing = tuple(name for name in active_fields if name not in supplied)
        if missing:
            names = ", ".join(missing)
            raise ValueError(f"{schema.model} selector {selector}={selected!r} requires: {names}.")
        inactive_fields = set(schema.mutually_exclusive_fields[selector]).difference(active_fields)
        supplied_inactive = tuple(name for name in inactive_fields if name in supplied)
        if supplied_inactive:
            names = ", ".join(sorted(supplied_inactive))
            raise ValueError(f"{schema.model} selector {selector}={selected!r} does not accept: {names}.")


def _realize_component_values(schema, parameter_values, options):
    values = {name: field.default for name, field in schema.fields.items() if field.default is not _MISSING}
    values.update(options)
    values.update(parameter_values)
    _validate_realized_fields(schema, values)
    return MappingProxyType(values)


@dataclass(frozen=True, init=False)
class CausticsLensSpec:
    """Immutable specification of one registered Caustics lens component."""

    model: str
    parameters: Mapping[str, object]
    options: Mapping[str, object]

    def __init__(self, model, parameters, options=None):
        schema = _get_lens_model_schema(model)
        parameter_snapshot = _snapshot_string_mapping("parameters", parameters)
        option_snapshot = _snapshot_string_mapping("options", {} if options is None else options)
        for name, value in option_snapshot.items():
            _validate_option_literal(f"options[{name!r}]", value)
        _validate_spec_fields(schema, parameter_snapshot, option_snapshot)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "parameters", MappingProxyType(parameter_snapshot))
        object.__setattr__(self, "options", MappingProxyType(option_snapshot))


@dataclass(frozen=True, init=False)
class CausticsSinglePlaneSpec:
    """Immutable ordered collection of lens components on one plane."""

    components: Mapping[str, CausticsLensSpec]

    def __init__(self, components):
        component_snapshot = _snapshot_string_mapping("components", components)
        if not component_snapshot:
            raise ValueError("CausticsSinglePlaneSpec components must be nonempty.")
        for name, component in component_snapshot.items():
            if not name.isidentifier() or keyword.iskeyword(name):
                raise ValueError(f"Component name {name!r} must be a non-keyword Python identifier.")
            if not isinstance(component, CausticsLensSpec):
                raise TypeError(f"Component {name!r} must be a CausticsLensSpec.")
        object.__setattr__(self, "components", MappingProxyType(component_snapshot))


_RECOVERY_SEED_POINTS = 256
_RECOVERY_ROOT_REFINEMENTS = 8
_INITIAL_FOV_PADDING = 1.1


def _validate_lens_configuration(lens_model, lens_parameters):
    """Validate constructor inputs shared by Caustics-backed lens nodes.

    Parameters
    ----------
    lens_model : str
        Name of a lens class exposed by the top-level ``caustics`` package.
    lens_parameters : Mapping[str, object]
        Mapping from Caustics constructor parameter names to LightCurveLynx
        parameter setters. Values are not required to be numeric until graph
        sampling realizes them.

    Returns
    -------
    None
        The inputs are valid at the shared LightCurveLynx boundary.

    Raises
    ------
    TypeError
        If ``lens_model`` is not a non-empty string or ``lens_parameters`` is
        not a mapping with string keys.
    ValueError
        If a lens parameter would collide with a constructor argument managed
        internally by the adapter.

    Notes
    -----
    Caustics-model-specific physical parameter domains are owned by Caustics or
    the selected registered geometry adapter and are not checked here.
    """
    if not isinstance(lens_model, str) or not lens_model:
        raise TypeError("lens_model must be a non-empty Caustics class name.")
    if not isinstance(lens_parameters, Mapping):
        raise TypeError("lens_parameters must be a mapping.")
    if any(not isinstance(name, str) for name in lens_parameters):
        raise TypeError("lens_parameters keys must be strings.")

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
        import torch

        torch.set_default_dtype(torch.float64)
        import caustics
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
        Refined roots passing source-residual and singular-locality
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
    try:
        z_l = float(values["lens_redshift"])
    except (TypeError, ValueError, OverflowError) as err:
        raise TypeError("lens_redshift must realize to a scalar numeric value.") from err
    try:
        z_s = float(values["source_redshift"])
    except (TypeError, ValueError, OverflowError) as err:
        raise TypeError("source_redshift must realize to a scalar numeric value.") from err
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
    TypeError
        If a realized redshift cannot be converted to a scalar float.
    ValueError
        If redshifts are invalid or ``lens_model`` is not exposed by Caustics.

    Notes
    -----
    A fresh lens is constructed for every realization and is never cached on a
    LightCurveLynx node.
    """
    caustics, torch = _import_caustics_dependencies()
    z_l, z_s = _validate_lens_redshifts(values)

    try:
        lens_class = _LENS_MODEL_REGISTRY[lens_model].lens_class(caustics)
    except KeyError as err:
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
    """
    try:
        x0 = float(values.get("lens_x0", 0.0))
    except (TypeError, ValueError, OverflowError) as err:
        raise TypeError("lens_x0 must realize to a scalar numeric value in arcseconds.") from err
    try:
        y0 = float(values.get("lens_y0", 0.0))
    except (TypeError, ValueError, OverflowError) as err:
        raise TypeError("lens_y0 must realize to a scalar numeric value in arcseconds.") from err
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

    def reference_num_images(self, values):
        """Return the one-image reference count for an atomic lens."""
        return 1


@dataclass(frozen=True)
class _AffinePerturbationGeometryAdapter(_GeometryAdapter):
    """Geometry capabilities for an affine lens-plane perturbation."""

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
        """Return ordered generators capped by their nearest peer center."""
        owners = tuple(
            (
                component,
                component.adapter.search_center(values[component.name]),
            )
            for component in self.components
            if not component.affine
        )
        generators = []
        for owner_index, (component, owner_center) in enumerate(owners):
            for generator in component.adapter.pseudo_caustic_generators(values[component.name]):
                if len(owners) > 1:
                    nearest_distance = min(
                        hypot(
                            owner_center[0] - other_center[0],
                            owner_center[1] - other_center[1],
                        )
                        for other_index, (_, other_center) in enumerate(owners)
                        if other_index != owner_index
                    )
                    generator = _PseudoCausticGenerator(
                        center=generator.center,
                        max_initial_radius=(_PSEUDO_CAUSTIC_SEPARATION_FRACTION * nearest_distance),
                    )
                generators.append(generator)
        return tuple(generators)


def _caustics_scalar(value):
    """Convert one scalar Caustics tensor to an immutable Python float."""
    return float(np.asarray(_to_numpy(value)).item())


def _sis_geometry_adapter(lens, values):
    center = (
        _caustics_scalar(lens.x0.value),
        _caustics_scalar(lens.y0.value),
    )
    einstein_radius = _caustics_scalar(lens.Rein.value)
    unsoftened = float(values["s"]) == 0.0
    return _PointSingularityGeometryAdapter(
        center=center,
        resolution=einstein_radius,
        extent=2.0 * _INITIAL_FOV_PADDING * einstein_radius,
        mask_center=unsoftened,
        recover_center=unsoftened,
        generate_pseudo_caustic=unsoftened,
    )


def _sie_geometry_adapter(lens, values):
    center = (
        _caustics_scalar(lens.x0.value),
        _caustics_scalar(lens.y0.value),
    )
    einstein_radius = _caustics_scalar(lens.Rein.value)
    axis_ratio = _caustics_scalar(lens.q.value)
    unsoftened = float(values["s"]) == 0.0
    return _PointSingularityGeometryAdapter(
        center=center,
        resolution=einstein_radius,
        extent=(2.0 * _INITIAL_FOV_PADDING * einstein_radius / np.sqrt(axis_ratio)),
        mask_center=unsoftened,
        recover_center=unsoftened,
        generate_pseudo_caustic=unsoftened,
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
    unsoftened = float(values["s"]) == 0.0
    return _SmoothCuspGeometryAdapter(
        center=center,
        resolution=angular_scale,
        extent=2.0 * _INITIAL_FOV_PADDING * angular_scale,
        mask_center=unsoftened,
        recover_center=unsoftened,
    )


def _tnfw_geometry_adapter(lens, values):
    center = (
        _caustics_scalar(lens.x0.value),
        _caustics_scalar(lens.y0.value),
    )
    scale_radius = _caustics_scalar(lens.Rs.value)
    truncation = _caustics_scalar(lens.tau.value)
    unsoftened = float(values["s"]) == 0.0
    return _PointSingularityGeometryAdapter(
        center=center,
        resolution=scale_radius,
        extent=(2.0 * _INITIAL_FOV_PADDING * truncation * scale_radius),
        mask_center=unsoftened,
        recover_center=unsoftened,
        generate_pseudo_caustic=not unsoftened,
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
    )


def _external_shear_geometry_adapter(lens, values):
    return _AffinePerturbationGeometryAdapter()


def _mass_sheet_geometry_adapter(lens, values):
    return _AffinePerturbationGeometryAdapter()


_LENS_MODEL_REGISTRY = {
    "SIS": _RegisteredLensModel(_LENS_MODEL_SCHEMAS["SIS"], lambda c: c.SIS, _sis_geometry_adapter),
    "SIE": _RegisteredLensModel(_LENS_MODEL_SCHEMAS["SIE"], lambda c: c.SIE, _sie_geometry_adapter),
    "EPL": _RegisteredLensModel(_LENS_MODEL_SCHEMAS["EPL"], lambda c: c.EPL, _epl_geometry_adapter),
    "NFW": _RegisteredLensModel(_LENS_MODEL_SCHEMAS["NFW"], lambda c: c.NFW, _nfw_geometry_adapter),
    "TNFW": _RegisteredLensModel(_LENS_MODEL_SCHEMAS["TNFW"], lambda c: c.TNFW, _tnfw_geometry_adapter),
    "PseudoJaffe": _RegisteredLensModel(
        _LENS_MODEL_SCHEMAS["PseudoJaffe"], lambda c: c.PseudoJaffe, _pseudo_jaffe_geometry_adapter
    ),
    "ExternalShear": _RegisteredLensModel(
        _LENS_MODEL_SCHEMAS["ExternalShear"], lambda c: c.ExternalShear, _external_shear_geometry_adapter
    ),
    "MassSheet": _RegisteredLensModel(
        _LENS_MODEL_SCHEMAS["MassSheet"], lambda c: c.MassSheet, _mass_sheet_geometry_adapter
    ),
}


def _get_lens_geometry_adapter(lens_model):
    """Return the registered realized-adapter factory for a lens model."""
    return _LENS_MODEL_REGISTRY[lens_model].geometry_factory


@dataclass(frozen=True)
class _CompiledLensComponent:
    name: str
    registry: _RegisteredLensModel
    parameter_fields: tuple[str, ...]
    parameter_graph_names: Mapping[str, str]
    options: Mapping[str, object]


@dataclass(frozen=True)
class _CompiledLensSpec:
    components: tuple[_CompiledLensComponent, ...]
    graph_inputs: tuple[tuple[str, object], ...]
    is_single_plane: bool


@dataclass(frozen=True)
class _RealizedLensComponent:
    name: str
    registry: _RegisteredLensModel
    values: Mapping[str, object]
    lens: object
    geometry_adapter: _GeometryAdapter


@dataclass(frozen=True)
class _RealizedLensSystem:
    lens: object
    geometry_adapter: _GeometryAdapter
    values: Mapping[str, object]
    components: tuple[_RealizedLensComponent, ...]


def _component_name_namespace(caustics):
    """Return the live Caskade and SinglePlane attribute namespace."""
    import caskade

    single_plane = caustics.SinglePlane(
        cosmology=caustics.FlatLambdaCDM(name="namespace_cosmology"),
        lenses=(),
        name="namespace_single_plane",
    )
    node_tuple = caskade.NodeTuple((), name="namespace_tuple")
    namespace = set()
    for node_class, node in (
        (caskade.Node, caskade.Node(name="namespace_node")),
        (caskade.NodeCollection, caskade.NodeCollection(name="namespace_collection")),
        (caskade.NodeTuple, node_tuple),
        (caustics.SinglePlane, single_plane),
    ):
        namespace.update(dir(node))
        namespace.update(dir(node_class))
        namespace.update(name for name in inspect.signature(node_class).parameters if hasattr(node, name))
    return namespace


def _compile_lens_spec(lens, *, reserved_parameter_names=()):
    """Compile an immutable lens specification into deterministic graph metadata."""
    is_single_plane = isinstance(lens, CausticsSinglePlaneSpec)
    if is_single_plane:
        caustics, _ = _import_caustics_dependencies()
        component_namespace = _component_name_namespace(caustics)
        component_specs = tuple(lens.components.items())
        for name, _ in component_specs:
            if not name.isidentifier() or keyword.iskeyword(name):
                raise ValueError(f"Component name {name!r} must be a non-keyword Python identifier.")
            if name in component_namespace:
                raise ValueError(f"Component name {name!r} collides with the Caskade/SinglePlane namespace.")
    else:
        component_specs = (("lens", lens),)

    flattened_names = set(reserved_parameter_names)
    compiled_components = []
    graph_inputs = []
    for component_name, component_spec in component_specs:
        registry = _LENS_MODEL_REGISTRY[component_spec.model]
        parameter_fields = tuple(
            field_name for field_name in registry.schema.fields if field_name in component_spec.parameters
        )
        parameter_graph_names = {}
        for field_name in parameter_fields:
            if is_single_plane:
                graph_name = f"lens_{component_name}_{field_name}"
            else:
                graph_name = f"lens_{field_name}"
            if graph_name in flattened_names:
                raise ValueError(f"Duplicate or reserved flattened lens parameter name {graph_name!r}.")
            flattened_names.add(graph_name)
            parameter_graph_names[field_name] = graph_name
            graph_inputs.append((graph_name, component_spec.parameters[field_name]))
        compiled_components.append(
            _CompiledLensComponent(
                name=component_name,
                registry=registry,
                parameter_fields=parameter_fields,
                parameter_graph_names=MappingProxyType(parameter_graph_names),
                options=MappingProxyType(dict(component_spec.options)),
            )
        )

    if all(component.registry.schema.affine for component in compiled_components):
        raise ValueError("A lens specification must contain at least one non-affine component.")
    return _CompiledLensSpec(
        components=tuple(compiled_components),
        graph_inputs=tuple(graph_inputs),
        is_single_plane=is_single_plane,
    )


def _caustics_lens_kwargs(registry, values, torch):
    """Convert registered numeric fields while preserving configuration literals."""
    kwargs = {}
    for name, field in registry.schema.fields.items():
        if name not in values:
            continue
        value = values[name]
        if field.kind == "positive_int":
            value = int(value)
        elif field.kind == "chunk_size" and value is not None:
            value = int(value)
        elif field.kind == "bool":
            value = bool(value)
        elif field.kind != "selector" and value is not None:
            value = torch.as_tensor(value, dtype=torch.float64)
        kwargs[name] = value
    return kwargs


def _construct_registered_lens(
    component,
    component_values,
    *,
    caustics,
    torch,
    cosmology,
    z_l,
    z_s,
    name,
):
    """Construct one fresh explicitly registered Caustics lens component."""
    try:
        lens_class = component.registry.lens_class(caustics)
        lens_kwargs = _caustics_lens_kwargs(component.registry, component_values, torch)
        return lens_class(
            cosmology=cosmology,
            z_l=z_l,
            z_s=z_s,
            name=name,
            **lens_kwargs,
        )
    except Exception as err:
        raise ValueError(
            f"Failed to construct Caustics component {component.name!r} "
            f"with model {component.registry.schema.model!r}."
        ) from err


def _build_lens_system(compiled, *, cosmology, values):
    """Realize one fresh atomic or single-plane Caustics lens system."""
    z_l_value, z_s_value = _validate_lens_redshifts(values)
    caustics, torch = _import_caustics_dependencies()
    z_l = torch.as_tensor(z_l_value, dtype=torch.float64)
    z_s = torch.as_tensor(z_s_value, dtype=torch.float64)

    realized_values = []
    for component in compiled.components:
        parameter_values = {
            field_name: values[component.parameter_graph_names[field_name]]
            for field_name in component.parameter_fields
        }
        realized_values.append(
            _realize_component_values(
                component.registry.schema,
                parameter_values,
                component.options,
            )
        )

    non_affine = tuple(
        (component, component_values)
        for component, component_values in zip(compiled.components, realized_values, strict=True)
        if not component.registry.schema.affine
    )
    for first_index, (first, first_values) in enumerate(non_affine):
        for second, second_values in non_affine[first_index + 1 :]:
            if first_values["x0"] == second_values["x0"] and first_values["y0"] == second_values["y0"]:
                raise ValueError(
                    f"Non-affine lens components {first.name!r} and {second.name!r} "
                    "have exactly coincident centers."
                )

    if not compiled.is_single_plane:
        component = compiled.components[0]
        component_values = realized_values[0]
        lens = _construct_registered_lens(
            component,
            component_values,
            caustics=caustics,
            torch=torch,
            cosmology=cosmology,
            z_l=z_l,
            z_s=z_s,
            name="lens",
        )
        geometry_adapter = component.registry.geometry_factory(lens, component_values)
        realized_component = _RealizedLensComponent(
            name=component.name,
            registry=component.registry,
            values=component_values,
            lens=lens,
            geometry_adapter=geometry_adapter,
        )
        return _RealizedLensSystem(
            lens=lens,
            geometry_adapter=geometry_adapter,
            values=component_values,
            components=(realized_component,),
        )

    children = tuple(
        _construct_registered_lens(
            component,
            component_values,
            caustics=caustics,
            torch=torch,
            cosmology=cosmology,
            z_l=None,
            z_s=None,
            name=component.name,
        )
        for component, component_values in zip(compiled.components, realized_values, strict=True)
    )
    lens = caustics.SinglePlane(
        cosmology=cosmology,
        lenses=children,
        name="lens_plane",
        z_l=z_l,
        z_s=z_s,
    )
    realized_components = tuple(
        _RealizedLensComponent(
            name=component.name,
            registry=component.registry,
            values=component_values,
            lens=child,
            geometry_adapter=component.registry.geometry_factory(child, component_values),
        )
        for component, component_values, child in zip(
            compiled.components,
            realized_values,
            children,
            strict=True,
        )
    )
    geometry_adapter = _SinglePlaneGeometryAdapter(
        tuple(
            _GeometryComponent(
                name=component.name,
                adapter=component.geometry_adapter,
                affine=component.registry.schema.affine,
            )
            for component in realized_components
        )
    )
    system_values = MappingProxyType({component.name: component.values for component in realized_components})
    return _RealizedLensSystem(
        lens=lens,
        geometry_adapter=geometry_adapter,
        values=system_values,
        components=realized_components,
    )


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
    physical singularities: non-finite determinant samples and a one-spacing
    neighborhood around each singular point are deliberately masked as part of
    critical-curve extraction. The post-mask grid and outer-boundary
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
    for singular_x, singular_y in singular_points:
        squared_distance = (x_coordinates[np.newaxis, :] - singular_x) ** 2 + (
            y_coordinates[:, np.newaxis] - singular_y
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


def _boundary_regions(curves, *, geometry_tolerance):
    """Convert typed source-boundary curves to polygonal regions.

    Parameters
    ----------
    curves : iterable of array-like
        Closed source-plane curves, each with shape ``(P, 2)`` in
        arcseconds.
    geometry_tolerance : float
        Consecutive-vertex curve-normalization tolerance in arcseconds.

    Returns
    -------
    tuple of shapely.Polygon or shapely.MultiPolygon
        Repaired finite positive-area polygonal region corresponding to each
        input curve, in input order.

    Raises
    ------
    ImportError
        If the optional Shapely dependency is unavailable.
    RuntimeError
        If curve normalization collapses a boundary, validity repair leaves
        non-polygonal geometry, or a repaired region is empty, non-finite, or
        has non-positive area.

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
        region = shapely.make_valid(shapely.Polygon(coordinates), method='structure')
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

    Raises
    ------
    ImportError
        If the optional Shapely dependency is unavailable.
    RuntimeError
        If normalization collapses a curve below three unique vertices.
    """
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
        If a boundary cannot be normalized into positive-area polygonal
        geometry.
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
        Whether typed curve counts and the complete topology signature match.

    Notes
    -----
    Matching never crosses true/pseudo boundary types. Reordered arrays remain
    realization-local and are not cached on a node or across snapshots.
    """
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
    shapely.Polygon or shapely.MultiPolygon
        Valid, finite, positive-area strong-lensing geometry. Disconnected
        components, concavities, and holes are retained.

    Raises
    ------
    ImportError
        If Shapely is unavailable.
    RuntimeError
        If normalization or validity repair cannot retain polygonal geometry,
        or precision snapping, union, and final repair produce an empty,
        non-finite, or non-positive-area result.

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
    region = shapely.make_valid(region, method='structure')
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
    lens_model : str
        Non-empty top-level Caustics lens-class name with a registered complete
        geometry adapter.
    lens_parameters : Mapping[str, object]
        Caustics constructor parameter names mapped to graph setters.
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
        If the lens configuration has the wrong type, a lens-parameter key is
        not a string, or a normalized scalar cannot be converted to float.
    ValueError
        If a reserved parameter name is used, no complete adapter is
        registered, an integer setting is outside its ordinary type/range
        contract, a scalar is non-finite or outside its range, or the FOV,
        pixel-scale, and tolerance relations are inconsistent.
    """
    _validate_lens_configuration(lens_model, lens_parameters)
    _ = _get_lens_geometry_adapter(lens_model)

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
    lens_parameters : Mapping[str, object]
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
        Maximum number of FOV expansions after the initial attempt.
    fov_expansion_factor : float, optional
        Finite multiplier greater than one applied at each FOV expansion.
        Larger values can increase two-dimensional grid cost rapidly.
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
        Graph output for maximum matched-boundary displacement in arcseconds.
    source_boundary_clearance : AttributeIndicator
        Graph output for nearest typed-boundary distance in arcseconds.
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

    A registered adapter is resolved once per realization, threaded through
    certification, and never cached on the node. Results are saved in
    ``GraphState``; a caller RNG takes precedence over the seeded fallback,
    and sample-local sub-seeds isolate variable rejection counts. Persisted
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
        """Configure geometric source-position sampling.

        Parameters
        ----------
        lens_model : str
            Non-empty Caustics lens-class name with a registered complete
            geometry adapter.
        cosmology : caustics.Cosmology
            Fixed cosmology supplied to every realized lens.
        lens_redshift : object
            Graph setter for dimensionless lens redshift.
        source_redshift : object
            Graph setter for dimensionless source redshift.
        lens_parameters : Mapping[str, object]
            String Caustics parameter names mapped to graph setters.
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
            Positive initial singular-loop radius in arcseconds.
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
            If lens configuration, parameter keys, fractions, or normalized
            scalar settings have invalid types.
        ValueError
            If the lens is unsupported for source sampling, a setting is
            outside its range, or static FOV/scale/tolerance relations fail.

        Notes
        -----
        Redshifts and every lens parameter are registered as independent graph
        inputs, and all nine public outputs are registered in ``_OUTPUTS``
        order. Construction validates adapter support but stores no adapter.
        The seeded node-owned generator is used only when ``compute`` receives
        no caller generator.
        """
        pixelscale_fraction = _validate_optional_positive_fraction(
            "pixelscale_fraction",
            pixelscale_fraction,
        )
        normalized = _validate_source_position_configuration(
            lens_model=lens_model,
            lens_parameters=lens_parameters,
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

        self.lens_model = lens_model
        self.cosmology = cosmology
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

    def _realized_pixelscale_for_one_lens(self, geometry_adapter, values):
        """Realize the initial requested grid-spacing upper bound.

        Parameters
        ----------
        geometry_adapter : _PointSingularityGeometryAdapter
            Stable registered adapter for this lens realization.
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
        ``pixelscale_fraction * characteristic_angular_scale(values)``; the
        adapter's positive finite scale postcondition is trusted.
        """
        if self.pixelscale_fraction is None:
            return self.pixelscale
        characteristic_scale = geometry_adapter.characteristic_angular_scale(values)
        return min(
            self.pixelscale,
            self.pixelscale_fraction * characteristic_scale,
        )

    def _initial_fov_for_one_lens(self, geometry_adapter, values, *, pixelscale):
        """Realize the starting critical-curve search FOV.

        Parameters
        ----------
        geometry_adapter : _PointSingularityGeometryAdapter
            Stable registered adapter for this lens realization.
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
        """Extract complete caustics with bounded sample-local FOV recovery.

        Parameters
        ----------
        lens : object
            Realized Caustics lens.
        geometry_adapter : _PointSingularityGeometryAdapter
            Stable registered adapter for this realization.
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
        """Extract one typed-boundary snapshot for a realized lens.

        Parameters
        ----------
        lens : object
            Realized Caustics lens.
        geometry_adapter : _PointSingularityGeometryAdapter
            Adapter already resolved for the complete realization.
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
            Snapshot containing separate true and directly adapter-produced
            pseudo boundaries, the successful FOV, requested scale upper bound,
            and pseudo-caustic resolution.

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
        pseudo_caustic_curves = geometry_adapter.pseudo_caustics(
            lens,
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
        geometry_adapter : _PointSingularityGeometryAdapter
            Stable registered adapter for the complete realization.
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
        of pseudo-caustic vertices. Acceptance requires stable typed topology
        and displacement no greater than ``boundary_tolerance``.
        """
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
        geometry_adapter : _PointSingularityGeometryAdapter
            Registry adapter held stable for this realization.
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
        This method constructs one lens, resolves exactly one adapter for the
        entire realization, threads it through certification, and never caches
        it on the node.
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
                values=values,
                caustic_curves=previous_geometry.caustic_curves,
                pseudo_caustic_curves=previous_geometry.pseudo_caustic_curves,
            )
            final_count = geometry_adapter.expected_num_images(
                source_x[sample_index],
                source_y[sample_index],
                values=values,
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

    Parameters
    ----------
    lens_model : str
        Non-empty name of a top-level Caustics lens class.
    cosmology : caustics.Cosmology
        Fixed cosmology supplied to every realized lens.
    lens_redshift : object
        Graph setter for dimensionless lens redshift.
    source_redshift : object
        Graph setter for dimensionless source redshift.
    source_x : object
        Graph setter realizing to a finite source-plane x coordinate in
        arcseconds.
    source_y : object
        Graph setter realizing to a finite source-plane y coordinate in
        arcseconds.
    lens_parameters : Mapping[str, object]
        String Caustics constructor parameter names mapped to graph setters.
    max_images : int or numpy.integer
        Fixed output width and maximum accepted image count; at least two.
    min_images : int or numpy.integer, optional
        Minimum count accepted after bounded recovery, between one and
        ``max_images``.
    expected_num_images : object or None, optional
        Graph setter realizing to ``None`` or an integer between
        ``min_images`` and ``max_images``.
    fov : object, optional
        Graph setter realizing to a positive finite image-plane FOV in
        arcseconds.
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
        executed singular-seed batch.
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

    The physical lens center is distinct from the numerical grid center. A
    half-cell recovery shift changes only the numerical search grid and never
    translates returned physical image coordinates. Lens models without a
    registered geometry adapter retain absolute pixel/epsilon settings and
    skip targeted singular recovery. No adapter is cached on the node.

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
        """Configure deterministic point-image solving and bounded recovery.

        Parameters
        ----------
        lens_model : str
            Non-empty top-level Caustics lens-class name.
        cosmology : caustics.Cosmology
            Fixed cosmology supplied to each realized lens.
        lens_redshift : object
            Graph setter for dimensionless lens redshift.
        source_redshift : object
            Graph setter for dimensionless source redshift.
        source_x : object
            Graph setter for source-plane x position in arcseconds.
        source_y : object
            Graph setter for source-plane y position in arcseconds.
        lens_parameters : Mapping[str, object]
            String Caustics parameter names mapped to graph setters.
        max_images : int or numpy.integer
            Fixed output width and maximum active count; at least two.
        min_images : int or numpy.integer, optional
            Minimum acceptable count, from one through ``max_images``.
        expected_num_images : object or None, optional
            Graph setter for an optional realized expected count.
        fov : object, optional
            Graph setter realizing to a positive finite image-plane FOV in
            arcseconds.
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
            If lens configuration, parameter keys, fractions, or normalized
            scalar settings have invalid types.
        ValueError
            If a reserved lens parameter is used, or counts, depths, recovery
            limits, factors, or positive scalar settings violate their ranges
            or relations.

        Notes
        -----
        Redshifts, source coordinates, FOV, optional expected count, and every
        lens parameter are registered as graph inputs. All eleven public
        outputs are registered in ``_OUTPUTS`` order. Geometry adapters are
        resolved per realization and are never stored on the node.
        """
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

        self.lens_model = lens_model
        self.cosmology = cosmology
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
        """Realize per-lens grid-scale and residual-tolerance settings.

        Parameters
        ----------
        geometry_adapter : _PointSingularityGeometryAdapter or None
            Registry adapter held stable for this realization, or ``None``
            for an unsupported geometry model.
        values : Mapping[str, object]
            Realized inputs for one lens system.

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
        setting becomes the smaller of its absolute and relative value. A
        missing adapter retains both absolute settings.
        """
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
            dimensionless ``lens_redshift`` and ``source_redshift``; finite
            scalar source-plane ``source_x`` and ``source_y`` in arcseconds;
            positive finite realized ``fov`` in arcseconds; required
            ``expected_num_images``, whose value may be ``None`` or an
            ordinary integer in the configured range; and
            ``lens_<parameter>`` for every configured Caustics constructor
            parameter.

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

        The configured FOV setter realizes per lens and is multiplied by
        ``fov_multiplier`` to form ``initial_fov``. Realized pixel scale
        and epsilon are fixed once. Every global call and residual
        certification uses that fixed epsilon; targeted seed radius is
        ``min(realized_epsilon, actual_grid_spacing)``, while singular
        neighborhood occupancy and root locality use the actual grid spacing.
        Each outer attempt is independent, so its global coordinates replace
        rather than merge with any earlier attempt. Targeted roots are local to
        a successful, nonempty, deficient attempt and are skipped without an
        adapter.

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
            selected lens model violate their owned domains.
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

        try:
            realized_fov = float(values["fov"])
        except (TypeError, ValueError, OverflowError) as err:
            raise TypeError("fov must realize to a scalar numeric value.") from err
        if not np.isfinite(realized_fov) or realized_fov <= 0.0:
            raise ValueError("fov must realize to a positive finite value.")
        initial_fov = realized_fov * self.fov_multiplier
        geometry_adapter = _LENS_MODEL_REGISTRY[self.lens_model].geometry_factory
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
                not isinstance(expected_num_images, (int, np.integer))
                or not self.min_images <= expected_num_images <= self.max_images
            ):
                raise ValueError(
                    "expected_num_images must be None or an integer between min_images and max_images."
                )
            expected_num_images = int(expected_num_images)

        lens, torch = _construct_caustics_lens(
            lens_model=self.lens_model,
            cosmology=self.cosmology,
            values=values,
            lens_parameter_names=self._lens_parameter_names,
        )

        beta_x = torch.as_tensor(source_x, dtype=torch.float64)
        beta_y = torch.as_tensor(source_y, dtype=torch.float64)
        center_x, center_y = _lens_plane_origin(values)
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
            ``solver_attempts``. Each executed singular-seed batch adds one
            more attempt, while its internal root-refinement passes do not.
            Actual spacing is updated to ``current_fov / divisions`` for the
            invoked variant. The divisions-plus-one variant changes that
            spacing; the half-cell numerical-center shift uses base divisions
            and therefore retains base spacing. Both add a global attempt, and
            neither changes the outer expansion/refinement counters.

            Targeted recovery runs only after a successful, nonempty, deficient
            global result with a registered adapter. It selects one seed for
            each empty singular neighborhood. Its radius is the smaller of
            fixed realized epsilon and actual grid spacing. Residual
            certification uses fixed epsilon, while neighborhood occupancy and
            root locality use actual grid spacing. Retryable targeted failures
            leave the global result deficient. Targeted roots and global
            coordinates are local to this attempt; every later outer call
            replaces them.

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

            # TODO: Instead of geometry_adapter is None check, check if adapter is singular.
            # Relevant when we generalize geometry adapters to none singular models beyond
            # SIS and SIE.
            if complete or not len(coordinates) or geometry_adapter is None:
                return coordinates, complete

            singular_points = np.asarray(
                geometry_adapter.singular_points(values),
                dtype=float,
            ).reshape(-1, 2)
            empty_neighborhoods = _recovery_neighborhoods_are_empty(
                coordinates,
                singular_points,
                current_grid_pixelscale,
            )
            if not np.any(empty_neighborhoods):
                return coordinates, False

            seeds = geometry_adapter.singular_image_seeds(
                lens,
                values,
                source_x=source_x,
                source_y=source_y,
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
                is_singular_error = _is_singular_forward_raytrace_error(error)
                is_empty_candidate_error = _is_retryable_forward_raytrace_error(error)
                if not (is_singular_error or is_empty_candidate_error):
                    raise
                latest_retryable_error = f"{type(error).__name__}: {error}"
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
            9. ``solver_attempts``, global calls plus executed singular-seed
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

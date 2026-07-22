"""Private Caustics lens-system validation and realization helpers."""

import inspect
from collections.abc import Mapping
from dataclasses import dataclass
from inspect import Parameter
from math import hypot
from types import MappingProxyType

import numpy as np

from lightcurvelynx.models._caustics import runtime as _runtime

_REALIZATION_SUPPLIED_LENS_PARAMETERS = {"cosmology", "z_l", "z_s"}
_ROOT_LENS_PARAMETERS = {"z_l"}


def _is_required_parameter(parameter):
    """Determine whether an inspected constructor parameter is required.

    Parameters
    ----------
    parameter : inspect.Parameter
        Parameter from a registered Caustics constructor signature.

    Returns
    -------
    bool
        ``True`` when the parameter has no default and is not a variadic
        positional or keyword parameter.
    """
    return parameter.default is Parameter.empty and parameter.kind not in (
        Parameter.VAR_POSITIONAL,
        Parameter.VAR_KEYWORD,
    )


def _get_model_class(model):
    """Load one explicitly registered top-level Caustics lens class.

    Parameters
    ----------
    model : str
        Registry key naming one of the nine supported lens models.

    Returns
    -------
    type
        Corresponding class exported by the installed ``caustics`` package.

    Raises
    ------
    TypeError
        If ``model`` is unhashable during registry membership testing.
    ValueError
        If ``model`` is not a registered key.
    ImportError
        If Caustics or one of its import-time dependencies is unavailable.
    AttributeError
        If a registered name is unexpectedly absent from the installed
        Caustics module.

    Notes
    -----
    Registry membership is checked before the optional package is imported.
    """
    if model not in _LENS_MODEL_REGISTRY:
        raise ValueError(f"Requested model {model!r} is not supported currently.")
    caustics = _runtime._import_caustics()
    return getattr(caustics, model)


def _validated_constructor_parameters(model, parameters):
    """Validate and snapshot parameters for one registered lens constructor.

    Parameters
    ----------
    model : str
        Explicit registry key naming a supported Caustics lens model.
    parameters : Mapping[str, object]
        Constructor-name-to-graph-setter mapping.

    Returns
    -------
    dict
        Shallow parameter snapshot validated against the selected constructor.
        A SinglePlane lens sequence is normalized to a tuple.

    Raises
    ------
    TypeError
        If parameters is not a mapping or a key is not a string.
    ValueError
        If a node-owned parameter is supplied, a required argument is missing,
        or an explicit name is unsupported by the selected constructor.
    """
    if not isinstance(parameters, Mapping):
        raise TypeError("parameters must be a mapping.")
    parameter_snapshot = dict(parameters)
    if any(not isinstance(name, str) for name in parameter_snapshot):
        raise TypeError("parameters keys must be strings.")
    if "cosmology" in parameter_snapshot:
        raise ValueError("cosmology is supplied by the consuming node.")
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
            and name not in _REALIZATION_SUPPLIED_LENS_PARAMETERS
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
        parameter_snapshot["lenses"] = tuple(parameter_snapshot["lenses"])
    return parameter_snapshot


_INITIAL_FOV_PADDING = 1.1


_PSEUDO_CAUSTIC_SEPARATION_FRACTION = 0.25


@dataclass(frozen=True)
class _PseudoCausticGenerator:
    """Describe one component-owned pseudo-caustic loop generator.

    Attributes
    ----------
    center : tuple of float
        Image-plane ``(x, y)`` center in arcseconds.
    max_initial_radius : float or None
        Optional upper bound on the first image-plane loop radius in
        arcseconds. ``None`` leaves the caller's configured radius uncapped.

    Notes
    -----
    ``frozen=True`` prevents field reassignment but does not copy or deeply
    freeze supplied objects. The documented tuple-of-floats ``center`` contract
    is therefore caller-owned, and both fields are retained by identity.
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

    Subclasses implement each capability method. Calling an unimplemented base
    capability raises ``NotImplementedError``; ``winding_number`` and
    ``expected_num_images`` provide shared concrete algorithms.
    """

    def search_center(self, values):
        """Return the numerical image-plane search center.

        Parameters
        ----------
        values : Mapping[str, object]
            Realized adapter-owned values for one lens system.

        Returns
        -------
        x, y : tuple of float
            Image-plane center in arcseconds.

        Raises
        ------
        NotImplementedError
            Always raised by this base capability; concrete adapters must
            override it.
        """
        raise NotImplementedError

    def initial_fov(self, values):
        """Return the full initial image-plane search extent.

        Parameters
        ----------
        values : Mapping[str, object]
            Realized adapter-owned values for one lens system.

        Returns
        -------
        float or None
            Full square-search width in arcseconds, or ``None`` when the
            component has no independent finite extent.

        Raises
        ------
        NotImplementedError
            Always raised by this base capability; concrete adapters must
            override it.
        """
        raise NotImplementedError

    def resolution_scale(self, values):
        """Return the characteristic image-plane resolution scale.

        Parameters
        ----------
        values : Mapping[str, object]
            Realized adapter-owned values for one lens system.

        Returns
        -------
        float or None
            Characteristic angular scale in arcseconds, or ``None`` when the
            component supplies no independent scale.

        Raises
        ------
        NotImplementedError
            Always raised by this base capability; concrete adapters must
            override it.
        """
        raise NotImplementedError

    def jacobian_mask_points(self, values):
        """Return points to mask during lens-Jacobian contour extraction.

        Parameters
        ----------
        values : Mapping[str, object]
            Realized adapter-owned values for one lens system.

        Returns
        -------
        tuple of tuple of float
            Image-plane ``(x, y)`` points in arcseconds. An empty tuple means
            no explicit Jacobian masking is required.

        Raises
        ------
        NotImplementedError
            Always raised by this base capability; concrete adapters must
            override it.
        """
        raise NotImplementedError

    def root_recovery_points(self, values):
        """Return points requiring targeted image-root recovery.

        Parameters
        ----------
        values : Mapping[str, object]
            Realized adapter-owned values for one lens system.

        Returns
        -------
        tuple of tuple of float
            Image-plane ``(x, y)`` points in arcseconds. An empty tuple disables
            targeted recovery.

        Raises
        ------
        NotImplementedError
            Always raised by this base capability; concrete adapters must
            override it.
        """
        raise NotImplementedError

    def pseudo_caustic_generators(self, values):
        """Return image-plane loop generators for pseudo-caustic boundaries.

        Parameters
        ----------
        values : Mapping[str, object]
            Realized adapter-owned values for one lens system.

        Returns
        -------
        tuple of _PseudoCausticGenerator
            Ordered component-owned generators. An empty tuple means the lens
            contributes no pseudo-caustic boundary.

        Raises
        ------
        NotImplementedError
            Always raised by this base capability; concrete adapters must
            override it.
        """
        raise NotImplementedError

    def axisymmetry_center(self, values):
        """Return an exact rotational-symmetry center when certified.

        Parameters
        ----------
        values : Mapping[str, object]
            Realized adapter-owned values for one lens system.

        Returns
        -------
        tuple of float or None
            Exact image-plane ``(x, y)`` center in arcseconds, or ``None`` when
            the total geometry is not certified as axisymmetric.

        Raises
        ------
        NotImplementedError
            Always raised by this base capability; concrete adapters must
            override it.
        """
        raise NotImplementedError

    def preserves_axisymmetry(self, values):
        """Return whether this component preserves rotational symmetry.

        Parameters
        ----------
        values : Mapping[str, object]
            Realized adapter-owned values for one lens system.

        Returns
        -------
        bool
            Whether this component can preserve a peer's exact axisymmetry.

        Raises
        ------
        NotImplementedError
            Always raised by this base capability; concrete adapters must
            override it.
        """
        raise NotImplementedError

    @staticmethod
    def winding_number(curve, x, y):
        """Compute a curve's signed winding around a source-plane point.

        Parameters
        ----------
        curve : array-like, shape (P, 2)
            Ordered source-plane ``(x, y)`` vertices in arcseconds.
        x : float
            Source-plane evaluation x coordinate in arcseconds.
        y : float
            Source-plane evaluation y coordinate in arcseconds.

        Returns
        -------
        int
            Signed integer winding obtained by rounding the accumulated
            oriented angle in turns.
        """
        point = np.array([x, y], dtype=float)
        vectors = np.asarray(curve, dtype=float) - point
        following = np.roll(vectors, -1, axis=0)
        angles = np.arctan2(
            vectors[:, 0] * following[:, 1] - vectors[:, 1] * following[:, 0],
            np.sum(vectors * following, axis=1),
        )
        return int(round(float(np.sum(angles)) / (2.0 * np.pi)))

    def reference_num_images(self, values):
        """Return the regular-image count outside all typed boundaries.

        Parameters
        ----------
        values : Mapping[str, object]
            Realized adapter-owned values for one lens system.

        Returns
        -------
        int
            Reference number of regular images.

        Raises
        ------
        NotImplementedError
            Always raised by this base capability; concrete adapters must
            override it.
        """
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
        """Return the signed-boundary regular-image count.

        Parameters
        ----------
        source_x : float
            Source-plane x position in arcseconds.
        source_y : float
            Source-plane y position in arcseconds.
        values : Mapping[str, object]
            Realized adapter-owned values for one lens system.
        caustic_curves : iterable of array-like
            Ordered true-caustic curves with shape ``(P, 2)`` in source-plane
            arcseconds.
        pseudo_caustic_curves : iterable of array-like
            Ordered pseudo-caustic curves with shape ``(P, 2)`` in source-plane
            arcseconds.

        Returns
        -------
        int
            Reference count plus twice each signed true-caustic winding and
            once each signed pseudo-caustic winding.
        """
        count = self.reference_num_images(values)
        count += 2 * sum(self.winding_number(curve, source_x, source_y) for curve in caustic_curves)
        count += sum(self.winding_number(curve, source_x, source_y) for curve in pseudo_caustic_curves)
        return count


@dataclass(frozen=True)
class _PointSingularityGeometryAdapter(_GeometryAdapter):
    """Realized finite geometry with independently enabled point capabilities.

    Attributes
    ----------
    center : tuple of float
        Image-plane lens center in arcseconds.
    resolution : float
        Characteristic image-plane resolution in arcseconds.
    extent : float
        Full initial image-plane search extent in arcseconds.
    mask_center : bool
        Whether to return ``center`` as a Jacobian mask point.
    recover_center : bool
        Whether to return ``center`` as a targeted root-recovery point.
    generate_pseudo_caustic : bool
        Whether to return a pseudo-caustic generator centered at ``center``.
    axisymmetric : bool
        Whether this atomic non-affine lens has exact rotational symmetry.

    Notes
    -----
    The adapter is fully realized and ignores its methods' ``values`` mapping.
    ``frozen=True`` prevents field reassignment but does not copy or deeply
    freeze supplied objects; in particular, ``center`` is retained by identity
    under its documented tuple-of-floats contract.
    """

    center: tuple[float, float]
    resolution: float
    extent: float
    mask_center: bool
    recover_center: bool
    generate_pseudo_caustic: bool
    axisymmetric: bool = False

    def search_center(self, values):
        """Implement the base search-center capability with the stored center.

        Parameters
        ----------
        values : Mapping[str, object]
            Ignored because this adapter is fully realized.

        Returns
        -------
        tuple of float, length 2
            Stored image-plane ``(x, y)`` center in arcseconds.
        """
        return self.center

    def initial_fov(self, values):
        """Implement the base initial-FOV capability with the stored extent.

        Parameters
        ----------
        values : Mapping[str, object]
            Ignored because this adapter is fully realized.

        Returns
        -------
        float
            Stored full square image-plane extent in arcseconds; this concrete
            adapter never returns ``None``.
        """
        return self.extent

    def resolution_scale(self, values):
        """Implement the base resolution capability with the stored scale.

        Parameters
        ----------
        values : Mapping[str, object]
            Ignored because this adapter is fully realized.

        Returns
        -------
        float
            Stored image-plane resolution scale in arcseconds; this concrete
            adapter never returns ``None``.
        """
        return self.resolution

    def jacobian_mask_points(self, values):
        """Implement the base Jacobian-mask capability at the stored center.

        Parameters
        ----------
        values : Mapping[str, object]
            Ignored because this adapter is fully realized.

        Returns
        -------
        tuple of tuple of float
            One image-plane ``(x, y)`` point in arcseconds when center masking
            is enabled, otherwise the empty tuple required by the base contract.
        """
        return (self.center,) if self.mask_center else ()

    def root_recovery_points(self, values):
        """Implement the base root-recovery capability at the stored center.

        Parameters
        ----------
        values : Mapping[str, object]
            Ignored because this adapter is fully realized.

        Returns
        -------
        tuple of tuple of float
            One image-plane ``(x, y)`` point in arcseconds when center recovery
            is enabled, otherwise the empty tuple that disables base-contract
            targeted recovery.
        """
        return (self.center,) if self.recover_center else ()

    def pseudo_caustic_generators(self, values):
        """Implement the base pseudo-caustic capability at the stored center.

        Parameters
        ----------
        values : Mapping[str, object]
            Ignored because this adapter is fully realized.

        Returns
        -------
        tuple of _PseudoCausticGenerator
            One generator centered in image-plane arcseconds when enabled,
            otherwise the empty tuple denoting no pseudo-caustic boundary.
        """
        if self.generate_pseudo_caustic:
            return (_PseudoCausticGenerator(self.center),)
        return ()

    def axisymmetry_center(self, values):
        """Implement the base exact-axisymmetry-center capability.

        Parameters
        ----------
        values : Mapping[str, object]
            Ignored because this adapter is fully realized.

        Returns
        -------
        tuple of float, length 2, or None
            Stored image-plane center in arcseconds when exact axisymmetry is
            certified, otherwise ``None`` as required by the base contract.
        """
        return self.center if self.axisymmetric else None

    def preserves_axisymmetry(self, values):
        """Implement the base peer-axisymmetry-preservation capability.

        Parameters
        ----------
        values : Mapping[str, object]
            Ignored because this adapter is fully realized.

        Returns
        -------
        bool
            Always ``False`` because this non-affine adapter cannot serve as an
            affine symmetry-preserving peer.
        """
        return False

    def reference_num_images(self, values):
        """Implement the base regular-image reference-count capability.

        Parameters
        ----------
        values : Mapping[str, object]
            Ignored because this adapter is fully realized.

        Returns
        -------
        int
            Dimensionless one-image reference count for this atomic lens.
        """
        return 1


@dataclass(frozen=True)
class _SmoothCuspGeometryAdapter(_GeometryAdapter):
    """Realized finite geometry for a cusp without a pseudo-caustic.

    Attributes
    ----------
    center : tuple of float
        Image-plane lens center in arcseconds.
    resolution : float
        Characteristic image-plane resolution in arcseconds.
    extent : float
        Full initial image-plane search extent in arcseconds.
    mask_center : bool
        Whether to return ``center`` as a Jacobian mask point.
    recover_center : bool
        Whether to return ``center`` as a targeted root-recovery point.
    axisymmetric : bool
        Whether this atomic non-affine lens has exact rotational symmetry.

    Notes
    -----
    The adapter is fully realized, ignores its methods' ``values`` mapping,
    and always returns an empty pseudo-caustic-generator tuple.
    ``frozen=True`` prevents field reassignment but does not copy or deeply
    freeze supplied objects; in particular, ``center`` is retained by identity
    under its documented tuple-of-floats contract.
    """

    center: tuple[float, float]
    resolution: float
    extent: float
    mask_center: bool
    recover_center: bool
    axisymmetric: bool = False

    def search_center(self, values):
        """Implement the base search-center capability with the stored center.

        Parameters
        ----------
        values : Mapping[str, object]
            Ignored because this adapter is fully realized.

        Returns
        -------
        tuple of float, length 2
            Stored image-plane ``(x, y)`` center in arcseconds.
        """
        return self.center

    def initial_fov(self, values):
        """Implement the base initial-FOV capability with the stored extent.

        Parameters
        ----------
        values : Mapping[str, object]
            Ignored because this adapter is fully realized.

        Returns
        -------
        float
            Stored full square image-plane extent in arcseconds; this concrete
            adapter never returns ``None``.
        """
        return self.extent

    def resolution_scale(self, values):
        """Implement the base resolution capability with the stored scale.

        Parameters
        ----------
        values : Mapping[str, object]
            Ignored because this adapter is fully realized.

        Returns
        -------
        float
            Stored image-plane resolution scale in arcseconds; this concrete
            adapter never returns ``None``.
        """
        return self.resolution

    def jacobian_mask_points(self, values):
        """Implement the base Jacobian-mask capability at the stored center.

        Parameters
        ----------
        values : Mapping[str, object]
            Ignored because this adapter is fully realized.

        Returns
        -------
        tuple of tuple of float
            One image-plane ``(x, y)`` point in arcseconds when center masking
            is enabled, otherwise the empty tuple required by the base contract.
        """
        return (self.center,) if self.mask_center else ()

    def root_recovery_points(self, values):
        """Implement the base root-recovery capability at the stored center.

        Parameters
        ----------
        values : Mapping[str, object]
            Ignored because this adapter is fully realized.

        Returns
        -------
        tuple of tuple of float
            One image-plane ``(x, y)`` point in arcseconds when center recovery
            is enabled, otherwise the empty tuple that disables base-contract
            targeted recovery.
        """
        return (self.center,) if self.recover_center else ()

    def pseudo_caustic_generators(self, values):
        """Implement the base pseudo-caustic capability for a smooth cusp.

        Parameters
        ----------
        values : Mapping[str, object]
            Ignored because this adapter is fully realized.

        Returns
        -------
        tuple of _PseudoCausticGenerator
            Always the empty tuple because a smooth cusp contributes no
            pseudo-caustic boundary to the base adapter contract.
        """
        return ()

    def axisymmetry_center(self, values):
        """Implement the base exact-axisymmetry-center capability.

        Parameters
        ----------
        values : Mapping[str, object]
            Ignored because this adapter is fully realized.

        Returns
        -------
        tuple of float, length 2, or None
            Stored image-plane center in arcseconds when exact axisymmetry is
            certified, otherwise ``None`` as required by the base contract.
        """
        return self.center if self.axisymmetric else None

    def preserves_axisymmetry(self, values):
        """Implement the base peer-axisymmetry-preservation capability.

        Parameters
        ----------
        values : Mapping[str, object]
            Ignored because this adapter is fully realized.

        Returns
        -------
        bool
            Always ``False`` because this non-affine adapter cannot serve as an
            affine symmetry-preserving peer.
        """
        return False

    def reference_num_images(self, values):
        """Implement the base regular-image reference-count capability.

        Parameters
        ----------
        values : Mapping[str, object]
            Ignored because this adapter is fully realized.

        Returns
        -------
        int
            Dimensionless one-image reference count for this atomic lens.
        """
        return 1


@dataclass(frozen=True)
class _AffinePerturbationGeometryAdapter(_GeometryAdapter):
    """Geometry capabilities for an affine lens-plane perturbation.

    Attributes
    ----------
    axisymmetry_preserving : bool
        Whether the perturbation preserves a non-affine peer's rotational
        symmetry.

    Notes
    -----
    ``search_center`` reads ``x0`` and ``y0`` from the realized ``values``
    mapping. Affine perturbations provide no independent extent, resolution,
    mask point, recovery point, pseudo-caustic generator, or symmetry center.
    ``frozen=True`` prevents field reassignment; the documented Boolean field is
    retained without normalization or copying.
    """

    axisymmetry_preserving: bool = False

    def search_center(self, values):
        """Implement the base search-center capability from realized values.

        Parameters
        ----------
        values : Mapping[str, object]
            Realized adapter-owned mapping containing float-convertible ``x0``
            and ``y0`` coordinates in image-plane arcseconds.

        Returns
        -------
        tuple of float, length 2
            Realized image-plane ``(x0, y0)`` center in arcseconds.

        Raises
        ------
        KeyError
            If either required center coordinate is absent.
        TypeError
            If a center coordinate is not float-convertible.
        ValueError
            If a center coordinate's conversion to ``float`` rejects its value.
        OverflowError
            If a center coordinate overflows during conversion to ``float``.
        """
        return float(values["x0"]), float(values["y0"])

    def initial_fov(self, values):
        """Implement the base initial-FOV capability without a finite extent.

        Parameters
        ----------
        values : Mapping[str, object]
            Ignored because affine geometry supplies no independent extent.

        Returns
        -------
        None
            Always ``None``, the base-contract sentinel for no independent
            finite image-plane FOV.
        """
        return None

    def resolution_scale(self, values):
        """Implement the base resolution capability without an angular scale.

        Parameters
        ----------
        values : Mapping[str, object]
            Ignored because affine geometry supplies no independent resolution.

        Returns
        -------
        None
            Always ``None``, the base-contract sentinel for no independent
            image-plane resolution scale.
        """
        return None

    def jacobian_mask_points(self, values):
        """Implement the base Jacobian-mask capability with no mask points.

        Parameters
        ----------
        values : Mapping[str, object]
            Ignored because affine geometry requires no Jacobian mask.

        Returns
        -------
        tuple of tuple of float
            Always the empty tuple, denoting no image-plane mask points under
            the base adapter contract.
        """
        return ()

    def root_recovery_points(self, values):
        """Implement the base root-recovery capability with no recovery points.

        Parameters
        ----------
        values : Mapping[str, object]
            Ignored because affine geometry requires no targeted recovery.

        Returns
        -------
        tuple of tuple of float
            Always the empty tuple, which disables base-contract targeted
            image-plane recovery.
        """
        return ()

    def pseudo_caustic_generators(self, values):
        """Implement the base pseudo-caustic capability with no generators.

        Parameters
        ----------
        values : Mapping[str, object]
            Ignored because affine geometry contributes no pseudo-caustic.

        Returns
        -------
        tuple of _PseudoCausticGenerator
            Always the empty tuple, denoting no pseudo-caustic boundary under
            the base adapter contract.
        """
        return ()

    def axisymmetry_center(self, values):
        """Implement the base exact-axisymmetry-center capability without one.

        Parameters
        ----------
        values : Mapping[str, object]
            Ignored because an affine perturbation has no independent center of
            rotational symmetry.

        Returns
        -------
        None
            Always ``None``, the base-contract sentinel for no independently
            certified image-plane symmetry center.
        """
        return None

    def preserves_axisymmetry(self, values):
        """Implement the base peer-axisymmetry-preservation capability.

        Parameters
        ----------
        values : Mapping[str, object]
            Ignored because this capability is fixed on the realized adapter.

        Returns
        -------
        bool
            Whether this affine perturbation preserves a non-affine peer's exact
            rotational symmetry under the base adapter contract.
        """
        return self.axisymmetry_preserving

    def reference_num_images(self, values):
        """Implement the base regular-image reference-count capability.

        Parameters
        ----------
        values : Mapping[str, object]
            Ignored because this capability is fixed on the realized adapter.

        Returns
        -------
        int
            Dimensionless reference count of one, contributing zero excess over
            the enclosing single-plane reference image.
        """
        return 1


@dataclass(frozen=True)
class _GeometryComponent:
    """Associate one recursively named lens with its geometry adapter.

    Attributes
    ----------
    name : str
        Generated positional path name used to select the component's realized
        adapter values.
    adapter : _GeometryAdapter
        Fresh realized adapter for the atomic or nested component.
    affine : bool
        Effective affinity of the complete component subtree.

    Notes
    -----
    ``frozen=True`` prevents field reassignment but is only a shallow ownership
    boundary. ``name`` and ``adapter`` are retained by identity, and any mutable
    state reachable through the adapter is neither copied nor deeply frozen.
    """

    name: str
    adapter: _GeometryAdapter
    affine: bool


@dataclass(frozen=True)
class _SinglePlaneGeometryAdapter(_GeometryAdapter):
    """Aggregate ordered component capabilities across one lens plane.

    Attributes
    ----------
    components : tuple of _GeometryComponent
        Components in the recursive specification's original sequence order.

    Notes
    -----
    Method ``values`` arguments map each component's generated ``name`` to its
    recursively realized adapter-owned mapping. Point and generator outputs are
    concatenated in component order. Numerical search bounds and resolution
    ignore affine components because those components supply neither an
    independent finite extent nor a scale.
    ``frozen=True`` prevents field reassignment but does not copy or deeply
    freeze the supplied component tuple or objects reachable through its
    component adapters.
    """

    components: tuple[_GeometryComponent, ...]

    def axisymmetry_center(self, values):
        """Return the plane's exact image-plane symmetry center or ``None``.

        Parameters
        ----------
        values : Mapping[str, Mapping[str, object]]
            Realized adapter-owned values keyed by generated component name.

        Returns
        -------
        tuple of float or None
            Exact image-plane ``(x, y)`` center in arcseconds, or ``None`` when
            the plane cannot certify axisymmetry.

        Notes
        -----
        A center is returned only when exactly one component is effectively
        non-affine, that component certifies an exact center, and every affine
        peer reports that it preserves axisymmetry. Affine peer centers are not
        compared with the non-affine center; no alignment tolerance is applied.
        """
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
        """Return whether every child is affine and symmetry-preserving.

        Parameters
        ----------
        values : Mapping[str, Mapping[str, object]]
            Realized adapter-owned values keyed by generated component name.

        Returns
        -------
        bool
            Whether the complete plane can preserve an enclosing component's
            exact axisymmetry.

        Notes
        -----
        This capability is used when the whole plane is an affine component of
        an enclosing ``SinglePlane``. Any non-affine child makes it false; an
        empty component tuple returns true by ``all`` semantics.
        """
        return all(
            component.affine and component.adapter.preserves_axisymmetry(values[component.name])
            for component in self.components
        )

    def reference_num_images(self, values):
        """Implement the base regular-image reference-count capability.

        Parameters
        ----------
        values : Mapping[str, Mapping[str, object]]
            Realized adapter-owned values keyed by generated component name.

        Returns
        -------
        int
            Dimensionless count equal to one plus each ordered component's
            excess over its own one-image reference. An empty component tuple
            therefore returns one.
        """
        return 1 + sum(
            component.adapter.reference_num_images(values[component.name]) - 1
            for component in self.components
        )

    def jacobian_mask_points(self, values):
        """Return component Jacobian mask points in recursive input order.

        Parameters
        ----------
        values : Mapping[str, Mapping[str, object]]
            Realized adapter-owned values keyed by generated component name.

        Returns
        -------
        tuple of tuple of float
            Image-plane ``(x, y)`` mask points in arcseconds.

        Notes
        -----
        Components without mask points contribute an empty tuple.
        """
        return tuple(
            point
            for component in self.components
            for point in component.adapter.jacobian_mask_points(values[component.name])
        )

    def root_recovery_points(self, values):
        """Return component recovery points in recursive input order.

        Parameters
        ----------
        values : Mapping[str, Mapping[str, object]]
            Realized adapter-owned values keyed by generated component name.

        Returns
        -------
        tuple of tuple of float
            Image-plane ``(x, y)`` recovery points in arcseconds.

        Notes
        -----
        Components without recovery points contribute an empty tuple.
        """
        return tuple(
            point
            for component in self.components
            for point in component.adapter.root_recovery_points(values[component.name])
        )

    def resolution_scale(self, values):
        """Return the smallest positive non-affine scale in arcseconds.

        Parameters
        ----------
        values : Mapping[str, Mapping[str, object]]
            Realized adapter-owned values keyed by generated component name.

        Returns
        -------
        float
            Smallest strictly positive child resolution scale in arcseconds.

        Notes
        -----
        Non-affine components whose returned scale is not strictly positive are
        excluded.

        Raises
        ------
        ValueError
            If no non-affine component supplies a strictly positive scale and
            ``min`` therefore receives an empty sequence.
        """
        return min(
            resolution
            for component in self.components
            if not component.affine
            if (resolution := component.adapter.resolution_scale(values[component.name])) > 0.0
        )

    def _search_envelope(self, values):
        """Compute the aggregate non-affine image-plane envelope.

        Parameters
        ----------
        values : Mapping[str, Mapping[str, object]]
            Realized adapter-owned values keyed by generated component name.

        Returns
        -------
        min_x, max_x, min_y, max_y : tuple of float
            Aggregate non-affine image-plane bounds in arcseconds.

        Notes
        -----
        Each component contributes a square centered on its own search center
        with its full initial extent. Affine components are excluded.
        """
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
        """Implement the base search-center capability from the plane envelope.

        Parameters
        ----------
        values : Mapping[str, Mapping[str, object]]
            Realized adapter-owned values keyed by generated component name.

        Returns
        -------
        tuple of float, length 2
            Midpoint ``(x, y)`` of the aggregate non-affine image-plane envelope
            in arcseconds.

        Raises
        ------
        ValueError
            If no non-affine component contributes envelope bounds and the
            delegated minimum or maximum reduction receives an empty sequence.

        Notes
        -----
        This concrete implementation fulfills ``_GeometryAdapter.search_center``
        by delegating its bounds calculation to ``_search_envelope``.
        """
        min_x, max_x, min_y, max_y = self._search_envelope(values)
        return 0.5 * (min_x + max_x), 0.5 * (min_y + max_y)

    def initial_fov(self, values):
        """Implement the base initial-FOV capability from the plane envelope.

        Parameters
        ----------
        values : Mapping[str, Mapping[str, object]]
            Realized adapter-owned values keyed by generated component name.

        Returns
        -------
        float
            Larger full width of the aggregate non-affine image-plane envelope
            in arcseconds; a valid non-affine plane never returns ``None``.

        Raises
        ------
        ValueError
            If no non-affine component contributes envelope bounds and the
            delegated minimum or maximum reduction receives an empty sequence.

        Notes
        -----
        This concrete implementation fulfills ``_GeometryAdapter.initial_fov``
        by delegating its bounds calculation to ``_search_envelope``.
        """
        min_x, max_x, min_y, max_y = self._search_envelope(values)
        return max(max_x - min_x, max_y - min_y)

    def pseudo_caustic_generators(self, values):
        """Return generators in component order with distinct-peer caps.

        Parameters
        ----------
        values : Mapping[str, Mapping[str, object]]
            Realized adapter-owned values keyed by generated component name.

        Returns
        -------
        tuple of _PseudoCausticGenerator
            Ordered child generators with any peer-separation caps applied.

        Notes
        -----
        Every non-affine component contributes its generator centers as peer
        locations, or its search center when it has no generator. For each
        returned generator, one quarter of the nearest strictly positive peer
        separation caps its initial radius; an existing smaller cap wins.
        Exact coincident peer locations do not create a zero-radius cap.
        With no contributing generator, the result is ``()``.
        """
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


def _caustics_scalar(value):
    """Convert one scalar Caustics tensor to a Python float.

    Parameters
    ----------
    value : torch.Tensor
        Scalar-valued realized Caustics tensor.

    Returns
    -------
    float
        Detached CPU scalar with Python ``float`` representation.
    """
    return float(np.asarray(_runtime._to_numpy(value)).item())


def _sis_geometry_adapter(lens, values):
    """Build realized point-singularity geometry for an SIS lens.

    Parameters
    ----------
    lens : caustics.SIS
        Realized SIS lens whose center and Einstein radius are static tensors.
    values : Mapping[str, object]
        Realized adapter-owned values; currently unused by this factory.

    Returns
    -------
    _PointSingularityGeometryAdapter
        Adapter centered on ``(x0, y0)`` with ``Rein`` as its resolution and
        ``2 * _INITIAL_FOV_PADDING * Rein`` as its full extent, all in
        arcseconds.

    Notes
    -----
    An exactly zero softening ``s`` enables center masking, root recovery, and
    pseudo-caustic generation. SIS geometry is marked exactly axisymmetric.
    """
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
    """Build realized point-singularity geometry for an SIE lens.

    Parameters
    ----------
    lens : caustics.SIE
        Realized SIE lens with static center, Einstein radius, and axis ratio.
    values : Mapping[str, object]
        Realized adapter-owned values; currently unused by this factory.

    Returns
    -------
    _PointSingularityGeometryAdapter
        Adapter centered on ``(x0, y0)`` with ``Rein`` as its resolution and
        ``2 * _INITIAL_FOV_PADDING * Rein / sqrt(q)`` as its full extent, all
        angular quantities in arcseconds.

    Notes
    -----
    An exactly zero softening ``s`` enables center masking, root recovery, and
    pseudo-caustic generation. Exact ``q == 1`` marks rotational symmetry.
    """
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
    """Build realized point-singularity geometry for an EPL lens.

    Parameters
    ----------
    lens : caustics.EPL
        Realized EPL lens with static center, Einstein radius, axis ratio, and
        dimensionless power-law slope.
    values : Mapping[str, object]
        Realized adapter-owned values; currently unused by this factory.

    Returns
    -------
    _PointSingularityGeometryAdapter
        Adapter centered on ``(x0, y0)`` with ``Rein`` as its resolution and
        ``2 * _INITIAL_FOV_PADDING * Rein / sqrt(q)`` as its full extent, all
        angular quantities in arcseconds.

    Notes
    -----
    Slopes ``t <= 1`` enable center masking and root recovery; exact ``t == 1``
    additionally enables pseudo-caustic generation. Exact ``q == 1`` marks
    rotational symmetry. The factory does not validate EPL's physical domain.
    """
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
    """Build realized smooth-cusp geometry for an NFW lens.

    Parameters
    ----------
    lens : caustics.NFW
        Realized NFW lens with a static center, redshift, mass, and
        concentration and a fixed cosmology.
    values : Mapping[str, object]
        Realized adapter-owned values; currently unused by this factory.

    Returns
    -------
    _SmoothCuspGeometryAdapter
        Axisymmetric adapter whose resolution is the physical scale radius in
        Mpc divided by angular-diameter distance in Mpc and converted from
        radians to arcseconds. Its full extent is twice that angular scale
        times ``_INITIAL_FOV_PADDING``.

    Raises
    ------
    ImportError
        If the installed Caustics constants module is unavailable.

    Notes
    -----
    An exactly zero softening ``s`` enables center masking and root recovery.
    Smooth-cusp geometry contributes no pseudo-caustic generator.
    """
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
    """Build realized point-singularity geometry for a TNFW lens.

    Parameters
    ----------
    lens : caustics.TNFW
        Realized TNFW lens with static center, angular scale radius, and
        dimensionless truncation ratio.
    values : Mapping[str, object]
        Realized adapter-owned values; currently unused by this factory.

    Returns
    -------
    _PointSingularityGeometryAdapter
        Axisymmetric adapter using ``Rs`` as its resolution and
        ``2 * _INITIAL_FOV_PADDING * tau * Rs`` as its full extent, in
        arcseconds.

    Notes
    -----
    An exactly zero softening ``s`` enables center masking and root recovery;
    a nonzero softening instead enables pseudo-caustic generation.
    """
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
    """Build realized smooth-cusp geometry for a Pseudo-Jaffe lens.

    Parameters
    ----------
    lens : caustics.PseudoJaffe
        Realized Pseudo-Jaffe lens with static center, core radius, and scale
        radius.
    values : Mapping[str, object]
        Realized adapter-owned values; currently unused by this factory.

    Returns
    -------
    _SmoothCuspGeometryAdapter
        Axisymmetric adapter using core radius ``Rc`` as its resolution and
        ``2 * _INITIAL_FOV_PADDING * Rs`` as its full extent, in arcseconds.

    Notes
    -----
    Center masking and root recovery are always enabled. Smooth-cusp geometry
    contributes no pseudo-caustic generator.
    """
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
    """Build realized affine geometry for an external-shear lens.

    Parameters
    ----------
    lens : caustics.ExternalShear
        Realized shear lens exposing dimensionless Cartesian components
        ``gamma_1`` and ``gamma_2``.
    values : Mapping[str, object]
        Realized adapter-owned values; currently unused by this factory and
        retained for later ``x0``/``y0`` center lookup by the adapter.

    Returns
    -------
    _AffinePerturbationGeometryAdapter
        Extent-free affine adapter that preserves axisymmetry only when both
        realized Cartesian shear components are exactly zero.
    """
    gamma_1 = _caustics_scalar(lens.gamma_1.value)
    gamma_2 = _caustics_scalar(lens.gamma_2.value)
    return _AffinePerturbationGeometryAdapter(
        axisymmetry_preserving=gamma_1 == 0.0 and gamma_2 == 0.0,
    )


def _mass_sheet_geometry_adapter(lens, values):
    """Build realized affine geometry for a mass-sheet lens.

    Parameters
    ----------
    lens : caustics.MassSheet
        Realized mass-sheet lens; currently unused by this factory.
    values : Mapping[str, object]
        Realized adapter-owned values; currently unused by this factory and
        retained for later ``x0``/``y0`` center lookup by the adapter.

    Returns
    -------
    _AffinePerturbationGeometryAdapter
        Extent-free affine adapter marked as preserving rotational symmetry.
    """
    return _AffinePerturbationGeometryAdapter(axisymmetry_preserving=True)


@dataclass(frozen=True)
class _RegisteredLensModel:
    """Record construction metadata for one supported model.

    Attributes
    ----------
    affine : bool
        Whether the atomic model is an extent-free affine perturbation.
        ``SinglePlane`` uses a fixed registry value but computes effective
        affinity recursively from its children.
    geometry_factory : object
        Callable constructing the model's realized geometry adapter. Atomic
        factories accept ``(lens, values)``; ``SinglePlane`` stores
        ``_SinglePlaneGeometryAdapter``, which accepts an ordered component
        tuple.

    Notes
    -----
    ``frozen=True`` prevents field reassignment but does not copy or deeply
    freeze the supplied callable. ``geometry_factory`` is retained by identity,
    including any mutable state reachable through a callable object.
    """

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


def _lens_spec_is_affine(lens):
    """Determine the effective affinity of a recursive lens specification.

    Parameters
    ----------
    lens : CausticsLensSpec
        Registered atomic or ``SinglePlane`` specification.

    Returns
    -------
    bool
        Registry affinity for an atomic model, or whether every child of a
        ``SinglePlane`` is recursively affine. An empty plane is therefore
        affine by ``all`` semantics.
    """
    if lens.model == "SinglePlane":
        is_affine = all(_lens_spec_is_affine(child) for child in lens.parameters["lenses"])
    else:
        is_affine = _LENS_MODEL_REGISTRY[lens.model].affine
    return is_affine


def _validate_root_lens_spec(lens):
    """Validate redshift ownership and non-affinity for a consumed root spec.

    Parameters
    ----------
    lens : CausticsLensSpec
        Recursive specification about to be consumed by a strong-lensing node.

    Returns
    -------
    None
        The specification is accepted unchanged.

    Raises
    ------
    ValueError
        If the root omits ``z_l``, any nested specification defines ``z_l``
        instead of inheriting it, or the complete tree has no non-affine model.
    """
    missing = _ROOT_LENS_PARAMETERS.difference(lens.parameters)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"The root lens specification requires: {names}.")

    def validate_children(spec):
        """Recursively reject root-owned parameters on nested specs.

        Parameters
        ----------
        spec : CausticsLensSpec
            Current recursive parent whose ``lenses`` children are inspected.

        Raises
        ------
        ValueError
            If a direct or transitive child explicitly defines ``z_l``.
        """
        for child in spec.parameters.get("lenses", ()):
            inherited = _ROOT_LENS_PARAMETERS.intersection(child.parameters)
            if inherited:
                names = ", ".join(sorted(inherited))
                raise ValueError(f"Nested lens specifications inherit rather than define: {names}.")
            validate_children(child)

    validate_children(lens)
    if _lens_spec_is_affine(lens):
        raise ValueError("Caustics strong-lensing nodes require at least one non-affine lens model.")


def _lens_graph_inputs(lens, prefix="lens"):
    """Flatten one recursive specification into graph input names and setters.

    Parameters
    ----------
    lens : CausticsLensSpec
        Registered atomic or recursive specification.
    prefix : str, optional
        Generated graph-name prefix for the current specification. The public
        nodes use ``"lens"`` at the root.

    Returns
    -------
    tuple of tuple of (str, object)
        ``(graph_name, setter)`` pairs in mapping and recursive child order.
        Setter objects retain the identities stored by ``CausticsLensSpec``.

    Notes
    -----
    Ordinary fields become ``<prefix>_<field>``. ``SinglePlane.lenses`` is not
    a graph input; child index ``i`` instead recurses with
    ``<prefix>_<i>``. Thus nested names encode the full positional path before
    the final constructor field.
    """
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
    cosmology,
    z_l=None,
    z_s=None,
    root=False,
):
    """Recursively construct fresh Caustics objects from one graph sample.

    Parameters
    ----------
    lens_spec : CausticsLensSpec
        Atomic or recursive specification for the current subtree.
    values : Mapping[str, object]
        One realized graph sample keyed by flattened graph input name.
    caustics : module
        Imported Caustics package exporting every registered lens class.
    torch : module
        Imported PyTorch backend used to create float64 redshift tensors.
    prefix : str
        Flattened positional graph prefix for the current subtree.
    name : str
        Generated positional Caustics object name for the current subtree.
    cosmology : caustics.Cosmology
        Fixed node-owned cosmology object passed by identity to every
        constructor.
    z_l : torch.Tensor or None, optional
        Inherited lens-plane redshift supplied to this constructor.
    z_s : torch.Tensor or None, optional
        Inherited source redshift supplied to this constructor.
    root : bool, optional
        Whether to consume ``<prefix>_z_l`` from ``values`` as the root plane's
        float64 lens-redshift tensor.

    Returns
    -------
    object
        Fresh realized Caustics atomic lens or ``SinglePlane`` tree.

    Notes
    -----
    An explicit sampled ``name`` overrides the generated name for that object;
    descendant defaults continue to follow the positional ``lens_0_1`` path.
    ``SinglePlane`` children are constructed first in specification order with
    ``z_l`` and ``z_s`` left as ``None`` so the containing Caustics plane owns
    the root redshifts. Constructor defaults and physical validation remain
    Caustics responsibilities. No constructed object is cached or reused.
    """
    lens_values = {
        parameter_name: values[f"{prefix}_{parameter_name}"]
        for parameter_name in lens_spec.parameters
        if parameter_name != "lenses"
    }
    if root:
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
    """Staticize atomic lenses and recursively build realized geometry.

    Parameters
    ----------
    lens_spec : CausticsLensSpec
        Specification corresponding exactly to ``lens``.
    lens : object
        Fresh Caustics atomic lens or ``SinglePlane`` subtree.
    values : Mapping[str, object]
        One realized graph sample keyed by flattened graph input name.
    torch : module
        PyTorch backend used to cast eligible configured Caustics parameters to
        float64.
    prefix : str
        Flattened positional graph prefix for the current subtree.
    name : str
        Generated positional component name for the current subtree.
    root : bool, optional
        Whether the current spec owns and therefore removes ``z_l`` from its
        adapter values.

    Returns
    -------
    adapter : _GeometryAdapter
        Fresh adapter for the realized atomic lens or composite plane.
    adapter_values : Mapping[str, object]
        Read-only atomic adapter values with structural ``name`` and root
        ``z_l`` removed, or a read-only recursive mapping from generated child
        names to their adapter values.
    affine : bool
        Registry affinity for an atom or effective all-child affinity for a
        composite.
    non_affine_components : tuple of tuple
        ``(generated_name, (x, y))`` pairs for every atomic non-affine
        descendant, with image-plane centers in arcseconds and recursive input
        order.

    Raises
    ------
    ValueError
        If a realized ``SinglePlane`` has a different child count from its
        specification, or any two non-affine descendants have exactly equal
        center tuples. The center-collision error names both generated
        positional components.

    Notes
    -----
    Each explicitly configured atomic parameter exposing Caustics ``value``
    and ``to`` capabilities is cast to ``torch.float64`` before the atomic lens
    is changed to static mode. Composite planes are represented recursively;
    their atomic descendants, rather than the ``SinglePlane`` wrapper, are
    staticized. Adapter values preserve the realized graph objects that are not
    structural fields.
    """
    lens_values = {
        parameter_name: values[f"{prefix}_{parameter_name}"]
        for parameter_name in lens_spec.parameters
        if parameter_name != "lenses"
    }
    if root:
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


def _build_lens_system(lens_spec, *, cosmology, values):
    """Build one fresh registered lens system and its geometry contract.

    Parameters
    ----------
    lens_spec : CausticsLensSpec
        Validated root atomic or recursive lens specification.
    cosmology : caustics.Cosmology
        Fixed node-owned cosmology passed by identity throughout the tree.
    values : Mapping[str, object]
        Realized flattened lens inputs plus dimensionless ``source_redshift``
        for one graph sample.

    Returns
    -------
    lens : object
        Fresh root Caustics lens or ``SinglePlane`` object.
    geometry_adapter : _GeometryAdapter
        Fresh total-system geometry adapter derived from that same lens.
    adapter_values : Mapping[str, object]
        Read-only atomic or recursively nested realized values consumed by the
        adapter.

    Raises
    ------
    ImportError
        If Caustics or its PyTorch backend is unavailable.
    ValueError
        If recursive geometry realization finds exactly coincident non-affine
        component centers or inconsistent composite child counts.

    Notes
    -----
    Root ``lens_z_l`` and ``source_redshift`` are converted to float64 tensors
    and injected as ``z_l`` and ``z_s``. A call constructs new lens and adapter
    objects and performs no node-level caching. Importing the runtime also sets
    Torch's process-wide default dtype to float64.
    """
    caustics, torch = _runtime._import_caustics_dependencies()
    lens = _construct_lens_tree(
        lens_spec,
        values,
        caustics=caustics,
        torch=torch,
        prefix="lens",
        name="lens",
        cosmology=cosmology,
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

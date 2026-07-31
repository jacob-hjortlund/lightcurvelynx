"""Caustics-backed lens specifications and strong-lensing graph nodes.

``CausticsLensSpec`` recursively describes one of the nine registered Caustics
models: the atomic ``SIS``, ``SIE``, ``EPL``, ``NFW``, ``TNFW``,
``PseudoJaffe``, ``ExternalShear``, and ``MassSheet`` models, or a
``SinglePlane`` composition. ``CausticsSourcePositionNode`` samples a certified
geometric multiple-imaging source region for supported realized geometries,
while ``CausticsLensImageNode`` solves for point-source macro-images and their
magnifications and relative delays.

Lens- and source-plane coordinates, angular radii, field-of-view widths, and
solver scales are expressed in arcseconds. Redshifts, axis ratios, lens-profile
slopes, shear, convergence, and scale fractions are dimensionless; masses use
solar masses, physical NFW radii use Mpc, and image delays use days. The root
specification owns the lens-redshift graph dependency ``z_l``. The consuming
node owns one fixed cosmology object and supplies its sampled source redshift as
``z_s`` to every fresh realization.

Caustics is imported lazily for specification inspection and lens realization;
PyTorch is loaded explicitly only with the realization runtime. ContourPy and
Shapely are additional lazy dependencies of source-region sampling. Every
successful call to
``lightcurvelynx.models._caustics.runtime._import_caustics_dependencies()``
sets Torch's process-wide default dtype to ``torch.float64``; this side effect
is not restored by this module.
"""

# TODO: Add the Caustics archival DOI/paper and ContourPy credit to the relevant
# public-node reference sections.

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from citation_compass import CiteClass

from lightcurvelynx.base_models import FunctionNode
from lightcurvelynx.models._caustics import image_recovery as _image_recovery
from lightcurvelynx.models._caustics import lens_system as _lens_system
from lightcurvelynx.models._caustics import runtime as _runtime
from lightcurvelynx.models._caustics import source_geometry as _source_geometry

__all__ = [
    "CausticsLensImageNode",
    "CausticsLensSpec",
    "CausticsSourcePositionNode",
]


@dataclass(frozen=True, init=False)
class CausticsLensSpec:
    """Describe one registered Caustics constructor in a recursive lens graph.

    Parameters
    ----------
    model : str
        Explicit registry key: ``SIS``, ``SIE``, ``EPL``, ``NFW``, ``TNFW``,
        ``PseudoJaffe``, ``ExternalShear``, ``MassSheet``, or ``SinglePlane``.
    parameters : Mapping[str, object]
        Constructor arguments other than node-owned ``cosmology`` and ``z_s``.
        Values other than ``SinglePlane.lenses`` remain graph dependency
        setters with their original object identity.

    Attributes
    ----------
    model : str
        Registered constructor name.
    parameters : Mapping[str, object]
        Read-only shallow snapshot of the supplied constructor mapping. A
        ``SinglePlane`` lens sequence is stored as a tuple; its nested
        ``CausticsLensSpec`` objects retain their identities.

    Notes
    -----
    Construction validates the selected class's current Python signature, not
    the physical domain of realized values. Required constructor arguments
    must be explicit or realization-owned, and explicit names must occur in
    that signature. ``ExternalShear`` additionally switches accepted shear
    names from ``gamma_1``/``gamma_2`` to ``gamma``/``phi`` when its explicit
    string ``parametrization`` is ``"angular"``. A non-string parametrization
    admits both Cartesian and angular names so a graph dependency can select
    the mode at realization time. Caustics remains responsible for the
    realized parametrization value, mutually dependent fields, ranges, shapes,
    and all other physical or numerical constraints.

    Root and nested redshift ownership is checked when a graph node consumes
    the specification: the root must define dimensionless ``z_l`` and every
    nested spec must omit it and inherit its plane redshift. The node-owned
    cosmology is fixed rather than sampled, and the node registers and realizes
    dimensionless source redshift as ``z_s``. ``SinglePlane.lenses`` describes
    structure and is never registered as a graph input.
    """

    model: str
    parameters: Mapping[str, object]

    def __init__(self, model, parameters):
        """Snapshot and structurally validate one registered lens description.

        Parameters
        ----------
        model : str
            Name of one explicitly registered Caustics class.
        parameters : Mapping[str, object]
            Constructor-name-to-graph-setter mapping. ``cosmology`` and ``z_s``
            are forbidden because the consuming node supplies them. For
            ``SinglePlane``, ``lenses`` must be an iterable containing only
            ``CausticsLensSpec`` objects and is converted to a tuple.

        Raises
        ------
        TypeError
            If ``parameters`` is not a mapping, a key is not a string, the
            ``SinglePlane.lenses`` value is not iterable, or that iterable
            contains a non-``CausticsLensSpec`` object. An unhashable ``model``
            also propagates the registry lookup's ``TypeError``.
        ValueError
            If ``model`` is not registered; ``cosmology`` or ``z_s`` is
            supplied explicitly; a required constructor argument is missing;
            or an explicit parameter name is unsupported by the selected
            constructor and active ``ExternalShear`` parametrization.
        ImportError
            If Caustics or one of its import-time dependencies is unavailable
            while the constructor signature is inspected.

        Notes
        -----
        The mapping copy is shallow. Mutable values and graph dependency
        setters are not copied, and no realized value is physically validated
        here. Root ``z_l`` presence and nested ``z_l`` absence are deferred to
        consuming-node validation.
        """
        parameter_snapshot = _lens_system._validated_constructor_parameters(model, parameters)
        if model == "SinglePlane":
            lenses = parameter_snapshot["lenses"]
            if any(not isinstance(lens, CausticsLensSpec) for lens in lenses):
                raise TypeError("SinglePlane lenses must contain only CausticsLensSpec objects.")

        object.__setattr__(self, "model", model)
        object.__setattr__(self, "parameters", MappingProxyType(parameter_snapshot))


class CausticsSourcePositionNode(FunctionNode, CiteClass):
    """Sample the complete geometric strong-lensing source region.

    For each realized atomic or single-plane lens configuration, this node
    maps total-lens critical curves into true caustics and maps component-owned
    pseudo-caustics through the total lens. It structurally repairs the regular
    source-plane interiors, draws uniform candidate positions from their union,
    and certifies the regular-image count with signed boundary winding.
    Candidates with insufficient boundary clearance are retried within one
    total draw budget, while an image-count mismatch is immediately fatal. The
    accepted point, complete geometric cross-section, cumulative draw count,
    image count, and boundary diagnostics form nine newly computed results.
    After every sample succeeds, they are passed to ``_save_results``; its
    ``GraphState.set`` calls update eligible nonfixed output entries and
    preserve any pre-fixed output entries. ``compute`` nevertheless returns
    the newly computed value for every output.

    Parameters
    ----------
    lens : CausticsLensSpec
        Registered atomic or recursive lens specification. The root parameters
        include the lens-redshift setter ``z_l``.
    cosmology : caustics.Cosmology
        Fixed cosmology inherited by every realized lens.
    source_redshift : object
        Graph setter for dimensionless source redshift.
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
        Maximum configured initial image-plane loop radius for pseudo-caustics
        in arcseconds. When ``pseudo_caustic_epsilon_fraction`` is enabled,
        this becomes an upper bound on the realized per-lens radius.
    pseudo_caustic_epsilon_fraction : float or None, optional
        Positive dimensionless initial pseudo-caustic loop radius as a fraction
        of the realized characteristic angular scale, or ``None`` to use only
        ``pseudo_caustic_epsilon``.
    geometry_tolerance : float, optional
        Maximum configured curve-closure, convergence, and topology precision
        in arcseconds. When ``geometry_tolerance_fraction`` is enabled, this
        becomes an upper bound on the realized per-lens tolerance.
    geometry_tolerance_fraction : float or None, optional
        Positive dimensionless curve and topology tolerance as a fraction of
        the realized characteristic angular scale, or ``None`` to use only
        ``geometry_tolerance``.
    boundary_tolerance : float, optional
        Maximum configured matched-boundary displacement required for
        certification in arcseconds. When ``boundary_tolerance_fraction`` is
        enabled, this becomes an upper bound on the realized per-lens
        tolerance. It must be at least ``geometry_tolerance``.
    boundary_tolerance_fraction : float or None, optional
        Positive dimensionless matched-boundary displacement as a fraction of
        the realized characteristic angular scale, or ``None`` to use only
        ``boundary_tolerance``. It must be at least
        ``geometry_tolerance_fraction`` when both are enabled.
    max_boundary_refinements : int, optional
        Maximum number of factor-of-two resolution refinements used to certify
        boundary displacement and topology.
    max_attempts : int, optional
        Maximum coordinate-pair draw budget per lens realization. Polygon,
        exact-point-caustic, and narrow-boundary rejections all consume it.
    seed : object, optional
        Seed accepted by ``numpy.random.default_rng`` for the node-owned
        fallback generator.
    node_label : str, optional
        Human-readable graph node identifier.

    Attributes
    ----------
    lens : CausticsLensSpec
        Stored registered specification used to construct a fresh realized lens
        tree for every graph sample.
    cosmology : caustics.Cosmology
        Fixed node-owned cosmology excluded from the sampled ``GraphState``.
    fov : float or None
        Normalized configured initial image-plane search width in arcseconds,
        or ``None`` for adapter-derived realization.
    pixelscale : float
        Normalized absolute Jacobian-grid spacing upper bound in arcseconds.
    pixelscale_fraction : float or None
        Normalized dimensionless relative-spacing fraction, or ``None``.
    max_fov_expansions : int
        Stored maximum number of bounded FOV expansions.
    fov_expansion_factor : float
        Stored dimensionless expansion multiplier.
    pseudo_caustic_points : int
        Stored initial unique-vertex count for each pseudo-caustic loop.
    pseudo_caustic_epsilon : float
        Stored absolute pseudo-caustic loop-radius cap in arcseconds.
    pseudo_caustic_epsilon_fraction : float or None
        Stored positive dimensionless pseudo-caustic loop-radius fraction, or
        ``None``.
    geometry_tolerance : float
        Stored absolute curve-normalization and topology-tolerance cap in
        arcseconds.
    geometry_tolerance_fraction : float or None
        Stored positive dimensionless curve and topology tolerance fraction,
        or ``None``.
    boundary_tolerance : float
        Stored absolute certification-displacement tolerance cap in
        arcseconds.
    boundary_tolerance_fraction : float or None
        Stored positive dimensionless certification-displacement tolerance
        fraction, or ``None``.
    max_boundary_refinements : int
        Stored maximum number of factor-of-two refinement steps.
    max_attempts : int
        Stored maximum coordinate-pair draw budget per realization.
    _rng : numpy.random.Generator
        Node-owned seeded fallback generator used only when ``compute`` is not
        given ``rng_info``.
    source_redshift : AttributeIndicator
        Dynamic graph input for dimensionless source redshift.
    lens_<path>_<field> : AttributeIndicator
        Dynamic graph inputs flattened from ``lens``. The root path begins
        ``lens`` (including ``lens_z_l``); nested ``SinglePlane`` child indices
        are inserted before the atomic field name.
    source_x : AttributeIndicator
        Graph output for sampled source-plane x position in arcseconds.
    source_y : AttributeIndicator
        Graph output for sampled source-plane y position in arcseconds.
    strong_lensing_area : AttributeIndicator
        Graph output for geometric source-plane area in square arcseconds.
    sampling_attempts : AttributeIndicator
        Graph output for the cumulative coordinate-pair draw count, including
        polygon, exact-point-caustic, and narrow-boundary rejections.
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
        Graph output for the number of completed factor-of-two refinement
        steps.

    Notes
    -----
    ``strong_lensing_area`` is the complete geometric source-plane
    cross-section in square arcseconds. It is not reduced when accepted source
    positions are conditioned to exceed the boundary uncertainty. It does not
    include magnification bias, detectability, cadence, image resolution, or
    cross-section weighting of the upstream lens sample.
    The configured ``pixelscale`` is an absolute upper bound. An optional
    fraction produces a realized initial upper bound per lens; each boundary
    refinement requests half the previous scale, while
    ``critical_curve_fov`` records the FOV that succeeded for that snapshot
    and ``boundary_uncertainty`` compares consecutive snapshots. Adaptive FOV
    recovery keeps the requested scale fixed and multiplies its FOV by
    ``fov_expansion_factor``.
    All configured attributes remain unchanged during sampling; scale-aware
    settings are realized separately for each lens. The equal numerical
    absolute and fractional defaults reproduce the existing setting when the
    characteristic angular scale is one arcsecond.

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
    curves provides the certified image count. A candidate is accepted only
    after the penultimate and final winding counts agree and its final regular
    boundary clearance is strictly greater than the matched-boundary
    uncertainty. Insufficient clearance retries from the remaining
    ``max_attempts`` budget; a count mismatch remains fatal and is never
    retried.

    The exact output order is ``source_x``, ``source_y``,
    ``strong_lensing_area``, ``sampling_attempts``, ``expected_num_images``,
    ``critical_curve_fov``, ``boundary_uncertainty``,
    ``source_boundary_clearance``, and ``boundary_refinements``. Each is a
    NumPy scalar for one graph sample and has shape ``(S,)`` for ``S > 1``;
    ``strong_lensing_area`` is in square arcseconds,
    ``sampling_attempts`` counts all polygon, exact-point-caustic, and
    narrow-boundary rejected coordinate pairs as well as the accepted pair,
    and ``boundary_refinements`` counts completed factor-of-two steps. Only
    after every graph sample succeeds are these newly computed results passed
    in order to ``_save_results``. Exhaustion therefore leaves all nine outputs
    from this call unsaved. Its
    ``GraphState.set`` calls update eligible nonfixed output entries and leave
    pre-fixed output entries unchanged. A successful return still contains all
    newly computed values, including those corresponding to preserved
    pre-fixed entries. A caller RNG takes precedence over the seeded fallback,
    and exactly one unsigned 64-bit sub-seed per graph sample is drawn up front
    before any per-sample work. Sample-local generators isolate variable
    rejection counts; downstream use of the ``GraphState`` sees each output's
    resulting updated or pre-fixed value.

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
        cosmology,
        source_redshift,
        fov=None,
        pixelscale=0.01,
        pixelscale_fraction=None,
        max_fov_expansions=5,
        fov_expansion_factor=1.25,
        pseudo_caustic_points=2_048,
        pseudo_caustic_epsilon=1.0e-5,
        pseudo_caustic_epsilon_fraction=1.0e-5,
        geometry_tolerance=1.0e-6,
        geometry_tolerance_fraction=1.0e-6,
        boundary_tolerance=1.0e-4,
        boundary_tolerance_fraction=1.0e-4,
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
            the dimensionless lens-redshift setter ``z_l``.
        cosmology : caustics.Cosmology
            Fixed cosmology inherited by every realized lens and excluded from
            graph parameters.
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
            Positive absolute initial pseudo-caustic loop-radius cap in
            arcseconds when ``pseudo_caustic_epsilon_fraction`` is enabled.
        pseudo_caustic_epsilon_fraction : float-convertible scalar or None, optional
            Positive dimensionless initial pseudo-caustic loop radius as a
            fraction of the realized characteristic angular scale, or
            ``None`` to use only ``pseudo_caustic_epsilon``. A
            zero-dimensional NumPy array is accepted.
        geometry_tolerance : float-convertible scalar, optional
            Positive absolute curve and topology tolerance cap in arcseconds
            when ``geometry_tolerance_fraction`` is enabled; smaller than
            ``pixelscale``.
        geometry_tolerance_fraction : float-convertible scalar or None, optional
            Positive dimensionless curve and topology tolerance as a fraction
            of the realized characteristic angular scale, or ``None`` to use
            only ``geometry_tolerance``. A zero-dimensional NumPy array is
            accepted.
        boundary_tolerance : float-convertible scalar, optional
            Positive absolute matched-boundary tolerance cap in arcseconds when
            ``boundary_tolerance_fraction`` is enabled; at least
            ``geometry_tolerance``.
        boundary_tolerance_fraction : float-convertible scalar or None, optional
            Positive dimensionless matched-boundary tolerance as a fraction of
            the realized characteristic angular scale, or ``None`` to use only
            ``boundary_tolerance``. It must be at least
            ``geometry_tolerance_fraction`` when both are enabled. A
            zero-dimensional NumPy array is accepted.
        max_boundary_refinements : int, optional
            Positive maximum number of factor-of-two boundary-refinement
            steps.
        max_attempts : int, optional
            Positive total coordinate-pair draw budget per realization.
            Polygon, exact-point-caustic, and narrow-boundary rejections all
            consume it.
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
        ``source_redshift`` is registered directly. The fixed ``cosmology`` is
        stored on the node rather than registered. Lens parameters use
        ``lens_<field>`` at the root and positional path segments inside nested
        planes. The nine public outputs are registered in exact order:
        ``source_x``, ``source_y``, ``strong_lensing_area``,
        ``sampling_attempts``, ``expected_num_images``,
        ``critical_curve_fov``, ``boundary_uncertainty``,
        ``source_boundary_clearance``, and ``boundary_refinements``.
        Source-node integer checks use ``isinstance(value, int)``: NumPy
        integer scalars are rejected, while booleans follow Python's integer
        subclass semantics before range checks.
        The seeded node-owned generator is used only when ``compute`` receives
        no caller generator. Exceptions raised by
        ``numpy.random.default_rng`` for an unsupported ``seed`` propagate
        unchanged.
        """
        if not isinstance(lens, CausticsLensSpec):
            raise TypeError("lens must be a CausticsLensSpec.")
        _lens_system._validate_root_lens_spec(lens)
        lens_graph_inputs = _lens_system._lens_graph_inputs(lens)

        pixelscale_fraction = _runtime._validate_optional_positive_fraction(
            "pixelscale_fraction",
            pixelscale_fraction,
        )
        pseudo_caustic_epsilon_fraction = _runtime._validate_optional_positive_fraction(
            "pseudo_caustic_epsilon_fraction",
            pseudo_caustic_epsilon_fraction,
        )
        geometry_tolerance_fraction = _runtime._validate_optional_positive_fraction(
            "geometry_tolerance_fraction",
            geometry_tolerance_fraction,
        )
        boundary_tolerance_fraction = _runtime._validate_optional_positive_fraction(
            "boundary_tolerance_fraction",
            boundary_tolerance_fraction,
        )
        normalized = _source_geometry._validate_source_position_configuration(
            fov=fov,
            pixelscale=pixelscale,
            pixelscale_fraction=pixelscale_fraction,
            max_fov_expansions=max_fov_expansions,
            fov_expansion_factor=fov_expansion_factor,
            pseudo_caustic_points=pseudo_caustic_points,
            pseudo_caustic_epsilon=pseudo_caustic_epsilon,
            geometry_tolerance=geometry_tolerance,
            geometry_tolerance_fraction=geometry_tolerance_fraction,
            boundary_tolerance=boundary_tolerance,
            boundary_tolerance_fraction=boundary_tolerance_fraction,
            max_boundary_refinements=max_boundary_refinements,
            max_attempts=max_attempts,
        )

        self.lens = lens
        self.cosmology = cosmology
        self.fov = normalized.get("fov")
        self.pixelscale = normalized["pixelscale"]
        self.pixelscale_fraction = pixelscale_fraction
        self.max_fov_expansions = int(max_fov_expansions)
        self.fov_expansion_factor = normalized["fov_expansion_factor"]
        self.pseudo_caustic_points = int(pseudo_caustic_points)
        self.pseudo_caustic_epsilon = normalized["pseudo_caustic_epsilon"]
        self.pseudo_caustic_epsilon_fraction = pseudo_caustic_epsilon_fraction
        self.geometry_tolerance = normalized["geometry_tolerance"]
        self.geometry_tolerance_fraction = geometry_tolerance_fraction
        self.boundary_tolerance = normalized["boundary_tolerance"]
        self.boundary_tolerance_fraction = boundary_tolerance_fraction
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
        """Describe the configured root lens deterministically.

        Returns
        -------
        str
            Text of the form ``"lens model '<model>'"`` using the root
            specification's model name.
        """
        return f"lens model {self.lens.model!r}"

    def _lens_identifier(self, sample_index):
        """Build a contextual identifier for one graph realization.

        Parameters
        ----------
        sample_index : int
            Zero-based graph sample index.

        Returns
        -------
        str
            Root lens description, sample index, and current ``node_string``
            for use in failure diagnostics.
        """
        return f"{self._lens_description()} sample {sample_index} at node '{self.node_string}'"

    def _realized_angular_settings(self, geometry_adapter, values, *, sample_index):
        """Realize all source-geometry angular settings for one lens.

        Parameters
        ----------
        geometry_adapter : _GeometryAdapter
            Realized total-lens adapter for this lens system.
        values : Mapping[str, object]
            Realized inputs for one lens system.
        sample_index : int
            Zero-based graph sample index used in contextual exceptions.

        Returns
        -------
        characteristic_scale : float or None
            Adapter-provided characteristic angular scale in arcseconds when
            any fraction is enabled, otherwise ``None``.
        realized_pixelscale : float
            Sample-local Jacobian-grid spacing upper bound in arcseconds.
        realized_pseudo_caustic_epsilon : float
            Sample-local initial pseudo-caustic loop radius in arcseconds.
        realized_geometry_tolerance : float
            Sample-local curve and topology tolerance in arcseconds.
        realized_boundary_tolerance : float
            Sample-local matched-boundary tolerance in arcseconds.

        Raises
        ------
        TypeError
            If a required characteristic scale is not a scalar value
            convertible to ``float``.
        ValueError
            If a required characteristic scale is not finite and positive, or
            the realized geometry/pixel or boundary/geometry relations fail.

        Notes
        -----
        When at least one fraction is enabled, this method calls
        ``resolution_scale(values)`` exactly once and reuses that scale for all
        four policies. Each enabled fraction produces the smaller of its
        configured absolute cap and the fraction times the characteristic
        scale. A ``None`` fraction uses only the corresponding absolute value.
        When every fraction is ``None``, no scale lookup occurs and the first
        return value is ``None``. Scale and dynamic-relation failures include
        the lens/sample identifier and configured, fractional, and realized
        context where applicable.
        """
        configured = (
            self.pixelscale,
            self.pseudo_caustic_epsilon,
            self.geometry_tolerance,
            self.boundary_tolerance,
        )
        fractions = (
            self.pixelscale_fraction,
            self.pseudo_caustic_epsilon_fraction,
            self.geometry_tolerance_fraction,
            self.boundary_tolerance_fraction,
        )
        if all(fraction is None for fraction in fractions):
            return (None, *configured)

        raw_scale = geometry_adapter.resolution_scale(values)
        if isinstance(raw_scale, np.ndarray) and raw_scale.ndim != 0:
            raise TypeError(
                "Characteristic angular scale must be a scalar value convertible "
                f"to float for {self._lens_identifier(sample_index)}."
            )
        try:
            characteristic_scale = float(raw_scale)
        except (TypeError, ValueError, OverflowError) as err:
            raise TypeError(
                "Characteristic angular scale must be a scalar value convertible "
                f"to float for {self._lens_identifier(sample_index)}."
            ) from err
        if not np.isfinite(characteristic_scale) or characteristic_scale <= 0.0:
            raise ValueError(
                "Characteristic angular scale must be finite and positive for "
                f"{self._lens_identifier(sample_index)}; "
                f"characteristic_scale={characteristic_scale}."
            )

        realized = tuple(
            absolute if fraction is None else min(absolute, fraction * characteristic_scale)
            for absolute, fraction in zip(configured, fractions, strict=True)
        )
        (
            realized_pixelscale,
            realized_pseudo_caustic_epsilon,
            realized_geometry_tolerance,
            realized_boundary_tolerance,
        ) = realized
        context = (
            f"characteristic_scale={characteristic_scale}, "
            f"configured_pixelscale={self.pixelscale}, "
            f"pixelscale_fraction={self.pixelscale_fraction}, "
            f"realized_pixelscale={realized_pixelscale}, "
            f"configured_pseudo_caustic_epsilon={self.pseudo_caustic_epsilon}, "
            "pseudo_caustic_epsilon_fraction="
            f"{self.pseudo_caustic_epsilon_fraction}, "
            f"realized_pseudo_caustic_epsilon={realized_pseudo_caustic_epsilon}, "
            f"configured_geometry_tolerance={self.geometry_tolerance}, "
            f"geometry_tolerance_fraction={self.geometry_tolerance_fraction}, "
            f"realized_geometry_tolerance={realized_geometry_tolerance}, "
            f"configured_boundary_tolerance={self.boundary_tolerance}, "
            f"boundary_tolerance_fraction={self.boundary_tolerance_fraction}, "
            f"realized_boundary_tolerance={realized_boundary_tolerance}"
        )
        if realized_geometry_tolerance >= realized_pixelscale:
            raise ValueError(
                "realized_geometry_tolerance must be smaller than "
                f"realized_pixelscale for {self._lens_identifier(sample_index)}; {context}."
            )
        if realized_boundary_tolerance < realized_geometry_tolerance:
            raise ValueError(
                "realized_boundary_tolerance must be at least "
                f"realized_geometry_tolerance for {self._lens_identifier(sample_index)}; "
                f"{context}."
            )
        return (characteristic_scale, *realized)

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
        geometry_tolerance,
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
            Already-realized sample-local maximum Jacobian-grid spacing in
            arcseconds, held fixed throughout FOV recovery.
        geometry_tolerance : float
            Already-realized sample-local curve and topology tolerance in
            arcseconds.
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
        ImportError
            If optional Caustics, Torch, or ContourPy dependencies are
            unavailable.
        ValueError
            If dynamic initial-FOV realization is no larger than the requested
            scale.
        RuntimeError
            If contour or mapping validation fails, or if
            ``_CausticFOVError`` persists through the configured bounded
            expansion schedule.

        Notes
        -----
        Each retry multiplies the current FOV by
        ``fov_expansion_factor`` while retaining the same requested
        ``pixelscale``. Only ``_CausticFOVError`` is caught; structural
        ``RuntimeError`` instances and backend exceptions propagate unchanged.
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
                caustic_curves = _source_geometry._find_all_caustics(
                    lens,
                    center=search_center,
                    fov=current_fov,
                    pixelscale=pixelscale,
                    geometry_tolerance=geometry_tolerance,
                    jacobian_mask_points=jacobian_mask_points,
                )
                return tuple(caustic_curves), current_fov
            except _source_geometry._CausticFOVError as err:
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
        pseudo_caustic_epsilon,
        geometry_tolerance,
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
            Already-realized sample-local critical-curve grid-spacing upper
            bound in arcseconds.
        pseudo_caustic_points : int
            Number of unique image-plane loop vertices per pseudo-caustic.
        pseudo_caustic_epsilon : float
            Already-realized sample-local initial pseudo-caustic loop radius in
            arcseconds.
        geometry_tolerance : float
            Already-realized sample-local curve and topology tolerance in
            arcseconds.
        initial_fov : float or None, optional
            Initial image-plane search FOV in arcseconds, or ``None`` for
            configured/adapter policy.

        Returns
        -------
        _BoundaryGeometry
            Snapshot containing separate total-lens true and mapped pseudo
            boundaries, the successful FOV, requested scale upper bound, and
            pseudo-caustic resolution. True curves have shape ``(P, 2)`` and
            each pseudo curve has shape ``(pseudo_caustic_points + 1, 2)`` in
            source-plane arcseconds, including its repeated endpoint.

        Raises
        ------
        ImportError
            If an optional Caustics, Torch, or ContourPy dependency is
            unavailable.
        ValueError
            If dynamic initial-FOV realization violates the realized scale.
        RuntimeError
            If bounded critical-curve extraction, contour validation, mapped
            closure, or pseudo-caustic convergence fails.

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
            geometry_tolerance=geometry_tolerance,
            initial_fov=initial_fov,
        )
        pseudo_caustic_curves = _source_geometry._trace_pseudo_caustics(
            lens,
            geometry_adapter,
            values,
            num_points=pseudo_caustic_points,
            epsilon=pseudo_caustic_epsilon,
            geometry_tolerance=geometry_tolerance,
        )
        return _source_geometry._BoundaryGeometry(
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
        pseudo_caustic_epsilon,
        geometry_tolerance,
        boundary_tolerance,
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
        pseudo_caustic_epsilon : float
            Already-realized sample-local initial pseudo-caustic loop radius in
            arcseconds.
        geometry_tolerance : float
            Already-realized sample-local curve and topology tolerance in
            arcseconds.
        boundary_tolerance : float
            Already-realized sample-local matched-boundary tolerance in
            arcseconds.

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
        ImportError
            If an optional Caustics, Torch, ContourPy, or Shapely dependency is
            unavailable.
        ValueError
            If a dynamic initial FOV is no larger than its requested grid
            scale.
        RuntimeError
            If snapshot extraction or geometry normalization fails, or if
            displacement and typed topology do not converge within
            ``max_boundary_refinements``.

        Notes
        -----
        Each refinement halves the requested grid scale and doubles the number
        of pseudo-caustic vertices. After the baseline snapshot, each refined
        extraction warm-starts its FOV recovery from the preceding snapshot's
        successful ``critical_curve_fov``. Non-axisymmetric systems compare two
        raw snapshots. Exactly axisymmetric systems first partition contracting
        point caustics across three raw snapshots, then apply the unchanged
        regular-boundary topology and displacement policy. Certified point
        centers must also retain their count and move no farther than
        the already-realized ``boundary_tolerance``, but they do not enter
        boundary uncertainty.
        ``axisymmetry_center(values)`` is used only as a ``None``/non-``None``
        capability gate; its coordinate is not used by the contraction test.
        Axisymmetric certification requires three snapshots and therefore at
        least two completed refinements even though configuration accepts a
        refinement limit of one. Exhaustion diagnostics report the last
        penultimate and current requested scales, pseudo-caustic point counts,
        successful FOVs, displacement, and topology flag.
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
                pseudo_caustic_epsilon=pseudo_caustic_epsilon,
                geometry_tolerance=geometry_tolerance,
                initial_fov=None if previous is None else previous.critical_curve_fov,
            )
            if previous is not None:
                last_previous = previous
            if axisymmetry_center is None:
                if previous is not None:
                    (
                        current,
                        last_uncertainty,
                        last_topology_stable,
                    ) = _source_geometry._compare_boundary_geometry(
                        previous, current, geometry_tolerance=geometry_tolerance
                    )
                    if last_topology_stable and last_uncertainty <= boundary_tolerance:
                        return previous, current, last_uncertainty, refinement
                previous = current
                continue

            if older is not None and previous is not None:
                partition = _source_geometry._partition_axisymmetric_point_caustics(
                    older.caustic_curves,
                    previous.caustic_curves,
                    current.caustic_curves,
                    boundary_tolerance=boundary_tolerance,
                )
                if partition is None:
                    last_uncertainty = np.inf
                    last_topology_stable = False
                else:
                    comparison_previous = _source_geometry._BoundaryGeometry(
                        caustic_curves=partition.previous_curves,
                        pseudo_caustic_curves=previous.pseudo_caustic_curves,
                        critical_curve_fov=previous.critical_curve_fov,
                        pixelscale=previous.pixelscale,
                        pseudo_caustic_points=previous.pseudo_caustic_points,
                        point_caustics=partition.previous_points,
                    )
                    comparison_current = _source_geometry._BoundaryGeometry(
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
                    ) = _source_geometry._compare_boundary_geometry(
                        comparison_previous,
                        comparison_current,
                        geometry_tolerance=geometry_tolerance,
                    )
                    point_centers_stable = len(comparison_previous.point_caustics) == len(
                        comparison_current.point_caustics
                    ) and all(
                        np.linalg.norm(previous_point - current_point) <= boundary_tolerance
                        for previous_point, current_point in zip(
                            comparison_previous.point_caustics,
                            comparison_current.point_caustics,
                            strict=True,
                        )
                    )
                    last_topology_stable = last_topology_stable and point_centers_stable
                    if last_topology_stable and last_uncertainty <= boundary_tolerance:
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
            f"configured_pseudo_caustic_epsilon={self.pseudo_caustic_epsilon} arcsec, "
            "pseudo_caustic_epsilon_fraction="
            f"{self.pseudo_caustic_epsilon_fraction}, "
            f"realized_pseudo_caustic_epsilon={pseudo_caustic_epsilon} arcsec, "
            f"configured_geometry_tolerance={self.geometry_tolerance} arcsec, "
            f"geometry_tolerance_fraction={self.geometry_tolerance_fraction}, "
            f"realized_geometry_tolerance={geometry_tolerance} arcsec, "
            f"configured_boundary_tolerance={self.boundary_tolerance} arcsec, "
            f"boundary_tolerance_fraction={self.boundary_tolerance_fraction}, "
            f"realized_boundary_tolerance={boundary_tolerance} arcsec, "
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
            Number of completed factor-of-two boundary-refinement steps.
        region : shapely.Polygon or shapely.MultiPolygon
            Complete supported strong-lensing source region with coordinates
            in arcseconds and area in square arcseconds.
        angular_settings : tuple
            Five values in this exact order: characteristic angular scale or
            ``None`` when no fractions are enabled, realized initial pixel
            scale, realized pseudo-caustic epsilon, realized geometry
            tolerance, and realized boundary tolerance. Every numerical value
            is in arcseconds.

        Raises
        ------
        ImportError
            If an optional Caustics, Torch, ContourPy, or Shapely dependency is
            unavailable.
        KeyError
            If a required flattened lens or source-redshift input is absent.
        TypeError
            If a realized backend scalar cannot be converted as required.
        ValueError
            If realized lens, adapter, or FOV/scale inputs violate their owned
            domains.
        NotImplementedError
            If a realized EPL component has ``t > 1`` and therefore lies
            outside the source-region certification tier. The propagated
            message advises restricting ``t`` or using the image node.
        RuntimeError
            If critical-curve or boundary certification exhausts, or topology
            construction collapses.

        Notes
        -----
        This method builds exactly one shared lens system for the entire
        realization and threads its total lens, adapter, and values through
        certification without caching sample-local objects on the node.
        """
        lens, geometry_adapter, adapter_values = _lens_system._build_lens_system(
            self.lens,
            cosmology=self.cosmology,
            values=values,
        )
        _source_geometry._validate_source_geometry_support(self.lens, lens)
        angular_settings = self._realized_angular_settings(
            geometry_adapter,
            adapter_values,
            sample_index=sample_index,
        )
        (
            characteristic_scale,
            realized_pixelscale,
            realized_pseudo_caustic_epsilon,
            realized_geometry_tolerance,
            realized_boundary_tolerance,
        ) = angular_settings
        previous_geometry, geometry, uncertainty, refinements = (
            self._certified_boundary_geometry_for_one_lens(
                lens,
                geometry_adapter,
                adapter_values,
                sample_index=sample_index,
                pixelscale=realized_pixelscale,
                pseudo_caustic_epsilon=realized_pseudo_caustic_epsilon,
                geometry_tolerance=realized_geometry_tolerance,
                boundary_tolerance=realized_boundary_tolerance,
            )
        )
        region = _source_geometry._build_strong_lensing_region(
            geometry.caustic_curves,
            geometry.pseudo_caustic_curves,
            geometry_tolerance=realized_geometry_tolerance,
        )
        return (
            geometry_adapter,
            adapter_values,
            previous_geometry,
            geometry,
            uncertainty,
            refinements,
            region,
            angular_settings,
        )

    def _sample_certified_position_for_one_lens(
        self,
        region,
        rng,
        geometry_adapter,
        adapter_values,
        previous_geometry,
        geometry,
        uncertainty,
        refinements,
        *,
        sample_index,
        geometry_tolerance,
        geometry_settings,
    ):
        """Sample one source position within a bounded certification budget.

        Parameters
        ----------
        region : shapely.Polygon or shapely.MultiPolygon
            Complete repaired strong-lensing source region with coordinates in
            arcseconds.
        rng : numpy.random.Generator
            Sample-local random generator used for every proposal attempt.
        geometry_adapter : _GeometryAdapter
            Realized aggregate geometry adapter used for signed image counts.
        adapter_values : Mapping
            Recursive realized values consumed by ``geometry_adapter``.
        previous_geometry : _BoundaryGeometry
            Penultimate certified boundary snapshot.
        geometry : _BoundaryGeometry
            Final certified boundary snapshot, including point-caustic
            exclusions.
        uncertainty : float
            Maximum matched-boundary displacement in arcseconds.
        refinements : int
            Number of completed factor-of-two boundary-refinement steps.
        sample_index : int
            Zero-based graph sample index included in diagnostics.
        geometry_tolerance : float
            Already-realized sample-local curve and topology tolerance in
            arcseconds, used for boundary-clearance normalization.
        geometry_settings : Mapping
            Realized and configured source-geometry settings included in
            proposal-exhaustion diagnostics.

        Returns
        -------
        source_x : float
            Accepted source-plane x coordinate in arcseconds.
        source_y : float
            Accepted source-plane y coordinate in arcseconds.
        strong_lensing_area : float
            Area of the complete repaired region in square arcseconds.
        sampling_attempts : int
            Cumulative coordinate-pair draws, including polygon,
            exact-point-caustic, and narrow-boundary rejections.
        expected_num_images : int
            Final certified regular-image count.
        source_boundary_clearance : float
            Accepted source's nearest regular-boundary distance in arcseconds.

        Raises
        ------
        RuntimeError
            If proposal sampling exhausts, penultimate and final image counts
            disagree, or narrow-boundary rejections consume ``max_attempts``.

        Notes
        -----
        A count disagreement is immediately fatal. Only candidates whose
        clearance is no greater than ``uncertainty`` are redrawn, and every
        coordinate pair consumes the one cumulative ``max_attempts`` budget.
        The returned area remains that of the complete repaired region rather
        than the clearance-conditioned accepted-position support. Exhaustion
        propagates before ``compute`` can pass any results to ``_save_results``.
        """
        remaining_attempts = self.max_attempts
        total_attempts = 0
        narrow_boundary_rejections = 0
        last_clearance = None

        def exhaustion_message():
            settings = ", ".join(f"{name}={value}" for name, value in geometry_settings.items())
            return (
                "Unable to sample a certified source position for "
                f"{self._lens_identifier(sample_index)} after {self.max_attempts} attempts; "
                f"narrow_boundary_rejections={narrow_boundary_rejections}, "
                f"last_source_boundary_clearance={last_clearance} arcsec, "
                f"boundary_uncertainty={uncertainty} arcsec, "
                f"penultimate pixelscale={previous_geometry.pixelscale} arcsec, "
                "penultimate pseudo_caustic_points="
                f"{previous_geometry.pseudo_caustic_points}, "
                f"final pixelscale={geometry.pixelscale} arcsec, "
                f"final pseudo_caustic_points={geometry.pseudo_caustic_points}, "
                f"boundary_refinements={refinements}, {settings}."
            )

        while remaining_attempts > 0:
            try:
                source_x, source_y, area, candidate_attempts = _source_geometry._sample_position(
                    region,
                    rng,
                    max_attempts=remaining_attempts,
                    lens_identifier=self._lens_identifier(sample_index),
                    geometry_settings=geometry_settings,
                    excluded_points=geometry.point_caustics,
                )
            except RuntimeError as err:
                if narrow_boundary_rejections == 0:
                    raise
                total_attempts += remaining_attempts
                remaining_attempts = 0
                raise RuntimeError(exhaustion_message()) from err
            total_attempts += candidate_attempts
            remaining_attempts -= candidate_attempts
            previous_count = geometry_adapter.expected_num_images(
                source_x,
                source_y,
                values=adapter_values,
                caustic_curves=previous_geometry.caustic_curves,
                pseudo_caustic_curves=previous_geometry.pseudo_caustic_curves,
            )
            final_count = geometry_adapter.expected_num_images(
                source_x,
                source_y,
                values=adapter_values,
                caustic_curves=geometry.caustic_curves,
                pseudo_caustic_curves=geometry.pseudo_caustic_curves,
            )
            clearance = _source_geometry._source_boundary_clearance(
                source_x,
                source_y,
                geometry,
                geometry_tolerance,
            )
            if previous_count != final_count:
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
            if clearance > uncertainty:
                return source_x, source_y, area, total_attempts, final_count, clearance
            narrow_boundary_rejections += 1
            last_clearance = clearance

        raise RuntimeError(exhaustion_message())

    def compute(self, graph_state, rng_info=None, **kwargs):
        """Sample one uniform strong-lensing source position per graph sample.

        A fixed number of sub-seeds is drawn from ``rng_info`` (or the node-owned
        fallback generator) before any rejection sampling. Variable rejection
        counts for one lens therefore cannot perturb later lens samples.

        Parameters
        ----------
        graph_state : GraphState
            State containing the realized node inputs. Only after every sample
            succeeds is it offered the nine computed outputs through
            ``_save_results``; eligible nonfixed entries are updated and
            pre-fixed output entries are preserved.
        rng_info : numpy.random.Generator, optional
            Caller-owned random generator. When omitted, the node-owned generator
            configured by ``seed`` is used.
        **kwargs : dict, optional
            Call-local overrides keyed by registered node input name. These
            input overrides are not persisted to ``graph_state``.

        Returns
        -------
        results : list
            Nine values in this exact order:

            1. ``source_x``, source-plane arcseconds;
            2. ``source_y``, source-plane arcseconds;
            3. ``strong_lensing_area``, complete geometric region area in
               square arcseconds;
            4. ``sampling_attempts``, cumulative coordinate-pair draw count;
            5. ``expected_num_images``, regular-image count;
            6. ``critical_curve_fov``, successful image-plane FOV in
               arcseconds;
            7. ``boundary_uncertainty``, matched-boundary displacement in
               arcseconds;
            8. ``source_boundary_clearance``, nearest typed-boundary
               distance in arcseconds;
            9. ``boundary_refinements``, completed factor-of-two refinement
               step count.

            Each value is a NumPy scalar when
            ``graph_state.num_samples == 1`` and a NumPy array with shape
            ``(S,)`` for ``S`` graph samples otherwise.

        Raises
        ------
        ImportError
            If an optional Caustics, Torch, ContourPy, or Shapely dependency is
            unavailable.
        KeyError
            If a required realized source or flattened lens input is absent.
        TypeError
            If a realized scalar cannot be normalized by its owning lens or
            backend operation.
        ValueError
            If realized lens parameters, redshifts, adapter geometry, or
            dynamic FOV/scale relations violate their owned domains.
        NotImplementedError
            If any realized EPL component has ``t > 1``; the source-position
            node cannot yet certify that source-region tier.
        RuntimeError
            If bounded critical-curve extraction, boundary certification, or
            rejection sampling exhausts, or the sampled source fails
            penultimate/final count and clearance certification.

        Notes
        -----
        Registered ``**kwargs`` input overrides apply only to this call and are
        not persisted to the ``GraphState``. Only after every sample succeeds
        are the nine newly computed outputs passed to ``_save_results``. Its
        ``GraphState.set`` calls update eligible nonfixed output entries while
        preserving pre-fixed ones. The returned ``results`` still contains the
        newly computed value for every output, including an output whose
        pre-fixed graph entry was preserved. The caller generator takes
        precedence over the fallback generator. Exactly ``S`` unsigned 64-bit
        sub-seeds in ``[0, 2**63)`` are drawn before any per-sample geometry or
        rejection work, then one independent generator is created per sample
        so variable rejection counts cannot perturb later samples. The chosen
        parent generator has therefore consumed all ``S`` seeds even if a later
        sample raises.

        Each proposal must lie strictly inside the repaired union and must not
        equal a certified point caustic exactly. Polygon and exact-point
        rejections occur within the proposal sampler. If the penultimate and
        final signed-winding image counts disagree, the method raises
        immediately without retrying. If the counts agree but final
        regular-boundary clearance is no greater than boundary uncertainty, a
        new candidate is drawn from the remaining total ``max_attempts``
        budget. ``sampling_attempts`` includes every coordinate-pair draw from
        polygon, exact-point-caustic, and narrow-boundary rejection. The
        accepted-position distribution is therefore clearance-conditioned,
        while ``strong_lensing_area`` retains the complete repaired geometric
        region area. Exhaustion and count mismatch both occur before any of
        this call's results are passed to ``_save_results``.
        Sampling-exhaustion context includes
        ``fov``, configured/fractional/realized settings for pixel scale,
        pseudo-caustic epsilon, geometry tolerance, and boundary tolerance,
        ``max_fov_expansions``, pseudo-caustic resolution, final boundary
        settings, and ``max_boundary_refinements``. The sampler reports
        ``max_attempts`` separately; the context mapping does not include
        ``fov_expansion_factor``.
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
                name: _runtime._sample_value(value, sample_index, num_samples)
                for name, value in input_values.items()
            }
            (
                geometry_adapter,
                adapter_values,
                previous_geometry,
                geometry,
                uncertainty,
                refinements,
                region,
                angular_settings,
            ) = self._region_for_one_lens(
                values,
                sample_index=sample_index,
            )
            (
                characteristic_scale,
                realized_pixelscale,
                realized_pseudo_caustic_epsilon,
                realized_geometry_tolerance,
                realized_boundary_tolerance,
            ) = angular_settings
            sample_rng = np.random.default_rng(sample_seed)
            geometry_settings = {
                "fov": self.fov,
                "characteristic_scale": characteristic_scale,
                "configured_pixelscale": self.pixelscale,
                "pixelscale_fraction": self.pixelscale_fraction,
                "realized_pixelscale": realized_pixelscale,
                "max_fov_expansions": self.max_fov_expansions,
                "pseudo_caustic_points": self.pseudo_caustic_points,
                "configured_pseudo_caustic_epsilon": self.pseudo_caustic_epsilon,
                "pseudo_caustic_epsilon_fraction": self.pseudo_caustic_epsilon_fraction,
                "realized_pseudo_caustic_epsilon": realized_pseudo_caustic_epsilon,
                "configured_geometry_tolerance": self.geometry_tolerance,
                "geometry_tolerance_fraction": self.geometry_tolerance_fraction,
                "realized_geometry_tolerance": realized_geometry_tolerance,
                "configured_boundary_tolerance": self.boundary_tolerance,
                "boundary_tolerance_fraction": self.boundary_tolerance_fraction,
                "realized_boundary_tolerance": realized_boundary_tolerance,
                "max_boundary_refinements": self.max_boundary_refinements,
                "final_pixelscale": geometry.pixelscale,
                "final_pseudo_caustic_points": geometry.pseudo_caustic_points,
                "final_critical_curve_fov": geometry.critical_curve_fov,
                "boundary_uncertainty": uncertainty,
                "boundary_refinements": refinements,
            }
            (
                source_x[sample_index],
                source_y[sample_index],
                areas[sample_index],
                attempts[sample_index],
                expected_num_images[sample_index],
                source_boundary_clearance[sample_index],
            ) = self._sample_certified_position_for_one_lens(
                region,
                sample_rng,
                geometry_adapter,
                adapter_values,
                previous_geometry,
                geometry,
                uncertainty,
                refinements,
                sample_index=sample_index,
                geometry_tolerance=realized_geometry_tolerance,
                geometry_settings=geometry_settings,
            )

            critical_curve_fov[sample_index] = geometry.critical_curve_fov
            boundary_uncertainty[sample_index] = uncertainty
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
        include the lens-redshift setter ``z_l``.
    cosmology : caustics.Cosmology
        Fixed cosmology inherited by every realized lens.
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
    lens : CausticsLensSpec
        Stored registered specification used to construct a fresh realized lens
        tree for every graph sample.
    cosmology : caustics.Cosmology
        Fixed node-owned cosmology excluded from the sampled ``GraphState``.
    max_images : int
        Stored fixed image-output width and maximum accepted active count.
    min_images : int
        Stored minimum count required after bounded recovery.
    fov_multiplier : float
        Stored dimensionless multiplier applied to each realized initial FOV.
    pixelscale : float
        Stored absolute requested grid-spacing upper bound in arcseconds.
    pixelscale_fraction : float or None
        Stored dimensionless relative-spacing fraction, or ``None``.
    epsilon : float
        Stored absolute source-residual tolerance in arcseconds.
    epsilon_fraction : float or None
        Stored dimensionless relative-residual fraction, or ``None``.
    max_depth : int
        Stored Caustics global-search tree depth.
    max_fov_expansions : int
        Stored maximum number of outer FOV expansion steps.
    fov_expansion_factor : float
        Stored dimensionless FOV expansion multiplier.
    max_pixelscale_refinements : int
        Stored maximum number of outer requested-scale refinement steps.
    pixelscale_refinement_factor : float
        Stored dimensionless requested-scale refinement multiplier.
    source_redshift : AttributeIndicator
        Dynamic graph input for dimensionless source redshift.
    source_x : AttributeIndicator
        Dynamic graph input for finite source-plane x position in arcseconds.
    source_y : AttributeIndicator
        Dynamic graph input for finite source-plane y position in arcseconds.
    fov : AttributeIndicator
        Dynamic graph input realizing to an image-plane FOV in arcseconds or
        ``None`` for adapter-derived extent.
    expected_num_images : AttributeIndicator
        Dynamic graph input realizing to an integer target or ``None``.
    lens_<path>_<field> : AttributeIndicator
        Dynamic graph inputs flattened from ``lens``. The root path begins
        ``lens`` (including ``lens_z_l``); nested ``SinglePlane`` child indices
        are inserted before each atomic field name.
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
        Graph output for the final successful global variant's actual spacing
        in arcseconds.
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
    expansion/refinement counters. Newly computed fixed-width outputs use the
    padding and sentinel conventions documented above. Active images retain no
    local deduplication and are ordered by increasing delay, then x, then y.

    The exact output order is ``num_images``, ``image_x``, ``image_y``,
    ``macro_magnifications``, ``time_delays``, ``image_count_deficit``,
    ``solver_fov``, ``solver_pixelscale``, ``solver_attempts``,
    ``solver_fov_expansions``, and ``solver_pixelscale_refinements``. Image
    coordinates and delays are NaN-padded, magnifications are zero-padded, and
    ``image_count_deficit`` is ``-1`` when no expectation is supplied. All
    count and diagnostic outputs are NumPy scalars for one sample while the
    four image outputs have shape ``(M,)``, where ``M = max_images``. For
    ``S > 1`` those shapes become ``(S,)`` and ``(S, M)`` respectively. All
    eleven newly computed results are passed in that order to ``_save_results``
    only after every sample succeeds. Its ``GraphState.set`` calls update
    eligible nonfixed output entries and preserve pre-fixed output entries. A
    successful return still contains every newly computed value, including one
    whose pre-fixed graph entry was preserved. Image solving is deterministic
    for realized inputs and does not consume caller RNG state.

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
        "macro_convergences",
        "macro_shear",
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
        cosmology,
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
        max_depth=50,
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
            the dimensionless lens-redshift setter ``z_l``.
        cosmology : caustics.Cosmology
            Fixed cosmology inherited by every realized lens and excluded from
            graph parameters.
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
        are registered directly. The fixed ``cosmology`` is stored on the node
        rather than registered. Lens parameters use ``lens_<field>`` at the
        root and positional path segments inside nested planes. The eleven
        public outputs are registered in exact order: ``num_images``,
        ``image_x``, ``image_y``, ``macro_magnifications``, ``time_delays``,
        ``image_count_deficit``, ``solver_fov``, ``solver_pixelscale``,
        ``solver_attempts``, ``solver_fov_expansions``, and
        ``solver_pixelscale_refinements``. Image-node integer validation uses
        the ``(int, numpy.integer)`` contract; booleans follow Python's integer
        subclass semantics before the stated range checks. Graph setters and
        the identity-stored cosmology are not realized or physically validated
        by this constructor; those checks remain with per-sample solving and
        the Caustics backend.
        """
        if not isinstance(lens, CausticsLensSpec):
            raise TypeError("lens must be a CausticsLensSpec.")
        _lens_system._validate_root_lens_spec(lens)
        lens_graph_inputs = _lens_system._lens_graph_inputs(lens)

        pixelscale_fraction = _runtime._validate_optional_positive_fraction(
            "pixelscale_fraction",
            pixelscale_fraction,
        )
        epsilon_fraction = _runtime._validate_optional_positive_fraction(
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

        Raises
        ------
        RuntimeError
            If the Caustics global solver reports a numerical failure,
            including the exact singular failures classified by ``_solve_one``.
        IndexError
            If the Caustics global solver reports an indexing failure,
            including the exact empty-candidate failure classified by
            ``_solve_one``.

        Notes
        -----
        This helper is the architectural boundary for exactly one Caustics
        invocation. A shifted numerical grid center changes only the search
        grid; returned coordinates are never translated. Paired output shapes,
        types, and finiteness are trusted Caustics postconditions. It catches
        and classifies no exception itself: the nested ``attempt`` closure owns
        the two narrow retry predicates, and every other backend exception
        propagates unchanged. The returned rows are not translated, sorted,
        deduplicated, or independently validated.
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
        return np.column_stack((_runtime._to_numpy(image_x), _runtime._to_numpy(image_y)))

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
            an ``int`` or ``numpy.integer`` in the configured range; and every
            flattened lens-specification graph input.

        Returns
        -------
        image_x : numpy.ndarray, shape (I,)
            Active image-plane x positions in arcseconds.
        image_y : numpy.ndarray, shape (I,)
            Active image-plane y positions in arcseconds.
        macro_magnifications : numpy.ndarray, shape (I,)
            Absolute dimensionless macro-magnifications.
        macro_convergences: numpy.ndarray, shape (I,)
            Macro convergence at each image position
        macro_shears: numpy.ndarray, shape (I,)
            Macro shear at each image position
        time_delays : numpy.ndarray, shape (I,)
            Observer-frame relative delays in days, normalized to start at zero.
        diagnostics : dict
            Six scalar entries. ``image_count_deficit`` is expected minus
            recovered count, or ``-1`` without an expectation;
            ``solver_fov`` is the final accepted or bounded-deficit FOV in
            arcseconds; ``solver_pixelscale`` is the actual spacing of the
            final successful global grid variant in arcseconds;
            ``solver_attempts`` counts global calls and executed targeted
            batches; and ``solver_fov_expansions`` and
            ``solver_pixelscale_refinements`` count outer steps only.

        Notes
        -----
        All four arrays use the same deterministic ordering: increasing delay,
        then image x, then image y. Delays are normalized by subtracting their
        minimum before the shared sort. Padding to ``max_images`` is performed
        by ``compute``. Magnifications are absolute, so signed image parity is
        not retained.

        A ``None`` FOV uses the realized adapter's total-lens extent; an
        explicit realized FOV overrides that extent. Either value is multiplied
        by ``fov_multiplier`` to form ``initial_fov``. Realized pixel scale and
        epsilon are fixed once from the total adapter. Every global call and
        residual certification uses that fixed epsilon; targeted seed radius
        is ``min(realized_epsilon, actual_grid_spacing)``, while recovery-point
        neighborhood occupancy and root locality use the actual grid spacing.
        Each outer attempt is independent, so its global coordinates replace
        rather than merge with any earlier attempt; the solver does not retain
        the best earlier count. Supplemental roots are local to a successful,
        nonempty, deficient attempt and run only when the adapter supplies
        explicit recovery points with at least one unoccupied neighborhood.
        They are appended without local deduplication, so every returned array
        and count retains duplicates supplied by global or targeted solving.

        The divisions-plus-one variant changes actual spacing; the half-cell
        numerical-center shift retains base divisions and base spacing. Both
        add global attempts, and neither changes outer recovery counters. The
        numerical shift leaves physical lens values and returned coordinates
        unchanged. All bounded FOV expansions run before requested-scale
        refinements. After retryable exhaustion, ``None`` coordinates are
        normalized to an empty array; recovery below ``min_images`` raises,
        while a bounded deficit at or above that minimum may be returned. Such
        a nonzero bounded deficit can occur only when an explicit expected
        count is configured, because the no-expectation contract completes as
        soon as ``min_images`` is reached. Exceptions from Caustics or Torch
        that do not satisfy the two exact retry classifiers propagate
        unchanged. This helper returns realization-local values and does not
        persist anything to ``GraphState``.

        Raises
        ------
        ImportError
            If optional Caustics runtime dependencies are unavailable.
        KeyError
            If a required realized input is absent.
        TypeError
            If an owned source-coordinate or explicit-FOV scalar cannot be
            normalized, or a downstream constructor/backend reports a type
            failure.
        ValueError
            If source coordinates, FOV, expected count, redshifts, or the
            realized lens specification violate their owned domains.
        RuntimeError
            If a recovered count exceeds its expectation or ``max_images``,
            or bounded recovery exhausts below ``min_images``.
        IndexError
            If Caustics raises an indexing error outside the exact recognized
            empty-candidate retry classifier.
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

        lens, geometry_adapter, adapter_values = _lens_system._build_lens_system(
            self.lens,
            cosmology=self.cosmology,
            values=values,
        )
        _, torch = _runtime._import_caustics_dependencies()

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
            and multiplier are not separate diagnostic fields. The text also
            omits ``min_images``, source coordinates, and the numerical search
            center. ``latest_retryable_error`` is sticky recovery context, not
            a returned output diagnostic.
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

            Notes
            -----
            Expected-count overflow is checked before maximum-count overflow.
            Because a configured expectation cannot exceed ``max_images``, the
            distinct maximum-overflow branch is reachable only when no
            expectation is present. Coordinates are counted by row without
            deduplication.
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
            ImportError
                If targeted recovery reaches the optional Caustics
                root-refinement import and it is unavailable.
            RuntimeError
                If a recovered count exceeds the expected or maximum count.
            IndexError
                If a global or targeted backend call raises an indexing error
                outside the exact recognized empty-candidate classifier.

            Notes
            -----
            The global variant order is the base grid, a divisions-plus-one
            parity grid, then a half-cell shift in both numerical-center
            coordinates at the base division count. Only a ``RuntimeError``
            whose case-insensitive text contains ``linalg.solve`` and either
            ``input matrix is singular`` or ``singular u`` advances that inner
            sequence. Three such failures exhaust naturally through the loop
            ``else``. An ``IndexError`` whose case-insensitive text contains
            ``index 0 is out of bounds`` immediately returns ``(None, False)``
            for retry only by the surrounding bounded outer schedule. A
            successful but deficient global result also ends the variant
            sequence. Unrelated exceptions propagate unchanged.

            Every actual global Caustics invocation increments
            ``solver_attempts``. Each executed recovery-seed batch adds one
            more attempt, while its internal root-refinement passes do not.
            Actual spacing is updated to ``current_fov / divisions`` for the
            invoked variant. The divisions-plus-one variant changes that
            spacing; the half-cell numerical-center shift uses base divisions
            and therefore retains base spacing. Both add a global attempt, and
            neither changes the outer expansion/refinement counters.

            The adapter's recovery points are queried after every successful
            global solve, even a complete one. Targeted work itself runs only
            after a successful, nonempty, deficient global result with explicit
            points and at least one neighborhood for which every global image
            is strictly farther than actual grid spacing. Seed circles are
            raytraced for all adapter recovery points before selecting the
            empty-neighborhood seeds. Their radius is the smaller of fixed
            realized epsilon and actual grid spacing. Residual certification
            uses a strict fixed-epsilon bound, while neighborhood occupancy and
            root locality use actual grid spacing, inclusively for occupancy
            and locality. Each deterministic seed circle has 256 uniformly
            spaced angular locations, and active seeds undergo exactly eight
            root-refinement passes in one counted batch. Retryable targeted
            failures leave the global result deficient. Supplemental roots are
            appended without deduplication. Targeted roots and global
            coordinates are local to this attempt; every later outer call
            replaces them rather than preserving the best earlier count.

            Numerical-center shifts never modify physical lens values or
            translate returned image coordinates. If all outer calls remain
            retryable, the enclosing solver's reachable
            ``coordinates is None`` branch converts the result to shape
            ``(0, 2)`` before the positive ``min_images`` policy raises. The
            enclosing schedule completes every FOV expansion before beginning
            requested-scale refinement. Seed-circle raytraces occur outside
            the targeted-refinement exception classifier; their failures
            propagate and do not increment ``solver_attempts``. One entered
            root-refinement batch increments the counter once, not once per
            internal refinement pass.
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
                    is_singular_error = _image_recovery._is_singular_forward_raytrace_error(error)
                    is_empty_candidate_error = _image_recovery._is_retryable_forward_raytrace_error(error)
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

            empty_neighborhoods = _image_recovery._recovery_neighborhoods_are_empty(
                coordinates,
                recovery_points,
                current_grid_pixelscale,
            )
            if not np.any(empty_neighborhoods):
                return coordinates, False

            seeds = _image_recovery._recovery_image_seeds(
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
                recovery_coordinates = _image_recovery._refine_image_seeds(
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
                is_singular_error = _image_recovery._is_singular_forward_raytrace_error(error)
                is_empty_candidate_error = _image_recovery._is_retryable_forward_raytrace_error(error)
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
        convergences = lens.convergence(image_x_tensor, image_y_tensor)
        shear1, shear2 = lens.shear(image_x_tensor, image_y_tensor)

        image_x = coordinates[:, 0]
        image_y = coordinates[:, 1]
        magnifications = _runtime._to_numpy(magnifications)
        time_delays = _runtime._to_numpy(time_delays)
        convergences = _runtime._to_numpy(convergences)
        shear1 = _runtime._to_numpy(shear1)
        shear2 = _runtime._to_numpy(shear2)
        shears = np.sqrt(shear1**2 + shear2**2)

        time_delays = time_delays - np.min(time_delays)
        # np.lexsort uses the final key as the primary key: delay, then x, then y.
        order = np.lexsort((image_y, image_x, time_delays))
        return (
            image_x[order],
            image_y[order],
            magnifications[order],
            time_delays[order],
            convergences[order],
            shears[order],
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
        """Solve every sampled lens and return fixed-width numeric outputs.

        Parameters
        ----------
        graph_state : GraphState
            State containing realized node inputs. Only after every sample
            succeeds is it offered all eleven computed outputs through
            ``_save_results``; eligible nonfixed entries are updated and
            pre-fixed output entries are preserved.
        rng_info : object, optional
            Ignored. Image solving is deterministic for realized inputs.
        **kwargs : dict, optional
            Call-local overrides keyed by registered node input name. These
            input overrides are not persisted to ``graph_state``.

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
            6. ``convergences``, NaN-padded;
            7. ``shears``, NaN-padded;
            8. ``image_count_deficit``, expected minus recovered count or
               ``-1`` without an expectation;
            9. ``solver_fov``, final accepted or bounded-deficit FOV in
               arcseconds;
            10. ``solver_pixelscale``, final successful global variant's actual
               grid spacing in arcseconds;
            11. ``solver_attempts``, global calls plus executed recovery-seed
               batches;
            12. ``solver_fov_expansions``, outer FOV steps only;
            13. ``solver_pixelscale_refinements``, outer requested-scale
                steps only.

            For one sample, count and diagnostic outputs are NumPy scalars and
            each fixed-width image output has shape ``(M,)``, where
            ``M = max_images``. For ``S > 1``, count and diagnostic outputs
            have shape ``(S,)``, and image outputs have shape ``(S, M)``.

        Raises
        ------
        ImportError
            If an optional Caustics or Torch runtime, model component, or
            targeted root-refinement dependency is unavailable.
        KeyError
            If a required realized source, solver-control, redshift, or
            flattened lens input is absent.
        TypeError
            If an owned source-coordinate or explicit-FOV scalar cannot be
            normalized, or lens/backend construction reports a type failure.
        ValueError
            If realized source coordinates, expected count, FOV, lens
            geometry, redshifts, or the initial FOV/scale relation violate
            their owned domains.
        RuntimeError
            If image count exceeds its expectation or ``max_images``, bounded
            recovery exhausts below ``min_images``, or an unclassified backend
            runtime failure propagates.
        IndexError
            If Caustics raises an indexing error outside the exact recognized
            empty-candidate retry classifier.

        Notes
        -----
        The method deletes ``rng_info`` without inspecting or consuming it; no
        generator is created. Active image rows preserve the solver's lack of
        local deduplication and its delay/x/y ordering. Float image arrays are
        initialized with NaN for x, y, and delay and zero for absolute
        magnification, so only the first ``num_images`` entries are active.
        ``image_count_deficit`` alone uses ``-1`` as the no-expectation
        sentinel.

        Single-sample count and diagnostic outputs are NumPy scalar values;
        multi-sample count/counter arrays have integer dtype and angular
        diagnostics have floating dtype. Registered-name ``kwargs`` override
        input values for this call but are not written back as graph inputs.
        Only after every sample solves successfully are all eleven newly
        computed values passed to ``_save_results`` in ``_OUTPUTS`` order. Its
        ``GraphState.set`` calls update eligible nonfixed output entries while
        preserving pre-fixed ones. The returned ``results`` still contains the
        newly computed value for every output, including an output whose
        pre-fixed graph entry was preserved. Other unclassified exceptions
        from construction, adapters, Torch, Caustics, NumPy conversion,
        magnification, or delay evaluation propagate unchanged.
        """
        del rng_info  # The solver is deterministic for realized input parameters.
        input_values = self._build_inputs(graph_state, **kwargs)
        num_samples = graph_state.num_samples

        counts = np.empty(num_samples, dtype=int)
        image_x = np.full((num_samples, self.max_images), np.nan)
        image_y = np.full((num_samples, self.max_images), np.nan)
        magnifications = np.zeros((num_samples, self.max_images), dtype=float)
        time_delays = np.full((num_samples, self.max_images), np.nan)
        convergences = np.full((num_samples, self.max_images), np.nan)
        shears = np.full((num_samples, self.max_images), np.nan)
        image_count_deficit = np.empty(num_samples, dtype=int)
        solver_fov = np.empty(num_samples, dtype=float)
        solver_pixelscale = np.empty(num_samples, dtype=float)
        solver_attempts = np.empty(num_samples, dtype=int)
        solver_fov_expansions = np.empty(num_samples, dtype=int)
        solver_pixelscale_refinements = np.empty(num_samples, dtype=int)

        for sample_index in range(num_samples):
            current_values = {
                name: _runtime._sample_value(value, sample_index, num_samples)
                for name, value in input_values.items()
            }
            (
                current_x,
                current_y,
                current_mu,
                current_delay,
                current_convergence,
                current_shear,
                diagnostics,
            ) = self._solve_one(current_values)
            count = len(current_x)
            counts[sample_index] = count
            image_x[sample_index, :count] = current_x
            image_y[sample_index, :count] = current_y
            magnifications[sample_index, :count] = current_mu
            time_delays[sample_index, :count] = current_delay
            convergences[sample_index, :count] = current_convergence
            shears[sample_index, :count] = current_shear
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
                convergences[0],
                shears[0],
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
                convergences,
                shears,
                image_count_deficit,
                solver_fov,
                solver_pixelscale,
                solver_attempts,
                solver_fov_expansions,
                solver_pixelscale_refinements,
            ]

        self._save_results(results, graph_state)
        return results

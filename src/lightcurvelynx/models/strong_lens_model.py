"""Unresolved strong-lens models for a single physical source.

An unresolved system combines all static macro-images into one measured flux.
For absolute dimensionless magnifications ``mu_i`` and observer-frame arrival
delays in days, this module uses the convention

::

    F(t, wavelength) = sum_i mu_i * F_source(
        t - (delay_i - min(delay)), wavelength
    )

Thus the earliest active image has zero relative delay, and later images are
evaluated at earlier source times. An SED child owns its configured framework
redshift conversion and effects attached directly to it. A ``BandfluxModel``
instead produces observer-frame bandfluxes, with no framework redshift
conversion by either child or wrapper. Effects marked as rest-frame and added
through the wrapper are delegated to the child: an SED child applies them in
its rest-frame pipeline, whereas a ``BandfluxModel`` applies them through its
observer-frame band-pass API. Observer-frame effects added to the wrapper are
applied once after the image sum.

The fixed-width ``macro_magnifications`` and ``time_delays`` arrays and the
``num_images`` count from
:class:`~lightcurvelynx.models.caustics_models.CausticsLensImageNode` can be
connected directly to :class:`UnresolvedStrongLensModel` when each realization
has at least two images, as guaranteed by the lens node's default
``min_images=2``. A lens node configured with ``min_images=1`` can produce a
one-image realization that this wrapper rejects. Only the arrays' leading
active prefix is evaluated, so the lens node's inactive zero and NaN padding
does not contribute to the unresolved flux.
"""

import numpy as np

from lightcurvelynx.models.multi_object_model import MultiObjectModel
from lightcurvelynx.models.physical_model import BandfluxModel, BasePhysicalModel


class UnresolvedStrongLensModel(MultiObjectModel):
    """Wrap one physical source as an unresolved static macro-lens system.

    The source is evaluated at observer-frame times shifted by each active
    macro-image's relative arrival delay. Those evaluations are scaled by
    absolute macro-magnifications and summed before wrapper observer-frame
    effects are applied.

    Parameters
    ----------
    source_model : BasePhysicalModel
        Physical source whose SED or bandflux evaluation is lensed.
    macro_magnifications : parameter
        Setter for fixed-width absolute, dimensionless image magnifications.
        Realized values have shape ``(I,)`` for one sample and ``(S, I)`` for
        ``S`` samples.
    time_delays : parameter
        Setter for fixed-width image arrival delays in observer-frame days,
        with the same realized shapes as ``macro_magnifications``. An arbitrary
        common offset is allowed and removed before evaluation.
    num_images : parameter or None, optional
        Setter for the number of active leading entries. ``None`` activates all
        ``I`` entries; otherwise the realized value must satisfy
        ``2 <= num_images <= I``.
    node_label : str or None, optional
        Human-readable label for the outer model node.
    **kwargs : dict, optional
        Outer physical-model parameter overrides. By default ``ra``, ``dec``,
        ``redshift``, ``t0``, and ``distance`` are linked to ``source_model``.

    Attributes
    ----------
    source_model : BasePhysicalModel
        Child physical source evaluated for every active macro-image.
    objects : list of BasePhysicalModel
        The one-element child-model list used by ``MultiObjectModel``.
    num_objects : int
        Number of child models, always one.
    apply_redshift : bool
        ``False`` to suppress wrapper redshift conversion. An SED child owns
        its configured framework conversion; a ``BandfluxModel`` remains in
        the observer frame and does not support framework redshift conversion.

    Raises
    ------
    TypeError
        If ``source_model`` is not a ``BasePhysicalModel``.

    Notes
    -----
    Both full fixed-width arrays are float-coerced and checked for
    one-dimensional, equal-length structure before active-prefix truncation.
    Finiteness and value checks then apply only to the active prefix. Thus
    float-coercible inactive padding, including the Caustics zero and NaN
    sentinels, is ignored by active-value validation, but non-coercible padding
    or invalid full-array shape and length are not ignored. Active image pairs
    are normalized by the minimum delay and placed in ascending delay order
    with stable ordering for equal delays.

    Effects attached directly to an SED child run in that child's configured
    rest- or observer-frame pipeline for each shifted image. Effects marked as
    rest-frame and delegated to a ``BandfluxModel`` instead run through its
    observer-frame ``apply_bandflux`` pipeline because bandflux children have
    no rest-frame conversion. Observer-frame effects retained by the wrapper
    run once on the summed flux at the original observation times; on the
    bandflux path every such effect must implement ``apply_bandflux``. Public
    evaluation over an ``S``-sample state returns ``(S, T, W)`` SEDs or
    ``(S, T)`` bandfluxes, while a one-sample state returns ``(T, W)`` or
    ``(T,)`` respectively.

    Explicit outer metadata overrides are not child overrides: they replace
    the default wrapper linkage and can affect wrapper-level effects, but the
    child continues to use its own parameters for flux evaluation.
    """

    def __init__(
        self,
        source_model: BasePhysicalModel,
        *,
        macro_magnifications,
        time_delays,
        num_images=None,
        node_label=None,
        **kwargs,
    ):
        """Configure an unresolved strong-lens wrapper.

        Parameters
        ----------
        source_model : BasePhysicalModel
            Child physical source to evaluate at every active image delay.
        macro_magnifications : parameter
            Graph setter for fixed-width absolute dimensionless
            macro-magnifications. The parameter is registered without gradient
            support.
        time_delays : parameter
            Graph setter for fixed-width observer-frame image arrival delays in
            days. The parameter is registered without gradient support.
        num_images : parameter or None, optional
            Graph setter for the active leading-image count. ``None`` uses the
            full realized image-array width. The parameter is registered
            without gradient support.
        node_label : str or None, optional
            Human-readable label for the outer model node.
        **kwargs : dict, optional
            Outer ``BasePhysicalModel`` parameters. Missing ``ra``, ``dec``,
            ``redshift``, ``t0``, and ``distance`` setters are linked to the
            corresponding child setters; explicit values remain outer-only
            overrides.

        Raises
        ------
        TypeError
            If ``source_model`` is not a ``BasePhysicalModel``. This check is
            performed before graph links or model parameters are registered.

        Notes
        -----
        The source is retained as the wrapper's only child. The outer
        ``apply_redshift`` flag is set to ``False`` because this wrapper must
        not perform framework redshift conversion. A shifted SED-child
        evaluation applies that child's configured conversion; a
        ``BandfluxModel`` evaluation is already defined in the observer frame
        and applies no framework redshift conversion.
        """
        if not isinstance(source_model, BasePhysicalModel):
            raise TypeError("source_model must be a BasePhysicalModel.")

        kwargs.setdefault("ra", source_model.ra)
        kwargs.setdefault("dec", source_model.dec)
        kwargs.setdefault("redshift", source_model.redshift)
        kwargs.setdefault("t0", source_model.t0)
        kwargs.setdefault("distance", source_model.distance)

        super().__init__([source_model], node_label=node_label, **kwargs)
        self.source_model = source_model

        self.add_parameter(
            "macro_magnifications",
            macro_magnifications,
            description="Absolute macro-image magnifications (unitless).",
            allow_gradient=False,
        )
        self.add_parameter(
            "time_delays",
            time_delays,
            description="Macro-image observer-frame arrival delays (days).",
            allow_gradient=False,
        )
        self.add_parameter(
            "num_images",
            num_images,
            description="Number of active macro-images.",
            allow_gradient=False,
        )

        # This composite overrides the single-state evaluation pipeline. SED
        # children apply their own configured redshift conversion; BandfluxModel
        # children are already observer-frame. The wrapper applies no conversion.
        self.apply_redshift = False

    def minwave(self, graph_state=None):
        """Return the child model's minimum wavelength bound.

        Parameters
        ----------
        graph_state : GraphState or None, optional
            Sampled state forwarded unchanged to the child model.

        Returns
        -------
        float or None
            Child minimum wavelength bound in Angstroms, or ``None`` when
            unbounded.
        """
        return self.source_model.minwave(graph_state=graph_state)

    def maxwave(self, graph_state=None):
        """Return the child model's maximum wavelength bound.

        Parameters
        ----------
        graph_state : GraphState or None, optional
            Sampled state forwarded unchanged to the child model.

        Returns
        -------
        float or None
            Child maximum wavelength bound in Angstroms, or ``None`` when
            unbounded.
        """
        return self.source_model.maxwave(graph_state=graph_state)

    def _get_active_images(self, state):
        """Return validated active magnifications and relative delays.

        Parameters
        ----------
        state : GraphState
            One-sample state containing this wrapper's realized image
            parameters.

        Returns
        -------
        magnifications : numpy.ndarray, shape (I,)
            Active absolute dimensionless magnifications in ascending
            relative-delay order, where ``I`` is the active image count.
        relative_delays : numpy.ndarray, shape (I,)
            Active observer-frame delays in days, normalized to begin at zero
            and sorted in ascending order.

        Raises
        ------
        ValueError
            If the realized image arrays are not one-dimensional or have
            different lengths; if ``num_images`` is not scalar and
            integer-valued, is less than two, or exceeds the array width; if an
            active magnification is non-finite or negative; if all active
            magnifications are zero; or if an active delay is non-finite.

        Notes
        -----
        The full magnification and delay inputs are first coerced to
        floating-point arrays, then checked for one-dimensional shape and
        equal length. These operations occur before ``num_images`` validation
        or prefix truncation, so non-coercible inactive values and invalid
        full-array structure are not ignored. When ``num_images`` is ``None``,
        every entry is active. Otherwise, both arrays are truncated to their
        leading ``num_images`` entries before finiteness and value validation.
        Float-coercible inactive values, including zero and NaN padding from
        ``CausticsLensImageNode``, are therefore ignored only by those
        active-value checks.

        A common delay offset is removed by subtracting the minimum active
        delay. Magnifications remain paired with their delays during a stable
        ascending sort, so images with equal delays retain their input order.
        Conversion and graph-lookup exceptions are propagated unchanged.
        """
        params = self.get_local_params(state)
        magnifications = np.asarray(params["macro_magnifications"], dtype=float)
        time_delays = np.asarray(params["time_delays"], dtype=float)

        if magnifications.ndim != 1 or time_delays.ndim != 1:
            raise ValueError(
                "macro_magnifications and time_delays must be one-dimensional for a single GraphState sample."
            )
        if len(magnifications) != len(time_delays):
            raise ValueError("macro_magnifications and time_delays must have the same length.")

        raw_num_images = params["num_images"]
        if raw_num_images is None:
            num_images = len(magnifications)
        else:
            if np.ndim(raw_num_images) != 0:
                raise ValueError("num_images must be a scalar for one sample.")
            num_images = int(raw_num_images)
            if num_images != raw_num_images:
                raise ValueError("num_images must be an integer.")

        if num_images < 2:
            raise ValueError("A strong lens system must contain at least two images.")
        if num_images > len(magnifications):
            raise ValueError(f"num_images={num_images} exceeds the image-array length {len(magnifications)}.")

        magnifications = magnifications[:num_images]
        time_delays = time_delays[:num_images]

        if not np.all(np.isfinite(magnifications)):
            raise ValueError("Active macro_magnifications must be finite.")
        if np.any(magnifications < 0.0):
            raise ValueError("Active macro_magnifications must be non-negative.")
        if not np.any(magnifications > 0.0):
            raise ValueError("At least one macro_magnification must be positive.")
        if not np.all(np.isfinite(time_delays)):
            raise ValueError("Active time_delays must be finite.")

        relative_delays = time_delays - np.min(time_delays)
        order = np.argsort(relative_delays, kind="stable")
        return magnifications[order], relative_delays[order]

    def _apply_wrapper_sed_effects(
        self,
        flux_density,
        *,
        times,
        wavelengths,
        state,
    ):
        """Apply wrapper observer-frame effects to an unresolved SED.

        Parameters
        ----------
        flux_density : numpy.ndarray, shape (T, W)
            Summed unresolved spectral flux density in nJy.
        times : numpy.ndarray, shape (T,)
            Original observer-frame observation times in MJD/days.
        wavelengths : numpy.ndarray, shape (W,)
            Observer-frame wavelengths in Angstroms.
        state : GraphState
            One-sample state supplying realized wrapper effect parameters.

        Returns
        -------
        flux_density : numpy.ndarray, shape (T, W)
            Spectral flux density in nJy after every wrapper effect.

        Raises
        ------
        Exception
            The first exception raised by a delegated effect is propagated
            unchanged.

        Notes
        -----
        Effects are applied in ``obs_frame_effects`` registration order. Each
        effect receives the output of the preceding effect, the original times
        and wavelengths, and the wrapper's realized local parameters.
        """
        params = self.get_local_params(state)
        for effect in self.obs_frame_effects:
            flux_density = effect.apply(
                flux_density,
                times=times,
                wavelengths=wavelengths,
                **params,
            )
        return flux_density

    def _apply_wrapper_bandflux_effects(
        self,
        bandfluxes,
        *,
        times,
        filters,
        state,
    ):
        """Apply wrapper observer-frame effects to unresolved bandfluxes.

        Parameters
        ----------
        bandfluxes : numpy.ndarray, shape (T,)
            Summed unresolved bandfluxes in nJy.
        times : numpy.ndarray, shape (T,)
            Original observer-frame observation times in MJD/days.
        filters : numpy.ndarray, shape (T,)
            Filter identifiers corresponding to the observations.
        state : GraphState
            One-sample state supplying realized wrapper effect parameters.

        Returns
        -------
        bandfluxes : numpy.ndarray, shape (T,)
            Bandfluxes in nJy after every wrapper effect.

        Raises
        ------
        NotImplementedError
            If a wrapper observer-frame effect does not implement
            ``apply_bandflux`` and inherits the base method.
        Exception
            Any other exception raised by a delegated effect is propagated
            unchanged.

        Notes
        -----
        Effects are applied in ``obs_frame_effects`` registration order. Each
        effect receives the output of the preceding effect, the original times
        and filters, and the wrapper's realized local parameters. Every effect
        in this list must implement ``apply_bandflux`` for use on this path;
        the base method's ``NotImplementedError`` is not intercepted.
        """
        params = self.get_local_params(state)
        for effect in self.obs_frame_effects:
            bandfluxes = effect.apply_bandflux(
                bandfluxes,
                times=times,
                filters=filters,
                **params,
            )
        return bandfluxes

    def _evaluate_single(self, times, wavelengths, state, **kwargs):
        """Evaluate one unresolved lensed SED in observer-frame units.

        Parameters
        ----------
        times : numpy.ndarray, shape (T,)
            Observer-frame observation times in MJD/days.
        wavelengths : numpy.ndarray, shape (W,)
            Observer-frame wavelengths in Angstroms.
        state : GraphState
            One-sample state containing the child and wrapper parameters.
        **kwargs : dict, optional
            Additional keyword arguments forwarded unchanged to the child's
            SED evaluation.

        Returns
        -------
        flux_density : numpy.ndarray, shape (T, W)
            Magnification-weighted unresolved spectral flux density in nJy,
            including child and wrapper effects.

        Raises
        ------
        TypeError
            If the child is a ``BandfluxModel``, which has no SED evaluation
            path.
        ValueError
            If active image parameters fail validation.

        Notes
        -----
        The image-major shifted-time grid is
        ``times[None, :] - relative_delays[:, None]``. It is flattened so all
        ``T`` times for the first image precede those for the next image, then
        evaluated by the child in one call and reshaped to ``(I, T, W)``.
        Absolute magnifications weight the image axis before it is summed.

        The SED child evaluation owns its configured framework redshift
        conversion and child-level effects for each shifted image. Wrapper
        observer-frame effects are applied once to the final sum at the
        original times and wavelengths. Exceptions from image coercion or
        validation, child evaluation, output reshaping, and wrapper effects
        are propagated unchanged.
        """
        if isinstance(self.source_model, BandfluxModel):
            raise TypeError(
                "UnresolvedStrongLensModel contains a BandfluxModel, which does not support SED evaluation."
            )

        times = np.asarray(times, dtype=float)
        wavelengths = np.asarray(wavelengths, dtype=float)
        magnifications, relative_delays = self._get_active_images(state)

        num_images = len(magnifications)
        num_times = len(times)
        num_waves = len(wavelengths)

        shifted_times = times[np.newaxis, :] - relative_delays[:, np.newaxis]
        source_flux = self.source_model.evaluate_sed(
            shifted_times.reshape(num_images * num_times),
            wavelengths,
            state,
            **kwargs,
        )
        source_flux = np.asarray(source_flux).reshape(
            num_images,
            num_times,
            num_waves,
        )
        flux_density = np.einsum(
            "i,itw->tw",
            magnifications,
            source_flux,
            optimize=True,
        )

        return self._apply_wrapper_sed_effects(
            flux_density,
            times=times,
            wavelengths=wavelengths,
            state=state,
        )

    def _evaluate_bandfluxes_single(
        self,
        passband_group,
        times,
        filters,
        state,
    ):
        """Evaluate one unresolved lensed bandflux time series.

        Parameters
        ----------
        passband_group : PassbandGroup or None
            Passband collection delegated to the child evaluation. ``None``
            is valid only for child models that do not require passband
            definitions.
        times : numpy.ndarray, shape (T,)
            Observer-frame observation times in MJD/days.
        filters : numpy.ndarray, shape (T,)
            Filter identifiers selecting a passband for each observation.
        state : GraphState
            One-sample state containing the child and wrapper parameters.

        Returns
        -------
        bandfluxes : numpy.ndarray, shape (T,)
            Magnification-weighted unresolved bandfluxes in nJy, including
            child and wrapper effects.

        Raises
        ------
        ValueError
            If active image parameters fail validation.
        NotImplementedError
            If a ``BandfluxModel`` child effect or wrapper observer-frame
            effect does not implement ``apply_bandflux``.

        Notes
        -----
        The image-major grid uses ``times - relative_delay`` for each image and
        is flattened to length ``I * T``. The filter sequence is tiled once per
        image in the same order before both arrays are delegated to the child's
        bandflux evaluation. Its result is reshaped to ``(I, T)``, weighted by
        absolute magnification, and summed over images.

        This path supports either an SED-based or bandflux-only child. An SED
        child evaluates and integrates its SED, including that child's
        configured framework redshift conversion and rest- and observer-frame
        effects, for each shifted image. A ``BandfluxModel`` instead evaluates
        observer-frame bandfluxes with no framework redshift conversion. All
        of its child effects, including effects marked as rest-frame and
        delegated through this wrapper, execute through its observer-frame
        ``apply_bandflux`` API.

        Wrapper observer-frame effects run once on the final sum at the
        original times and filters, and each must implement
        ``apply_bandflux``. Exceptions from image coercion or validation,
        child evaluation, output reshaping, passband/filter lookup, and
        wrapper effects are propagated unchanged.
        """
        times = np.asarray(times, dtype=float)
        filters = np.asarray(filters)
        magnifications, relative_delays = self._get_active_images(state)

        num_images = len(magnifications)
        num_times = len(times)

        shifted_times = times[np.newaxis, :] - relative_delays[:, np.newaxis]
        shifted_filters = np.tile(filters, num_images)
        source_flux = self.source_model.evaluate_bandfluxes(
            passband_group,
            shifted_times.reshape(num_images * num_times),
            shifted_filters,
            state,
        )
        source_flux = np.asarray(source_flux).reshape(num_images, num_times)
        bandfluxes = np.einsum(
            "i,it->t",
            magnifications,
            source_flux,
            optimize=True,
        )

        return self._apply_wrapper_bandflux_effects(
            bandfluxes,
            times=times,
            filters=filters,
            state=state,
        )

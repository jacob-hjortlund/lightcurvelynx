import numpy as np

from lightcurvelynx.models.multi_object_model import MultiObjectModel
from lightcurvelynx.models.physical_model import BandfluxModel, BasePhysicalModel


class UnresolvedStrongLensModel(MultiObjectModel):
    """Wrap one physical source as an unresolved static macro-lens system.

    The model evaluates the source at observer-frame times shifted by each
    macro-image's relative arrival delay, scales those evaluations by absolute
    macro-magnifications, and returns their sum.

    Parameters
    ----------
    source_model : BasePhysicalModel
        The physical source to lens.
    macro_magnifications : parameter
        Absolute, dimensionless image magnifications. Each sampled value is a
        one-dimensional array.
    time_delays : parameter
        Image arrival delays in observer-frame days. An arbitrary common offset
        is allowed and removed before evaluation.
    num_images : parameter, optional
        Number of active entries in the image arrays. If None, every entry is
        active.
    node_label : str, optional
        Label for the outer model node.
    **kwargs : dict, optional
        Overrides for outer physical-model parameters such as ra, dec, t0, and
        redshift. Unspecified values are linked to the source model.
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

        # This composite overrides the single-state evaluation pipeline. Redshift
        # conversion is owned by the child source and must not be applied twice.
        self.apply_redshift = False

    def minwave(self, graph_state=None):
        """Return the child's minimum supported wavelength in Angstroms."""
        return self.source_model.minwave(graph_state=graph_state)

    def maxwave(self, graph_state=None):
        """Return the child's maximum supported wavelength in Angstroms."""
        return self.source_model.maxwave(graph_state=graph_state)

    def _get_active_images(self, state):
        """Return validated magnifications and normalized delays for one system."""
        params = self.get_local_params(state)
        magnifications = np.asarray(params["macro_magnifications"], dtype=float)
        time_delays = np.asarray(params["time_delays"], dtype=float)

        if magnifications.ndim != 1 or time_delays.ndim != 1:
            raise ValueError(
                "macro_magnifications and time_delays must be one-dimensional "
                "for a single GraphState sample."
            )
        if len(magnifications) != len(time_delays):
            raise ValueError(
                "macro_magnifications and time_delays must have the same length."
            )

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
            raise ValueError(
                f"num_images={num_images} exceeds the image-array length "
                f"{len(magnifications)}."
            )

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
        """Evaluate one unresolved lensed SED in observer-frame units."""
        if isinstance(self.source_model, BandfluxModel):
            raise TypeError(
                "UnresolvedStrongLensModel contains a BandfluxModel, which does "
                "not support SED evaluation."
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
        """Evaluate one unresolved lensed bandflux time series in nJy."""
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
import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord, SkyOffsetFrame

from lightcurvelynx.effects.basic_effects import ScaleFluxEffect
from lightcurvelynx.effects.effect_model import EffectModel
from lightcurvelynx.math_nodes.basic_math_node import BasicMathNode
from lightcurvelynx.math_nodes.given_sampler import GivenValueList
from lightcurvelynx.models._resolved_strong_lens import (
    _MacroMagnificationEffect,
    _ResolvedCoordinatesNode,
    _ResolvedImageDataNode,
)
from lightcurvelynx.models.physical_model import BandfluxModel, BasePhysicalModel, SEDModel
from lightcurvelynx.models.strong_lens_model import ResolvedStrongLensModel


class _PhaseSEDModel(SEDModel):
    """Return a deterministic SED as a function of phase and wavelength."""

    def compute_sed(self, times, wavelengths, graph_state, **kwargs):
        """Return phase plus wavelength in units of 1,000 Angstroms."""
        del kwargs
        t0 = self.get_param(graph_state, "t0")
        return (times - t0)[:, np.newaxis] + wavelengths[np.newaxis, :] / 1000.0


class _PhaseBandfluxModel(BandfluxModel):
    """Return deterministic observer-frame bandfluxes as a function of phase."""

    def compute_bandflux(self, times, filter, state):
        """Return phase plus a fixed filter-dependent offset."""
        t0 = self.get_param(state, "t0")
        return times - t0 + {"g": 1.0, "r": 10.0}[filter]


class _RecordingSEDModel(SEDModel):
    """Record the rest-frame coordinates used for an analytic SED."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.compute_calls = []

    def compute_sed(self, times, wavelengths, graph_state, **kwargs):
        """Record inputs and return ``time + wavelength / 1000``."""
        del graph_state, kwargs
        times = np.asarray(times, dtype=float)
        wavelengths = np.asarray(wavelengths, dtype=float)
        self.compute_calls.append((times.copy(), wavelengths.copy()))
        return times[:, None] + wavelengths[None, :] / 1_000.0


class _RecordingScaleEffect(EffectModel):
    """Scale an SED while recording the realized image metadata."""

    def __init__(self, scale, *, call_order=None, marker="scale"):
        super().__init__(rest_frame=True, scale=scale)
        self.calls = []
        self.call_order = call_order
        self.marker = marker

    def apply(
        self,
        flux_density,
        *,
        times=None,
        wavelengths=None,
        scale=None,
        ra=None,
        dec=None,
        t0=None,
        **kwargs,
    ):
        """Record image coordinates and apply the realized scale."""
        del kwargs
        if self.call_order is not None:
            self.call_order.append(self.marker)
        self.calls.append(
            {
                "times": np.asarray(times).copy(),
                "wavelengths": np.asarray(wavelengths).copy(),
                "scale": scale,
                "ra": ra,
                "dec": dec,
                "t0": t0,
            }
        )
        return np.asarray(flux_density) * scale

    def apply_bandflux(
        self,
        bandfluxes,
        *,
        times=None,
        filters=None,
        scale=None,
        ra=None,
        dec=None,
        t0=None,
        **kwargs,
    ):
        """Record image metadata and scale observer-frame bandfluxes."""
        del kwargs
        if self.call_order is not None:
            self.call_order.append(self.marker)
        self.calls.append(
            {
                "times": np.asarray(times).copy(),
                "filters": np.asarray(filters).copy(),
                "scale": scale,
                "ra": ra,
                "dec": dec,
                "t0": t0,
            }
        )
        return np.asarray(bandfluxes) * scale


class _RecordingAdditiveEffect(EffectModel):
    """Add one named sampled offset and record effect execution order."""

    def __init__(self, parameter_name, setter, *, call_order, marker):
        super().__init__(rest_frame=False)
        self.add_effect_parameter(parameter_name, setter)
        self.parameter_name = parameter_name
        self.call_order = call_order
        self.marker = marker
        self.calls = []

    def _apply(self, values, **params):
        offset = params[self.parameter_name]
        self.call_order.append(self.marker)
        self.calls.append(offset)
        return np.asarray(values) + offset

    def apply(self, flux_density, times=None, wavelengths=None, **kwargs):
        """Add the realized offset to an SED."""
        del times, wavelengths
        return self._apply(flux_density, **kwargs)

    def apply_bandflux(self, bandfluxes, *, times=None, filters=None, **kwargs):
        """Add the realized offset to bandfluxes."""
        del times, filters
        return self._apply(bandfluxes, **kwargs)


class _BoundedPhaseSEDModel(_PhaseSEDModel):
    """Return fixed wavelength bounds and record their supplied state."""

    def minwave(self, graph_state=None):
        """Record and return the exact minimum wavelength."""
        self.minwave_state = graph_state
        return 900.0

    def maxwave(self, graph_state=None):
        """Record and return the exact maximum wavelength."""
        self.maxwave_state = graph_state
        return 2_100.0


class _SEDOnlyEffect(EffectModel):
    """Implement only SED application to exercise the bandflux guard."""

    def __init__(self):
        super().__init__(rest_frame=False)

    def apply(self, flux_density, **kwargs):
        """Leave SED values unchanged."""
        del kwargs
        return flux_density


def _make_resolved_lens(*, source=None, node_label="resolved"):
    if source is None:
        source = _PhaseSEDModel(
            ra=GivenValueList([20.0, 30.0], stateful=False),
            dec=GivenValueList([10.0, -5.0], stateful=False),
            redshift=0.0,
            t0=GivenValueList([100.0, 200.0], stateful=False),
            node_label="source",
        )
    model = ResolvedStrongLensModel(
        source,
        source_x=GivenValueList([0.5, -0.25], stateful=False),
        source_y=GivenValueList([-0.25, 0.75], stateful=False),
        image_x=GivenValueList(
            [[2.0, -1.0, np.nan], [0.5, -0.5, 1.5]],
            stateful=False,
        ),
        image_y=GivenValueList(
            [[0.0, 1.0, np.nan], [1.0, -1.0, 0.25]],
            stateful=False,
        ),
        macro_magnifications=GivenValueList(
            [[3.0, 2.0, 0.0], [4.0, 5.0, 6.0]],
            stateful=False,
        ),
        time_delays=GivenValueList(
            [[12.0, 10.0, np.nan], [7.0, 9.0, 8.0]],
            stateful=False,
        ),
        num_images=GivenValueList([2, 3], stateful=False),
        node_label=node_label,
    )
    return source, model


def _resolved_constructor_kwargs():
    return {
        "source_x": 0.0,
        "source_y": 0.0,
        "image_x": [0.0, 1.0],
        "image_y": [0.0, 1.0],
        "macro_magnifications": [1.0, 2.0],
        "time_delays": [0.0, 1.0],
        "num_images": 2,
    }


def _source_configuration(source):
    return (
        source.setters.copy(),
        tuple(source.rest_frame_effects),
        tuple(source.obs_frame_effects),
    )


def _record_effect_application(monkeypatch, effect, method_name, call_order, marker):
    """Record one real effect method without replacing its implementation."""
    original_method = getattr(effect, method_name)

    def recording_method(*args, **kwargs):
        call_order.append(marker)
        return original_method(*args, **kwargs)

    monkeypatch.setattr(effect, method_name, recording_method)


def test_resolved_lens_owns_and_decorates_source():
    """Retain and decorate the exact source as the wrapper's only child."""
    source, model = _make_resolved_lens()

    assert model.source_model is source
    assert model.objects == [source]
    assert model.num_objects == 1
    assert model.apply_redshift is False
    assert model.simulation_metadata_params == ("system_id", "image_id")
    assert {"base_ra", "base_dec", "base_t0"}.issubset(source.setters)
    assert "macro_magnification" in source.setters


def test_resolved_lens_expands_systems_and_exposes_sorted_image_metadata():
    """Expand systems into delay-sorted image rows with stable provenance."""
    source, model = _make_resolved_lens()

    state = model.sample_parameters(num_samples=2)

    assert state.num_samples == 5
    np.testing.assert_array_equal(state["resolved.system_id"], [0, 0, 1, 1, 1])
    np.testing.assert_array_equal(state["resolved.image_id"], [0, 1, 0, 1, 2])
    np.testing.assert_array_equal(state["resolved.image_x"], [-1.0, 2.0, 0.5, 1.5, -0.5])
    np.testing.assert_array_equal(state["resolved.image_y"], [1.0, 0.0, 1.0, 0.25, -1.0])
    np.testing.assert_array_equal(state["resolved.time_delay"], [0.0, 2.0, 0.0, 1.0, 2.0])
    np.testing.assert_array_equal(state["resolved.macro_magnification"], [2.0, 3.0, 4.0, 6.0, 5.0])
    np.testing.assert_array_equal(
        state["source.base_t0"],
        [100.0, 100.0, 200.0, 200.0, 200.0],
    )
    np.testing.assert_array_equal(
        state["source.t0"],
        [100.0, 102.0, 200.0, 201.0, 202.0],
    )


def test_resolved_lens_exact_sed_evaluation():
    """Evaluate every expanded image with its exact delay and magnification."""
    source = _PhaseSEDModel(
        ra=20.0,
        dec=10.0,
        redshift=0.0,
        t0=100.0,
        node_label="source",
    )
    model = ResolvedStrongLensModel(
        source,
        source_x=0.0,
        source_y=0.0,
        image_x=[0.0, 1.0, 2.0],
        image_y=[0.0, 1.0, 2.0],
        macro_magnifications=[1.0, 2.0, 3.0],
        time_delays=[4.0, 0.0, 2.0],
        node_label="resolved",
    )
    state = model.sample_parameters()
    times = np.array([105.0, 106.0])
    wavelengths = np.array([4_000.0, 5_000.0])
    expected_t0 = np.array([100.0, 102.0, 104.0])
    expected_magnification = np.array([2.0, 3.0, 1.0])
    expected = np.stack(
        [
            mu * ((times - image_t0)[:, None] + wavelengths[None, :] / 1_000.0)
            for image_t0, mu in zip(
                expected_t0,
                expected_magnification,
                strict=True,
            )
        ]
    )

    np.testing.assert_array_equal(
        model.evaluate_sed(times, wavelengths, state),
        expected,
    )


def test_resolved_lens_exact_bandflux_evaluation():
    """Evaluate every expanded image's exact filter flux and magnification."""
    source = _PhaseBandfluxModel(
        ra=20.0,
        dec=10.0,
        redshift=0.0,
        t0=100.0,
        node_label="source",
    )
    model = ResolvedStrongLensModel(
        source,
        source_x=0.0,
        source_y=0.0,
        image_x=[0.0, 1.0, 2.0],
        image_y=[0.0, 1.0, 2.0],
        macro_magnifications=[1.0, 2.0, 3.0],
        time_delays=[4.0, 0.0, 2.0],
        node_label="resolved",
    )
    state = model.sample_parameters()
    passbands = None
    times = np.array([105.0, 106.0])
    filters = np.array(["g", "r"])
    expected_t0 = np.array([100.0, 102.0, 104.0])
    expected_magnification = np.array([2.0, 3.0, 1.0])
    expected = np.stack(
        [
            mu * (times - image_t0 + np.array([1.0, 10.0]))
            for image_t0, mu in zip(
                expected_t0,
                expected_magnification,
                strict=True,
            )
        ]
    )

    np.testing.assert_array_equal(
        model.evaluate_bandfluxes(passbands, times, filters, state),
        expected,
    )


def test_resolved_lens_applies_redshift_and_source_effect_once():
    """Share pre-wrapper latents while evaluating each final image once."""
    source = _RecordingSEDModel(
        ra=GivenValueList([20.0, 30.0], stateful=False),
        dec=GivenValueList([10.0, -5.0], stateful=False),
        redshift=1.0,
        t0=GivenValueList([4.0, 4.0], stateful=False),
        node_label="source",
    )
    effect = _RecordingScaleEffect(
        GivenValueList([2.0, 5.0], stateful=False),
    )
    source.add_effect(effect)
    _, model = _make_resolved_lens(source=source)
    state = model.sample_parameters(num_samples=2)
    observer_times = np.array([10.0, 14.0])
    observer_wavelengths = np.array([1_000.0, 3_000.0])

    result = model.evaluate_sed(
        observer_times,
        observer_wavelengths,
        state,
    )

    expected_scales = np.array([2.0, 2.0, 5.0, 5.0, 5.0])
    np.testing.assert_array_equal(state["source.scale"], expected_scales)
    np.testing.assert_allclose(
        [call["ra"] for call in effect.calls],
        state["source.ra"],
    )
    np.testing.assert_allclose(
        [call["dec"] for call in effect.calls],
        state["source.dec"],
    )
    np.testing.assert_array_equal(
        [call["t0"] for call in effect.calls],
        state["source.t0"],
    )

    expected_rest_times = np.array(
        [
            [7.0, 9.0],
            [8.0, 10.0],
            [7.0, 9.0],
            [7.5, 9.5],
            [8.0, 10.0],
        ]
    )
    expected_rest_wavelengths = np.array([500.0, 1_500.0])
    np.testing.assert_array_equal(
        np.stack([call[0] for call in source.compute_calls]),
        expected_rest_times,
    )
    for _, rest_wavelengths in source.compute_calls:
        np.testing.assert_array_equal(
            rest_wavelengths,
            expected_rest_wavelengths,
        )

    rest_frame_flux = expected_rest_times[:, :, None] + expected_rest_wavelengths[None, None, :] / 1_000.0
    expected_magnifications = np.array([2.0, 3.0, 4.0, 6.0, 5.0])
    expected = expected_magnifications[:, None, None] * 2.0 * expected_scales[:, None, None] * rest_frame_flux
    np.testing.assert_allclose(result, expected)


def test_post_wrapper_sed_effects_are_image_specific_and_ordered(monkeypatch):
    """Sample post-expansion SED effects per image in frame order."""
    source, model = _make_resolved_lens()
    call_order = []
    scale_values = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
    first_offsets = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    second_offsets = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
    scale_effect = _RecordingScaleEffect(
        GivenValueList(scale_values, stateful=False),
        call_order=call_order,
        marker="rest_scale",
    )
    first_effect = _RecordingAdditiveEffect(
        "first_offset",
        GivenValueList(first_offsets, stateful=False),
        call_order=call_order,
        marker="first_observer",
    )
    second_effect = _RecordingAdditiveEffect(
        "second_offset",
        GivenValueList(second_offsets, stateful=False),
        call_order=call_order,
        marker="second_observer",
    )
    model.add_effect(scale_effect)
    model.add_effect(first_effect)
    model.add_effect(second_effect)
    _record_effect_application(
        monkeypatch,
        model._macro_magnification_effect,
        "apply",
        call_order,
        "macro",
    )

    state = model.sample_parameters(num_samples=2)
    times = np.array([205.0, 206.0])
    wavelengths = np.array([4_000.0, 5_000.0])
    result = model.evaluate_sed(times, wavelengths, state)

    np.testing.assert_array_equal(state["source.scale"], scale_values)
    np.testing.assert_array_equal(state["resolved.first_offset"], first_offsets)
    np.testing.assert_array_equal(state["resolved.second_offset"], second_offsets)
    assert source.rest_frame_effects == [scale_effect]
    assert source.obs_frame_effects == [model._macro_magnification_effect]
    assert source.list_effects() == [
        scale_effect,
        model._macro_magnification_effect,
    ]
    assert model.obs_frame_effects == [first_effect, second_effect]
    assert call_order == [
        marker for _ in range(5) for marker in ("rest_scale", "macro", "first_observer", "second_observer")
    ]

    image_t0 = np.asarray(state["source.t0"])
    magnifications = np.asarray(state["source.macro_magnification"])
    unlensed = np.stack([(times - t0)[:, None] + wavelengths[None, :] / 1_000.0 for t0 in image_t0])
    expected = (
        magnifications[:, None, None] * scale_values[:, None, None] * unlensed
        + first_offsets[:, None, None]
        + second_offsets[:, None, None]
    )
    np.testing.assert_array_equal(result, expected)


def test_post_wrapper_bandflux_effects_are_image_specific_and_ordered(monkeypatch):
    """Keep bandflux effects in one registration-ordered child pipeline."""
    source = _PhaseBandfluxModel(
        ra=GivenValueList([20.0, 30.0], stateful=False),
        dec=GivenValueList([10.0, -5.0], stateful=False),
        redshift=0.0,
        t0=GivenValueList([100.0, 200.0], stateful=False),
        node_label="source",
    )
    _, model = _make_resolved_lens(source=source)
    call_order = []
    scale_values = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
    first_offsets = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    second_offsets = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
    scale_effect = _RecordingScaleEffect(
        GivenValueList(scale_values, stateful=False),
        call_order=call_order,
        marker="delegated_scale",
    )
    first_effect = _RecordingAdditiveEffect(
        "first_offset",
        GivenValueList(first_offsets, stateful=False),
        call_order=call_order,
        marker="first_observer",
    )
    second_effect = _RecordingAdditiveEffect(
        "second_offset",
        GivenValueList(second_offsets, stateful=False),
        call_order=call_order,
        marker="second_observer",
    )
    model.add_effect(scale_effect)
    model.add_effect(first_effect)
    model.add_effect(second_effect)
    _record_effect_application(
        monkeypatch,
        model._macro_magnification_effect,
        "apply_bandflux",
        call_order,
        "macro",
    )

    state = model.sample_parameters(num_samples=2)
    times = np.array([205.0, 206.0])
    filters = np.array(["g", "r"])
    result = model.evaluate_bandfluxes(None, times, filters, state)

    np.testing.assert_array_equal(state["source.scale"], scale_values)
    np.testing.assert_array_equal(state["resolved.first_offset"], first_offsets)
    np.testing.assert_array_equal(state["resolved.second_offset"], second_offsets)
    assert source.band_pass_effects == [
        model._macro_magnification_effect,
        scale_effect,
    ]
    assert model.obs_frame_effects == [first_effect, second_effect]
    assert call_order == [
        marker
        for _ in range(5)
        for marker in ("macro", "delegated_scale", "first_observer", "second_observer")
    ]

    image_t0 = np.asarray(state["source.t0"])
    magnifications = np.asarray(state["source.macro_magnification"])
    unlensed = np.stack([times - t0 + np.array([1.0, 10.0]) for t0 in image_t0])
    expected = (
        magnifications[:, None] * scale_values[:, None] * unlensed
        + first_offsets[:, None]
        + second_offsets[:, None]
    )
    np.testing.assert_array_equal(result, expected)


def test_scale_flux_effect_coexists_with_macro_magnification():
    """Keep public source scaling distinct from private macro scaling."""
    source = _PhaseSEDModel(
        ra=GivenValueList([20.0, 30.0], stateful=False),
        dec=GivenValueList([10.0, -5.0], stateful=False),
        redshift=0.0,
        t0=GivenValueList([100.0, 200.0], stateful=False),
        node_label="source",
    )
    source.add_effect(ScaleFluxEffect(flux_scale=2.0))
    _, model = _make_resolved_lens(source=source)

    assert "flux_scale" in source.setters
    assert "macro_magnification" in source.setters
    state = model.sample_parameters(num_samples=2)
    times = np.array([205.0, 206.0])
    wavelengths = np.array([4_000.0, 5_000.0])
    result = model.evaluate_sed(times, wavelengths, state)

    image_t0 = np.asarray(state["source.t0"])
    magnifications = np.asarray(state["source.macro_magnification"])
    unlensed = np.stack([(times - t0)[:, None] + wavelengths[None, :] / 1_000.0 for t0 in image_t0])
    expected = 2.0 * magnifications[:, None, None] * unlensed
    np.testing.assert_array_equal(result, expected)


def test_resolved_lens_public_sed_rejects_bandflux_child():
    """Raise the documented public error for bandflux-only SED evaluation."""
    source = _PhaseBandfluxModel(
        ra=20.0,
        dec=10.0,
        redshift=0.0,
        t0=100.0,
        node_label="source",
    )
    _, model = _make_resolved_lens(source=source)
    state = model.sample_parameters(num_samples=2)

    with pytest.raises(
        TypeError,
        match="ResolvedStrongLensModel contains a BandfluxModel, which does not support SED evaluation",
    ):
        model.evaluate_sed([105.0], [4_000.0], state)


def test_resolved_lens_public_bandflux_propagates_missing_effect_method():
    """Propagate a wrapper effect's unsupported bandflux operation."""
    source = _PhaseBandfluxModel(
        ra=20.0,
        dec=10.0,
        redshift=0.0,
        t0=100.0,
        node_label="source",
    )
    _, model = _make_resolved_lens(source=source)
    model.add_effect(_SEDOnlyEffect())
    state = model.sample_parameters(num_samples=2)

    with pytest.raises(NotImplementedError):
        model.evaluate_bandfluxes(
            None,
            np.array([105.0]),
            np.array(["g"]),
            state,
        )


def test_resolved_lens_public_wavelength_bounds_forward_state():
    """Return child wavelength bounds with the exact expanded state."""
    source = _BoundedPhaseSEDModel(
        ra=GivenValueList([20.0, 30.0], stateful=False),
        dec=GivenValueList([10.0, -5.0], stateful=False),
        redshift=0.0,
        t0=GivenValueList([100.0, 200.0], stateful=False),
        node_label="source",
    )
    _, model = _make_resolved_lens(source=source)
    state = model.sample_parameters(num_samples=2)

    assert model.minwave(state) == 900.0
    assert model.maxwave(state) == 2_100.0
    assert source.minwave_state is state
    assert source.maxwave_state is state


def test_resolved_lens_public_sed_shapes_follow_state_rows():
    """Return ordinary or row-stacked SED shapes from public evaluation."""
    _, model = _make_resolved_lens()
    state = model.sample_parameters(num_samples=2)
    one_image_state = next(iter(state))
    times = np.array([205.0, 206.0])
    wavelengths = np.array([4_000.0, 5_000.0])

    assert model.evaluate_sed(times, wavelengths, one_image_state).shape == (2, 2)
    assert model.evaluate_sed(times, wavelengths, state).shape == (5, 2, 2)


def test_resolved_lens_public_bandflux_shapes_follow_state_rows():
    """Return ordinary or row-stacked bandflux shapes from public evaluation."""
    source = _PhaseBandfluxModel(
        ra=GivenValueList([20.0, 30.0], stateful=False),
        dec=GivenValueList([10.0, -5.0], stateful=False),
        redshift=0.0,
        t0=GivenValueList([100.0, 200.0], stateful=False),
        node_label="source",
    )
    _, model = _make_resolved_lens(source=source)
    state = model.sample_parameters(num_samples=2)
    one_image_state = next(iter(state))
    times = np.array([205.0, 206.0])
    filters = np.array(["g", "r"])

    assert model.evaluate_bandfluxes(None, times, filters, one_image_state).shape == (2,)
    assert model.evaluate_bandfluxes(None, times, filters, state).shape == (5, 2)


def test_resolved_lens_broadcasts_constant_arrays_and_uses_full_width():
    """Broadcast constant image arrays and use all entries without a count."""
    source = _PhaseSEDModel(
        ra=20.0,
        dec=10.0,
        redshift=0.0,
        t0=100.0,
        node_label="source",
    )
    model = ResolvedStrongLensModel(
        source,
        source_x=0.0,
        source_y=0.0,
        image_x=[3.0, 1.0, 2.0],
        image_y=[30.0, 10.0, 20.0],
        macro_magnifications=[6.0, 4.0, 5.0],
        time_delays=[2.0, 0.0, 1.0],
        num_images=None,
        node_label="resolved",
    )

    state = model.sample_parameters(num_samples=2, sample_offset=4)

    assert state.num_samples == 6
    np.testing.assert_array_equal(state["resolved.system_id"], [4, 4, 4, 5, 5, 5])
    np.testing.assert_array_equal(state["resolved.image_id"], [0, 1, 2, 0, 1, 2])
    np.testing.assert_array_equal(state["resolved.image_x"], [1, 2, 3, 1, 2, 3])
    np.testing.assert_array_equal(state["resolved.time_delay"], [0, 1, 2, 0, 1, 2])


def test_resolved_lens_coordinates_preserve_source_and_image_offsets():
    """Recover every configured source- and image-plane spherical offset."""
    _, model = _make_resolved_lens()

    state = model.sample_parameters(num_samples=2)

    for image_index in range(state.num_samples):
        lens = SkyCoord(
            ra=state["resolved.lens_ra"][image_index] * u.deg,
            dec=state["resolved.lens_dec"][image_index] * u.deg,
        )
        source = SkyCoord(
            ra=state["source.base_ra"][image_index] * u.deg,
            dec=state["source.base_dec"][image_index] * u.deg,
        )
        recovered_source = source.transform_to(SkyOffsetFrame(origin=lens))
        assert recovered_source.lon.to_value(u.arcsec) == pytest.approx(
            state["resolved.source_x"][image_index],
            abs=1e-6,
        )
        assert recovered_source.lat.to_value(u.arcsec) == pytest.approx(
            state["resolved.source_y"][image_index],
            abs=1e-6,
        )

        image = SkyCoord(
            ra=state["resolved.ra"][image_index] * u.deg,
            dec=state["resolved.dec"][image_index] * u.deg,
        )
        recovered_image = image.transform_to(SkyOffsetFrame(origin=lens))
        assert recovered_image.lon.to_value(u.arcsec) == pytest.approx(
            state["resolved.image_x"][image_index],
            abs=1e-6,
        )
        assert recovered_image.lat.to_value(u.arcsec) == pytest.approx(
            state["resolved.image_y"][image_index],
            abs=1e-6,
        )


def test_resolved_lens_rejects_non_physical_source_without_mutation():
    """Reject a nonphysical source before changing its source-like state."""

    class _NonPhysicalSource:
        setters = {}
        rest_frame_effects = []
        obs_frame_effects = []

    source = _NonPhysicalSource()
    assert not isinstance(source, BasePhysicalModel)
    before = _source_configuration(source)

    with pytest.raises(TypeError, match="source_model must be a BasePhysicalModel"):
        ResolvedStrongLensModel(source, **_resolved_constructor_kwargs())

    assert _source_configuration(source) == before


@pytest.mark.parametrize(
    "reserved_name",
    ["base_ra", "base_dec", "base_t0", "macro_magnification"],
)
def test_resolved_lens_rejects_reserved_source_parameters_without_mutation(reserved_name):
    """Reject reserved source parameters without changing source configuration."""
    source = _PhaseSEDModel(
        ra=20.0,
        dec=10.0,
        redshift=0.0,
        t0=100.0,
        node_label="source",
    )
    source.add_parameter(reserved_name, 1.0)
    before = _source_configuration(source)

    with pytest.raises(ValueError, match=reserved_name):
        ResolvedStrongLensModel(source, **_resolved_constructor_kwargs())

    assert _source_configuration(source) == before


@pytest.mark.parametrize("parameter_name", ["ra", "dec", "t0"])
def test_resolved_lens_rejects_dependent_source_parameters_without_mutation(parameter_name):
    """Reject source-coordinate dependents without changing configuration."""
    source = _PhaseSEDModel(
        ra=20.0,
        dec=10.0,
        redshift=0.0,
        t0=100.0,
        node_label="source",
    )
    dependent = BasicMathNode(
        "value + 1.0",
        value=getattr(source, parameter_name),
        node_label=f"{parameter_name}_dependent",
    )
    source.set_parameter("distance", dependent)
    before = _source_configuration(source)

    with pytest.raises(ValueError, match=parameter_name):
        ResolvedStrongLensModel(source, **_resolved_constructor_kwargs())

    assert _source_configuration(source) == before


def test_resolved_lens_rejection_preserves_unlabeled_reachable_graph_identity():
    """Reject without assigning graph identities to any reachable node."""
    source = _PhaseSEDModel(
        ra=20.0,
        dec=10.0,
        redshift=0.0,
        t0=100.0,
    )
    dependent = BasicMathNode("value + 1.0", value=source.ra)
    source.set_parameter("distance", dependent)

    def graph_identity(node):
        return (
            node.node_pos,
            node.node_string,
            tuple(setter.node_name for setter in node.setters.values()),
        )

    before = tuple(graph_identity(node) for node in (source, dependent))

    with pytest.raises(ValueError, match="parameter ra has dependent parameters"):
        ResolvedStrongLensModel(source, **_resolved_constructor_kwargs())

    assert tuple(graph_identity(node) for node in (source, dependent)) == before


def test_image_data_node_normalizes_sorts_and_ignores_padding():
    """Active images are normalized and stably ordered ahead of padding."""
    node = _ResolvedImageDataNode(
        source_t0=GivenValueList([100.0, 200.0], stateful=False),
        source_x=GivenValueList([0.1, -0.2], stateful=False),
        source_y=GivenValueList([0.3, -0.4], stateful=False),
        image_x=GivenValueList(
            [[20.0, 10.0, 30.0, np.nan], [4.0, 5.0, np.nan, np.nan]],
            stateful=False,
        ),
        image_y=GivenValueList(
            [[2.0, 1.0, 3.0, np.nan], [8.0, 9.0, np.nan, np.nan]],
            stateful=False,
        ),
        macro_magnifications=GivenValueList(
            [[2.0, 1.0, 3.0, 0.0], [4.0, 5.0, 0.0, 0.0]],
            stateful=False,
        ),
        time_delays=GivenValueList(
            [[12.0, 10.0, 11.0, np.nan], [7.0, 7.0, np.nan, np.nan]],
            stateful=False,
        ),
        num_images=GivenValueList([3, 2], stateful=False),
        node_label="image_data",
    )

    state = node.sample_parameters(num_samples=2)
    rows = state["image_data.image_data"].tolist()

    assert len(rows) == 2
    np.testing.assert_array_equal(rows[0]["image_x"], [10.0, 30.0, 20.0])
    np.testing.assert_array_equal(rows[0]["image_y"], [1.0, 3.0, 2.0])
    np.testing.assert_array_equal(rows[0]["macro_magnification"], [1.0, 3.0, 2.0])
    np.testing.assert_array_equal(rows[0]["time_delay"], [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(rows[1]["image_x"], [4.0, 5.0])
    np.testing.assert_array_equal(rows[1]["time_delay"], [0.0, 0.0])


def test_image_data_node_one_system_returns_mapping_and_uses_full_width():
    """A single system uses every image when no active count is supplied."""
    node = _ResolvedImageDataNode(
        source_t0=100.0,
        source_x=0.0,
        source_y=0.0,
        image_x=[3.0, 1.0, 2.0],
        image_y=[30.0, 10.0, 20.0],
        macro_magnifications=[6.0, 4.0, 5.0],
        time_delays=[2.0, 0.0, 1.0],
        num_images=None,
        node_label="image_data",
    )

    state = node.sample_parameters()
    row = state["image_data.image_data"]

    assert isinstance(row, dict)
    np.testing.assert_array_equal(row["image_x"], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(row["image_y"], [10.0, 20.0, 30.0])
    np.testing.assert_array_equal(row["macro_magnification"], [4.0, 5.0, 6.0])
    np.testing.assert_array_equal(row["time_delay"], [0.0, 1.0, 2.0])


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"source_t0": np.nan}, "source_t0 must be finite"),
        ({"source_x": np.nan}, "source_x must be finite"),
        ({"source_y": np.inf}, "source_y must be finite"),
        ({"image_x": [[1.0, 2.0]]}, "one-dimensional"),
        ({"image_y": [1.0]}, "same fixed width"),
        ({"num_images": [2]}, "num_images must be scalar"),
        ({"num_images": 1.5}, "num_images must be an integer"),
        ({"num_images": 1}, "at least two images"),
        (
            {
                "image_x": [1.0],
                "image_y": [2.0],
                "macro_magnifications": [3.0],
                "time_delays": [0.0],
                "num_images": None,
            },
            "at least two images",
        ),
        ({"num_images": 4}, "exceeds the image-array width"),
        ({"image_x": [1.0, np.nan]}, "Active image_x must be finite"),
        ({"image_y": [1.0, np.inf]}, "Active image_y must be finite"),
        ({"macro_magnifications": [1.0, np.nan]}, "finite"),
        ({"macro_magnifications": [1.0, -1.0]}, "non-negative"),
        ({"macro_magnifications": [0.0, 0.0]}, "positive"),
        ({"time_delays": [0.0, np.inf]}, "Active time_delays must be finite"),
        (
            {"image_x": [1.0, 2.0, "not-a-number"], "num_images": 2},
            "float-coercible",
        ),
    ],
)
def test_image_data_node_rejects_invalid_realizations(overrides, message):
    """Invalid resolved-image realizations raise clear validation errors."""
    inputs = {
        "source_t0": 100.0,
        "source_x": 0.0,
        "source_y": 0.0,
        "image_x": [1.0, 2.0],
        "image_y": [3.0, 4.0],
        "macro_magnifications": [2.0, 3.0],
        "time_delays": [0.0, 1.0],
        "num_images": 2,
    }
    inputs.update(overrides)
    node = _ResolvedImageDataNode(**inputs)

    with pytest.raises(ValueError, match=message):
        node.sample_parameters()


@pytest.mark.parametrize(
    ("source_ra", "source_dec", "source_x", "source_y"),
    [
        (20.0, 0.0, 0.5, -0.25),
        (359.9999, 45.0, 0.8, 0.4),
        (120.0, 89.9, 0.1, -0.2),
    ],
)
def test_coordinate_node_recovers_lens_origin_and_image_offsets(
    source_ra,
    source_dec,
    source_x,
    source_y,
):
    """Lens origins and image offsets preserve spherical coordinates."""
    node = _ResolvedCoordinatesNode(
        source_ra=source_ra,
        source_dec=source_dec,
        source_x=source_x,
        source_y=source_y,
        image_x=1.25,
        image_y=-0.75,
        node_label="coordinates",
    )

    state = node.sample_parameters()
    lens = SkyCoord(
        ra=state["coordinates.lens_ra"] * u.deg,
        dec=state["coordinates.lens_dec"] * u.deg,
    )
    source = SkyCoord(ra=source_ra * u.deg, dec=source_dec * u.deg)
    recovered_source = source.transform_to(SkyOffsetFrame(origin=lens))
    assert recovered_source.lon.to_value(u.arcsec) == pytest.approx(source_x, abs=1e-6)
    assert recovered_source.lat.to_value(u.arcsec) == pytest.approx(source_y, abs=1e-6)

    image_ra = source_ra + state["coordinates.ra_offset"]
    image_dec = source_dec + state["coordinates.dec_offset"]
    image = SkyCoord(ra=image_ra * u.deg, dec=image_dec * u.deg)
    recovered_image = image.transform_to(SkyOffsetFrame(origin=lens))
    assert recovered_image.lon.to_value(u.arcsec) == pytest.approx(1.25, abs=1e-6)
    assert recovered_image.lat.to_value(u.arcsec) == pytest.approx(-0.75, abs=1e-6)
    assert 0.0 <= image.ra.deg < 360.0


def test_coordinate_node_rejects_nonfinite_source_metadata():
    """Nonfinite source sky coordinates are rejected."""
    node = _ResolvedCoordinatesNode(
        source_ra=np.nan,
        source_dec=0.0,
        source_x=0.0,
        source_y=0.0,
        image_x=1.0,
        image_y=0.0,
    )
    with pytest.raises(ValueError, match="source RA/Dec must be finite"):
        node.sample_parameters()


def test_macro_magnification_effect_scales_sed_and_bandflux():
    """Macro magnification scales both supported flux representations."""
    effect = _MacroMagnificationEffect(macro_magnification=3.0)
    sed = np.array([[1.0, 2.0], [3.0, 4.0]])
    bandflux = np.array([2.0, 5.0])

    np.testing.assert_array_equal(
        effect.apply(sed, macro_magnification=3.0),
        3.0 * sed,
    )
    np.testing.assert_array_equal(
        effect.apply_bandflux(bandflux, macro_magnification=3.0),
        3.0 * bandflux,
    )
    assert tuple(effect.parameters) == ("macro_magnification",)
    assert effect.rest_frame is False

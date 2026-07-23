from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord, SkyOffsetFrame

from lightcurvelynx.astro_utils.passbands import Passband, PassbandGroup
from lightcurvelynx.effects.basic_effects import ScaleFluxEffect
from lightcurvelynx.effects.effect_model import EffectModel
from lightcurvelynx.graph_state import GraphState
from lightcurvelynx.math_nodes.basic_math_node import BasicMathNode
from lightcurvelynx.math_nodes.given_sampler import GivenValueList
from lightcurvelynx.models._resolved_strong_lens import (
    _MacroMagnificationEffect,
    _ResolvedCoordinatesNode,
    _ResolvedImageDataNode,
)
from lightcurvelynx.models.physical_model import BandfluxModel, BasePhysicalModel, SEDModel
from lightcurvelynx.models.strong_lens_model import ResolvedStrongLensModel
from lightcurvelynx.noise_models.base_noise_models import (
    FluxNoiseModel,
    GivenNoiseModel,
)
from lightcurvelynx.obstable.fake_obs_table import FakeObsTable
from lightcurvelynx.simulate import simulate_lightcurves
from lightcurvelynx.survey_info import SurveyInfo


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


class _DeterministicNoise(FluxNoiseModel):
    def apply_noise(self, bandflux, **kwargs):
        del kwargs
        bandflux = np.asarray(bandflux, dtype=float)
        return bandflux + 0.5, np.full_like(bandflux, 0.25)


class _SynchronousExecutor:
    def map(self, function, iterable):
        return [function(item) for item in iterable]


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


_RESOLVED_OUTER_PARAMETER_NAMES = (
    "ra",
    "dec",
    "redshift",
    "t0",
    "distance",
    "system_id",
    "image_id",
    "source_x",
    "source_y",
    "lens_ra",
    "lens_dec",
    "image_x",
    "image_y",
    "macro_magnification",
    "time_delay",
)


def _strict_source_snapshot(source):
    effect_attributes = (
        "rest_frame_effects",
        "obs_frame_effects",
        "band_pass_effects",
    )
    return {
        "setter_keys": tuple(source.setters),
        "setters": tuple(
            (
                name,
                setter,
                setter.dependency,
                setter.node_name,
            )
            for name, setter in source.setters.items()
        ),
        "effect_lists": tuple(
            (
                attribute,
                getattr(source, attribute),
                tuple(getattr(source, attribute)),
            )
            for attribute in effect_attributes
            if hasattr(source, attribute)
        ),
        "node_pos": source.node_pos,
        "node_string": source.node_string,
    }


def _assert_strict_source_snapshot(source, snapshot):
    assert tuple(source.setters) == snapshot["setter_keys"]
    for name, setter, dependency, node_name in snapshot["setters"]:
        assert source.setters[name] is setter
        assert source.setters[name].dependency is dependency
        assert source.setters[name].node_name == node_name
    for attribute, effect_list, effects in snapshot["effect_lists"]:
        assert getattr(source, attribute) is effect_list
        assert len(effect_list) == len(effects)
        assert all(current is expected for current, expected in zip(effect_list, effects, strict=True))
    assert source.node_pos == snapshot["node_pos"]
    assert source.node_string == snapshot["node_string"]


def _make_source_for_strict_rejection(source_class=_PhaseSEDModel):
    source = source_class(
        ra=20.0,
        dec=10.0,
        redshift=GivenValueList([0.0], stateful=False),
        t0=100.0,
    )
    source.add_effect(ScaleFluxEffect(flux_scale=2.0))
    source.add_effect(_SEDOnlyEffect())
    return source


def _record_effect_application(monkeypatch, effect, method_name, call_order, marker):
    """Record one real effect method without replacing its implementation."""
    original_method = getattr(effect, method_name)

    def recording_method(*args, **kwargs):
        call_order.append(marker)
        return original_method(*args, **kwargs)

    monkeypatch.setattr(effect, method_name, recording_method)


def _make_test_passbands():
    return PassbandGroup(
        [
            Passband(
                np.array(
                    [
                        [4_000.0, 0.5],
                        [5_000.0, 1.0],
                        [6_000.0, 0.5],
                    ]
                ),
                "test",
                "g",
            )
        ]
    )


def _make_three_image_resolved_simulation():
    source = _PhaseBandfluxModel(
        ra=20.0,
        dec=10.0,
        redshift=0.0,
        t0=100.0,
        node_label="source",
    )
    resolved = ResolvedStrongLensModel(
        source,
        source_x=0.0,
        source_y=0.0,
        image_x=[-2.0, 0.0, 2.0],
        image_y=[0.0, 0.0, 0.0],
        macro_magnifications=[1.0, 2.0, 3.0],
        time_delays=[4.0, 0.0, 2.0],
        node_label="resolved",
    )
    state = resolved.sample_parameters()
    lens = SkyCoord(
        ra=state["resolved.lens_ra"][0] * u.deg,
        dec=state["resolved.lens_dec"][0] * u.deg,
    )
    image_positions = [lens.spherical_offsets_by(x * u.arcsec, 0.0 * u.arcsec) for x in [0.0, 2.0, -2.0]]
    return resolved, image_positions, _make_test_passbands()


def _make_uneven_resolved_simulation():
    source = _PhaseBandfluxModel(
        ra=GivenValueList([20.0, 30.0], stateful=False),
        dec=GivenValueList([10.0, -5.0], stateful=False),
        redshift=0.0,
        t0=GivenValueList([100.0, 200.0], stateful=False),
        node_label="source",
    )
    _, resolved = _make_resolved_lens(source=source)
    observations = FakeObsTable(
        {
            "time": [210.0, 210.0],
            "ra": [20.0, 30.0],
            "dec": [10.0, -5.0],
            "filter": ["g", "g"],
        },
        radius=5.0 / 3600.0,
        bandflux_error=0.0,
    )
    survey = SurveyInfo(
        obstable=observations,
        passbands=_make_test_passbands(),
        noise_model=_DeterministicNoise(),
    )
    return resolved, survey


def test_resolved_lens_accepts_direct_caustics_node_outputs():
    """Wire the real optional Caustics node outputs directly into the model."""
    caustics = pytest.importorskip("caustics")
    pytest.importorskip("torch")
    pytest.importorskip("contourpy")
    pytest.importorskip("shapely")

    from astropy.cosmology import FlatLambdaCDM as AstropyFlatLambdaCDM

    from lightcurvelynx.models.caustics_models import (
        CausticsLensImageNode,
        CausticsLensSpec,
        CausticsSourcePositionNode,
    )

    astropy_cosmology = AstropyFlatLambdaCDM(H0=70.0, Om0=0.3)
    cosmology = caustics.FlatLambdaCDM(
        h0=astropy_cosmology.h,
        critical_density_0=astropy_cosmology.critical_density0.to_value(u.Msun / u.Mpc**3),
        Om0=astropy_cosmology.Om0,
    )
    lens = CausticsLensSpec(
        "SIS",
        {"z_l": 0.5, "x0": 0.0, "y0": 0.0, "Rein": 1.0},
    )
    source_position = CausticsSourcePositionNode(
        lens,
        cosmology=cosmology,
        source_redshift=1.5,
        fov=4.0,
        pixelscale=0.1,
        max_fov_expansions=2,
        fov_expansion_factor=2.0,
        pseudo_caustic_points=32,
        pseudo_caustic_epsilon=0.05,
        geometry_tolerance=0.01,
        boundary_tolerance=0.05,
        max_boundary_refinements=4,
        max_attempts=50,
        seed=17,
        node_label="caustics_source",
    )
    images = CausticsLensImageNode(
        lens,
        cosmology=cosmology,
        source_redshift=1.5,
        source_x=source_position.source_x,
        source_y=source_position.source_y,
        expected_num_images=source_position.expected_num_images,
        max_images=4,
        min_images=2,
        fov=4.0,
        fov_multiplier=1.0,
        pixelscale=0.05,
        epsilon=0.01,
        max_depth=8,
        max_fov_expansions=1,
        fov_expansion_factor=2.0,
        max_pixelscale_refinements=1,
        pixelscale_refinement_factor=0.5,
        node_label="caustics_images",
    )
    source = _PhaseSEDModel(
        ra=20.0,
        dec=10.0,
        redshift=0.0,
        t0=100.0,
        node_label="source",
    )
    resolved = ResolvedStrongLensModel(
        source,
        source_x=source_position.source_x,
        source_y=source_position.source_y,
        image_x=images.image_x,
        image_y=images.image_y,
        macro_magnifications=images.macro_magnifications,
        time_delays=images.time_delays,
        num_images=images.num_images,
        node_label="resolved",
    )

    state = resolved.sample_parameters(
        num_samples=1,
        rng_info=np.random.default_rng(19),
    )

    assert state.num_samples >= 2
    assert np.all(np.isfinite(np.atleast_1d(state["resolved.ra"])))
    assert np.all(np.isfinite(np.atleast_1d(state["resolved.dec"])))
    assert np.all(np.atleast_1d(state["resolved.macro_magnification"]) >= 0.0)


def test_resolved_lens_simulation_matches_each_image_footprint():
    """Match survey observations against each resolved image coordinate."""
    resolved, image_positions, passbands = _make_three_image_resolved_simulation()
    observations = FakeObsTable(
        {
            "time": [110.0, 110.0, 110.0],
            "ra": [position.ra.deg for position in image_positions],
            "dec": [position.dec.deg for position in image_positions],
            "filter": ["g", "g", "g"],
        },
        radius=0.5 / 3600.0,
        bandflux_error=0.0,
    )
    survey = SurveyInfo(
        obstable=observations,
        passbands=passbands,
        noise_model=GivenNoiseModel(),
    )

    results = simulate_lightcurves(
        resolved,
        1,
        survey,
        progress_bar=False,
    )

    assert len(results) == 3
    np.testing.assert_array_equal(results["system_id"], [0, 0, 0])
    np.testing.assert_array_equal(results["image_id"], [0, 1, 2])
    np.testing.assert_array_equal(results["t0"], [100.0, 102.0, 104.0])
    np.testing.assert_array_equal(results["nobs"], [1, 1, 1])
    np.testing.assert_allclose(
        results["ra"],
        [position.ra.deg for position in image_positions],
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        results["dec"],
        [position.dec.deg for position in image_positions],
        rtol=0.0,
        atol=1e-12,
    )
    for image_id in range(3):
        lightcurve = results["lightcurve"].iloc[image_id]
        assert lightcurve["obs_idx"].tolist() == [image_id]
    np.testing.assert_array_equal(
        results["lightcurve.flux_perfect"],
        [22.0, 27.0, 7.0],
    )


def test_resolved_lens_simulation_applies_noise_and_retains_unmatched_image():
    """Apply noise per image and keep provenance when one image is unseen."""
    resolved, image_positions, passbands = _make_three_image_resolved_simulation()

    def make_survey(pointings):
        observations = FakeObsTable(
            {
                "time": [110.0, 110.0, 110.0],
                "ra": [position.ra.deg for position in pointings],
                "dec": [position.dec.deg for position in pointings],
                "filter": ["g", "g", "g"],
            },
            radius=0.5 / 3600.0,
            bandflux_error=0.0,
        )
        return SurveyInfo(
            obstable=observations,
            passbands=passbands,
            noise_model=_DeterministicNoise(),
        )

    results = simulate_lightcurves(
        resolved,
        1,
        make_survey(image_positions),
        progress_bar=False,
    )

    np.testing.assert_array_equal(results["nobs"], [1, 1, 1])
    np.testing.assert_array_equal(
        results["lightcurve.flux"],
        [22.5, 27.5, 7.5],
    )
    np.testing.assert_array_equal(
        results["lightcurve.fluxerr"],
        [0.25, 0.25, 0.25],
    )

    displaced_pointings = list(image_positions)
    displaced_pointings[1] = image_positions[1].spherical_offsets_by(
        0.0 * u.arcsec,
        20.0 * u.arcsec,
    )
    unmatched = simulate_lightcurves(
        resolved,
        1,
        make_survey(displaced_pointings),
        progress_bar=False,
    )

    assert len(unmatched) == 3
    np.testing.assert_array_equal(unmatched["system_id"], [0, 0, 0])
    np.testing.assert_array_equal(unmatched["image_id"], [0, 1, 2])
    np.testing.assert_array_equal(unmatched["nobs"], [1, 0, 1])
    assert unmatched["lightcurve"].iloc[1] is None
    for image_id, expected_flux in ((0, 22.5), (2, 7.5)):
        lightcurve = unmatched["lightcurve"].iloc[image_id]
        assert lightcurve["obs_idx"].tolist() == [image_id]
        np.testing.assert_array_equal(lightcurve["flux"], [expected_flux])
        np.testing.assert_array_equal(lightcurve["fluxerr"], [0.25])


def test_resolved_lens_simulation_batching_matches_serial_and_threaded():
    """Preserve global image rows across uneven synchronous/threaded batches."""
    resolved, survey = _make_uneven_resolved_simulation()
    simulation_kwargs = {
        "param_cols": ["resolved.macro_magnification"],
        "progress_bar": False,
    }
    serial = simulate_lightcurves(
        resolved,
        2,
        survey,
        **simulation_kwargs,
    )
    batched = simulate_lightcurves(
        resolved,
        2,
        survey,
        executor=_SynchronousExecutor(),
        batch_size=1,
        **simulation_kwargs,
    )

    threaded_resolved, threaded_survey = _make_uneven_resolved_simulation()
    threaded_resolved.set_graph_positions()
    with ThreadPoolExecutor(max_workers=2) as executor:
        threaded = simulate_lightcurves(
            threaded_resolved,
            2,
            threaded_survey,
            executor=executor,
            batch_size=1,
            **simulation_kwargs,
        )

    for results in (serial, batched, threaded):
        assert len(results) == 5
        np.testing.assert_array_equal(results["system_id"], [0, 0, 1, 1, 1])
        np.testing.assert_array_equal(results["image_id"], [0, 1, 0, 1, 2])
        np.testing.assert_array_equal(results["t0"], [100.0, 102.0, 200.0, 201.0, 202.0])
        np.testing.assert_array_equal(
            results["resolved_macro_magnification"],
            [2.0, 3.0, 4.0, 6.0, 5.0],
        )
        np.testing.assert_array_equal(results["nobs"], [1, 1, 1, 1, 1])

    for actual in (batched, threaded):
        for column in ("system_id", "image_id", "nobs"):
            np.testing.assert_array_equal(actual[column], serial[column])
        for column in ("ra", "dec", "t0", "resolved_macro_magnification"):
            np.testing.assert_allclose(actual[column], serial[column], rtol=0.0, atol=1e-12)
        for row_index in range(5):
            actual_lightcurve = actual["lightcurve"].iloc[row_index]
            serial_lightcurve = serial["lightcurve"].iloc[row_index]
            for column in ("flux_perfect", "flux"):
                np.testing.assert_array_equal(
                    actual_lightcurve[column],
                    serial_lightcurve[column],
                )


def test_resolved_lens_simulation_replay_uses_expanded_state(monkeypatch):
    """Replay realized image rows without resampling requested lens systems."""
    resolved, survey = _make_uneven_resolved_simulation()
    simulation_kwargs = {
        "param_cols": ["resolved.macro_magnification"],
        "progress_bar": False,
    }
    results = simulate_lightcurves(
        resolved,
        2,
        survey,
        **simulation_kwargs,
    )
    state = GraphState.from_list(results["params"].values)
    assert state.num_samples == 5

    def fail_if_called(*args, **kwargs):
        del args, kwargs
        pytest.fail("model resampled during replay")

    monkeypatch.setattr(resolved, "sample_parameters", fail_if_called)
    replay = simulate_lightcurves(
        resolved,
        state.num_samples,
        survey,
        graph_state=state,
        **simulation_kwargs,
    )

    for column in ("id", "system_id", "image_id", "nobs"):
        np.testing.assert_array_equal(replay[column], results[column])
    for column in ("ra", "dec", "t0", "z", "resolved_macro_magnification"):
        np.testing.assert_allclose(replay[column], results[column], rtol=0.0, atol=1e-12)
    for row_index in range(state.num_samples):
        np.testing.assert_array_equal(
            replay["lightcurve"].iloc[row_index]["flux_perfect"],
            results["lightcurve"].iloc[row_index]["flux_perfect"],
        )

    with pytest.raises(
        ValueError,
        match="Graph state has 5 samples, but simulation is set to 2 samples",
    ):
        simulate_lightcurves(
            resolved,
            2,
            survey,
            graph_state=state,
            **simulation_kwargs,
        )


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


@pytest.mark.parametrize(
    "reserved_name",
    ["base_ra", "base_dec", "base_t0", "macro_magnification"],
)
def test_resolved_lens_rejects_reserved_source_class_attributes_without_mutation(reserved_name):
    """Preflight source class attributes before changing any owned source state."""
    source_class = type(
        f"_SourceWithReserved{reserved_name.title().replace('_', '')}",
        (_PhaseSEDModel,),
        {reserved_name: object()},
    )
    source = _make_source_for_strict_rejection(source_class)
    before = _strict_source_snapshot(source)

    with pytest.raises(Exception) as exc_info:
        ResolvedStrongLensModel(source, **_resolved_constructor_kwargs())

    _assert_strict_source_snapshot(source, before)
    assert type(exc_info.value) is ValueError
    assert "source_model" in str(exc_info.value)
    assert "class attribute" in str(exc_info.value)
    assert reserved_name in str(exc_info.value)


@pytest.mark.parametrize("parameter_name", _RESOLVED_OUTER_PARAMETER_NAMES)
def test_resolved_lens_rejects_outer_class_attributes_without_source_mutation(parameter_name):
    """Preflight every outer parameter class attribute before source decoration."""
    wrapper_class = type(
        f"_ResolvedWithReserved{parameter_name.title().replace('_', '')}",
        (ResolvedStrongLensModel,),
        {parameter_name: object()},
    )
    source = _make_source_for_strict_rejection()
    before = _strict_source_snapshot(source)

    with pytest.raises(Exception) as exc_info:
        wrapper_class(source, **_resolved_constructor_kwargs())

    _assert_strict_source_snapshot(source, before)
    assert type(exc_info.value) is ValueError
    assert f"outer parameter '{parameter_name}'" in str(exc_info.value)
    assert "class attribute" in str(exc_info.value)


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

import numpy as np
import pytest

from lightcurvelynx.effects.effect_model import EffectModel
from lightcurvelynx.math_nodes.given_sampler import GivenValueList
from lightcurvelynx.models.physical_model import BandfluxModel, SEDModel
from lightcurvelynx.models.strong_lens_model import UnresolvedStrongLensModel


class _LinearTimeSEDModel(SEDModel):
    """Return a deterministic time- and wavelength-dependent SED."""

    def compute_sed(self, times, wavelengths, graph_state, **kwargs):
        """Return ``time + wavelength / 1000`` in nJy."""
        del graph_state, kwargs
        times = np.asarray(times, dtype=float)
        wavelengths = np.asarray(wavelengths, dtype=float)
        return times[:, np.newaxis] + wavelengths[np.newaxis, :] / 1_000.0


class _RecordingLinearTimeSEDModel(_LinearTimeSEDModel):
    """Record observer- and rest-frame coordinates around real SED evaluation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.evaluate_calls = []
        self.compute_calls = []

    def evaluate_sed(self, times, wavelengths, *args, **kwargs):
        """Record observer-frame inputs before running the real SED pipeline."""
        self.evaluate_calls.append((np.asarray(times).copy(), np.asarray(wavelengths).copy()))
        return super().evaluate_sed(times, wavelengths, *args, **kwargs)

    def compute_sed(self, times, wavelengths, graph_state, **kwargs):
        """Record coordinates before evaluating the same analytic SED."""
        self.compute_calls.append((np.asarray(times).copy(), np.asarray(wavelengths).copy()))
        return super().compute_sed(times, wavelengths, graph_state, **kwargs)


class _LinearTimeBandfluxModel(BandfluxModel):
    """Return deterministic time- and filter-dependent bandfluxes."""

    _FILTER_OFFSETS = {"g": 1.0, "r": 10.0}

    def compute_bandflux(self, times, filter, state):
        """Return time plus the selected filter's fixed offset in nJy."""
        del state
        return np.asarray(times, dtype=float) + self._FILTER_OFFSETS[filter]


class _RecordingAdditiveEffect(EffectModel):
    """Add a fixed offset and record the coordinates passed by the model."""

    def __init__(self, offset, *, rest_frame=False):
        """Configure the additive offset and coordinate frame."""
        super().__init__(rest_frame=rest_frame)
        self.offset = offset
        self.calls = []

    def apply(self, flux_density, times=None, wavelengths=None, **kwargs):
        """Record SED coordinates and add the fixed offset."""
        del kwargs
        self.calls.append((np.asarray(times).copy(), np.asarray(wavelengths).copy()))
        return np.asarray(flux_density) + self.offset

    def apply_bandflux(self, bandfluxes, *, times=None, filters=None, **kwargs):
        """Record bandflux coordinates and add the fixed offset."""
        del kwargs
        self.calls.append((np.asarray(times).copy(), np.asarray(filters).copy()))
        return np.asarray(bandfluxes) + self.offset


class _BoundedLinearTimeSEDModel(_LinearTimeSEDModel):
    """Return fixed wavelength bounds while recording the forwarded state."""

    def minwave(self, graph_state=None):
        """Record the state and return the fixed minimum wavelength."""
        self.minwave_state = graph_state
        return 900.0

    def maxwave(self, graph_state=None):
        """Record the state and return the fixed maximum wavelength."""
        self.maxwave_state = graph_state
        return 2_100.0


def _make_sed_lens(
    magnifications=(2.0, 3.0),
    delays=(12.0, 10.0),
    num_images=None,
    *,
    source=None,
):
    """Build a deterministic SED source and unresolved lens wrapper."""
    if source is None:
        source = _LinearTimeSEDModel(
            redshift=0.0,
            t0=0.0,
            node_label="source",
        )
    lens = UnresolvedStrongLensModel(
        source,
        macro_magnifications=magnifications,
        time_delays=delays,
        num_images=num_images,
        node_label="lens",
    )
    return source, lens


def _make_bandflux_lens(
    magnifications=(2.0, 3.0),
    delays=(12.0, 10.0),
    num_images=None,
):
    """Build a deterministic bandflux source and unresolved lens wrapper."""
    source = _LinearTimeBandfluxModel(
        redshift=0.0,
        t0=0.0,
        node_label="source",
    )
    lens = UnresolvedStrongLensModel(
        source,
        macro_magnifications=magnifications,
        time_delays=delays,
        num_images=num_images,
        node_label="lens",
    )
    return source, lens


def _sample_active_images(lens):
    """Sample one lens realization and return its validated active images."""
    return lens._get_active_images(lens.sample_parameters())


def test_rejects_non_physical_source():
    """Reject a source that is not a physical model before graph construction."""
    with pytest.raises(TypeError, match="source_model must be a BasePhysicalModel"):
        UnresolvedStrongLensModel(
            object(),
            macro_magnifications=[1.0, 2.0],
            time_delays=[0.0, 1.0],
        )


def test_registers_lens_parameters_and_links_source_metadata():
    """Register lens parameters and link default wrapper metadata to the child."""
    source = _LinearTimeSEDModel(
        ra=12.5,
        dec=-4.0,
        redshift=0.0,
        t0=0.0,
        distance=900.0,
        node_label="source",
    )
    lens = UnresolvedStrongLensModel(
        source,
        macro_magnifications=[2.0, 3.0],
        time_delays=[12.0, 10.0],
        node_label="lens",
    )

    state = lens.sample_parameters()

    assert lens.objects == [source]
    assert lens.source_model is lens.objects[0]
    assert lens.num_objects == 1
    assert lens.apply_redshift is False
    lens_parameter_names = {"macro_magnifications", "time_delays", "num_images"}
    assert lens_parameter_names <= set(lens.list_params())
    assert all(lens.setters[name].allow_gradient is False for name in lens_parameter_names)

    metadata_names = ("ra", "dec", "redshift", "t0", "distance")
    for name in metadata_names:
        assert lens.setters[name].dependency is source
        assert lens.setters[name].value == name
        assert state["lens"][name] == state["source"][name]

    np.testing.assert_array_equal(state["lens"]["macro_magnifications"], [2.0, 3.0])
    np.testing.assert_array_equal(state["lens"]["time_delays"], [12.0, 10.0])
    assert state["lens"]["num_images"] is None


def test_explicit_wrapper_metadata_overrides_remain_outer_only():
    """Keep explicit wrapper metadata separate from the child's own values."""
    child_metadata = {
        "ra": 1.0,
        "dec": 2.0,
        "redshift": 0.0,
        "t0": 0.0,
        "distance": 3.0,
    }
    wrapper_metadata = {
        "ra": 101.0,
        "dec": -22.0,
        "redshift": 0.5,
        "t0": 40.0,
        "distance": 300.0,
    }
    source = _LinearTimeSEDModel(**child_metadata, node_label="source")
    lens = UnresolvedStrongLensModel(
        source,
        macro_magnifications=[2.0, 3.0],
        time_delays=[12.0, 10.0],
        node_label="lens",
        **wrapper_metadata,
    )

    state = lens.sample_parameters()

    for name in child_metadata:
        assert state["source"][name] == child_metadata[name]
        assert state["lens"][name] == wrapper_metadata[name]
        assert state["source"][name] != state["lens"][name]


def test_wavelength_bounds_delegate_to_source():
    """Forward the exact wavelength-bound state object to the source."""
    source = _BoundedLinearTimeSEDModel(
        redshift=0.0,
        t0=0.0,
        node_label="source",
    )
    _, lens = _make_sed_lens(source=source)
    state = object()

    assert lens.minwave(state) == 900.0
    assert lens.maxwave(state) == 2_100.0
    assert source.minwave_state is state
    assert source.maxwave_state is state


@pytest.mark.parametrize(
    ("magnifications", "delays", "num_images", "message"),
    [
        ([[1.0, 2.0]], [0.0, 1.0], None, "one-dimensional"),
        ([1.0, 2.0], [[0.0, 1.0]], None, "one-dimensional"),
        ([1.0, 2.0], [0.0], None, "same length"),
        ([1.0, 2.0], [0.0, 1.0], [2], "scalar"),
        ([1.0, 2.0], [0.0, 1.0], 1.5, "integer"),
        ([1.0, 2.0], [0.0, 1.0], 1, "at least two"),
        ([1.0, 2.0], [0.0, 1.0], 3, "exceeds"),
        ([1.0, np.nan], [0.0, 1.0], None, "finite"),
        ([1.0, -1.0], [0.0, 1.0], None, "non-negative"),
        ([0.0, 0.0], [0.0, 1.0], None, "positive"),
        ([1.0, 2.0], [0.0, np.inf], None, "finite"),
    ],
)
def test_rejects_invalid_active_images(magnifications, delays, num_images, message):
    """Reject every documented invalid active-image realization."""
    _, lens = _make_sed_lens(magnifications, delays, num_images)
    state = lens.sample_parameters()

    with pytest.raises(ValueError, match=message):
        lens.evaluate_sed([10.0], [1_000.0], state)


def test_num_images_none_activates_all_entries():
    """Treat every fixed-width array entry as active when the count is None."""
    _, lens = _make_sed_lens(
        magnifications=[1.0, 2.0, 3.0],
        delays=[4.0, 2.0, 3.0],
    )

    magnifications, relative_delays = _sample_active_images(lens)

    np.testing.assert_array_equal(magnifications, [2.0, 3.0, 1.0])
    np.testing.assert_array_equal(relative_delays, [0.0, 1.0, 2.0])


def test_inactive_padding_is_ignored_after_prefix_truncation():
    """Ignore float-coercible invalid values outside the active prefix."""
    _, lens = _make_sed_lens(
        magnifications=[2.0, 3.0, np.nan, -1.0],
        delays=[12.0, 10.0, np.inf, np.nan],
        num_images=2,
    )

    magnifications, relative_delays = _sample_active_images(lens)

    np.testing.assert_array_equal(magnifications, [3.0, 2.0])
    np.testing.assert_array_equal(relative_delays, [0.0, 2.0])


def test_common_delay_offsets_normalize_identically():
    """Remove an arbitrary common delay offset before image evaluation."""
    _, reference_lens = _make_sed_lens(
        magnifications=[2.0, 3.0, 4.0],
        delays=[2.0, 4.0, 3.0],
    )
    _, offset_lens = _make_sed_lens(
        magnifications=[2.0, 3.0, 4.0],
        delays=[102.0, 104.0, 103.0],
    )

    reference_images = _sample_active_images(reference_lens)
    offset_images = _sample_active_images(offset_lens)

    np.testing.assert_array_equal(offset_images[0], reference_images[0])
    np.testing.assert_array_equal(offset_images[1], reference_images[1])


def test_unsorted_delays_reorder_paired_magnifications():
    """Sort delays while retaining each delay's paired magnification."""
    _, lens = _make_sed_lens(
        magnifications=[20.0, 10.0, 30.0],
        delays=[2.0, 0.0, 1.0],
    )

    magnifications, relative_delays = _sample_active_images(lens)

    np.testing.assert_array_equal(magnifications, [10.0, 30.0, 20.0])
    np.testing.assert_array_equal(relative_delays, [0.0, 1.0, 2.0])


def test_equal_delays_preserve_original_order():
    """Use stable sorting so equal-delay images retain their input order."""
    _, lens = _make_sed_lens(
        magnifications=[20.0, 10.0, 30.0, 40.0],
        delays=[2.0, 0.0, 0.0, 1.0],
    )

    magnifications, relative_delays = _sample_active_images(lens)

    np.testing.assert_array_equal(magnifications, [10.0, 30.0, 40.0, 20.0])
    np.testing.assert_array_equal(relative_delays, [0.0, 0.0, 1.0, 2.0])


def test_evaluate_sed_returns_exact_weighted_image_sum():
    """Return the analytic weighted sum of the shifted source SEDs."""
    _, lens = _make_sed_lens()
    state = lens.sample_parameters()
    times = np.array([10.0, 11.0])
    wavelengths = np.array([1_000.0, 2_000.0])

    result = lens.evaluate_sed(times, wavelengths, state)

    zero_delay_source = np.array([[11.0, 12.0], [12.0, 13.0]])
    two_day_delay_source = np.array([[9.0, 10.0], [10.0, 11.0]])
    expected = 3.0 * zero_delay_source + 2.0 * two_day_delay_source
    assert result.shape == (2, 2)
    np.testing.assert_allclose(result, expected)
    np.testing.assert_allclose(result, [[51.0, 56.0], [56.0, 61.0]])


def test_evaluate_sed_child_owns_nonzero_redshift_conversion_once():
    """Apply redshift once in the child around the unresolved image sum."""
    source = _RecordingLinearTimeSEDModel(
        redshift=1.0,
        t0=4.0,
        node_label="source",
    )
    _, lens = _make_sed_lens(source=source)
    state = lens.sample_parameters()
    observer_times = np.array([10.0, 14.0])
    observer_wavelengths = np.array([1_000.0, 3_000.0])

    result = lens.evaluate_sed(observer_times, observer_wavelengths, state)

    assert source.apply_redshift is True
    assert lens.apply_redshift is False
    assert len(source.evaluate_calls) == 1
    shifted_observer_times, shifted_observer_wavelengths = source.evaluate_calls[0]
    np.testing.assert_array_equal(shifted_observer_times, [10.0, 14.0, 8.0, 12.0])
    np.testing.assert_array_equal(
        shifted_observer_wavelengths,
        [1_000.0, 3_000.0],
    )
    assert len(source.compute_calls) == 1
    rest_times, rest_wavelengths = source.compute_calls[0]
    image_major_rest_times = np.array([7.0, 9.0, 6.0, 8.0])
    np.testing.assert_array_equal(rest_times, np.sort(image_major_rest_times))
    np.testing.assert_array_equal(rest_wavelengths, [500.0, 1_500.0])

    rest_frame_sed = np.array(
        [
            [7.5, 8.5],
            [9.5, 10.5],
            [6.5, 7.5],
            [8.5, 9.5],
        ]
    )
    observer_frame_sed = 2.0 * rest_frame_sed
    expected = 3.0 * observer_frame_sed[:2] + 2.0 * observer_frame_sed[2:]
    np.testing.assert_allclose(result, expected)
    np.testing.assert_allclose(result, [[71.0, 81.0], [91.0, 101.0]])


def test_evaluate_bandfluxes_preserves_image_major_filter_alignment():
    """Align tiled filters with shifted times before weighted bandflux summation."""
    _, lens = _make_bandflux_lens()
    state = lens.sample_parameters()
    times = np.array([10.0, 11.0])
    filters = np.array(["g", "r"])

    result = lens.evaluate_bandfluxes(None, times, filters, state)

    zero_delay_source = np.array([11.0, 21.0])
    two_day_delay_source = np.array([9.0, 19.0])
    expected = 3.0 * zero_delay_source + 2.0 * two_day_delay_source
    assert result.shape == (2,)
    np.testing.assert_allclose(result, expected)
    np.testing.assert_allclose(result, [51.0, 101.0])


def test_bandflux_child_rejects_sed_evaluation():
    """Reject SED evaluation when the wrapped source is bandflux-only."""
    _, lens = _make_bandflux_lens()
    state = lens.sample_parameters()

    with pytest.raises(TypeError, match="BandfluxModel, which does not support SED evaluation"):
        lens.evaluate_sed([10.0], [1_000.0], state)


def test_sed_child_effect_sees_shifted_coordinates_before_summation():
    """Delegate a rest-frame SED effect to the child at shifted coordinates."""
    source, lens = _make_sed_lens()
    effect = _RecordingAdditiveEffect(4.0, rest_frame=True)
    lens.add_effect(effect)
    state = lens.sample_parameters()
    times = np.array([10.0, 11.0])
    wavelengths = np.array([1_000.0, 2_000.0])

    result = lens.evaluate_sed(times, wavelengths, state)

    assert source.rest_frame_effects == [effect]
    assert lens.rest_frame_effects == []
    assert len(effect.calls) == 1
    np.testing.assert_array_equal(effect.calls[0][0], [10.0, 11.0, 8.0, 9.0])
    np.testing.assert_array_equal(effect.calls[0][1], wavelengths)
    np.testing.assert_allclose(result, [[71.0, 76.0], [76.0, 81.0]])


def test_sed_wrapper_effect_sees_original_coordinates_after_summation():
    """Apply a wrapper observer-frame SED effect once after image summation."""
    source, lens = _make_sed_lens()
    effect = _RecordingAdditiveEffect(4.0)
    lens.add_effect(effect)
    state = lens.sample_parameters()
    times = np.array([10.0, 11.0])
    wavelengths = np.array([1_000.0, 2_000.0])

    result = lens.evaluate_sed(times, wavelengths, state)

    assert source.obs_frame_effects == []
    assert lens.obs_frame_effects == [effect]
    assert len(effect.calls) == 1
    np.testing.assert_array_equal(effect.calls[0][0], times)
    np.testing.assert_array_equal(effect.calls[0][1], wavelengths)
    np.testing.assert_allclose(result, [[55.0, 60.0], [60.0, 65.0]])


def test_bandflux_child_effect_sees_shifted_coordinates_before_summation():
    """Delegate a rest-frame bandflux effect to the child at shifted coordinates."""
    source, lens = _make_bandflux_lens()
    effect = _RecordingAdditiveEffect(4.0, rest_frame=True)
    lens.add_effect(effect)
    state = lens.sample_parameters()
    times = np.array([10.0, 11.0])
    filters = np.array(["g", "r"])

    result = lens.evaluate_bandfluxes(None, times, filters, state)

    assert source.band_pass_effects == [effect]
    assert lens.rest_frame_effects == []
    assert len(effect.calls) == 1
    np.testing.assert_array_equal(effect.calls[0][0], [10.0, 11.0, 8.0, 9.0])
    np.testing.assert_array_equal(effect.calls[0][1], ["g", "r", "g", "r"])
    np.testing.assert_allclose(result, [71.0, 121.0])


def test_bandflux_wrapper_effect_sees_original_coordinates_after_summation():
    """Apply a wrapper observer bandflux effect once after image summation."""
    source, lens = _make_bandflux_lens()
    effect = _RecordingAdditiveEffect(4.0)
    lens.add_effect(effect)
    state = lens.sample_parameters()
    times = np.array([10.0, 11.0])
    filters = np.array(["g", "r"])

    result = lens.evaluate_bandfluxes(None, times, filters, state)

    assert source.band_pass_effects == []
    assert lens.obs_frame_effects == [effect]
    assert len(effect.calls) == 1
    np.testing.assert_array_equal(effect.calls[0][0], times)
    np.testing.assert_array_equal(effect.calls[0][1], filters)
    np.testing.assert_allclose(result, [55.0, 105.0])


def test_evaluate_sed_multiple_samples_returns_explicit_values():
    """Evaluate independent image arrays into an explicit S-by-T-by-W result."""
    magnifications = GivenValueList(
        [[2.0, 3.0], [1.0, 4.0]],
        stateful=False,
        node_label="magnification_values",
    )
    delays = GivenValueList(
        [[12.0, 10.0], [1.0, 4.0]],
        stateful=False,
        node_label="delay_values",
    )
    _, lens = _make_sed_lens(magnifications, delays)
    state = lens.sample_parameters(num_samples=2)
    times = np.array([10.0, 11.0])
    wavelengths = np.array([1_000.0, 2_000.0])

    result = lens.evaluate_sed(times, wavelengths, state)

    expected = np.array(
        [
            [[51.0, 56.0], [56.0, 61.0]],
            [[43.0, 48.0], [48.0, 53.0]],
        ]
    )
    assert result.shape == (2, 2, 2)
    np.testing.assert_allclose(result, expected)


def test_evaluate_bandfluxes_multiple_samples_returns_explicit_values():
    """Evaluate independent image arrays into an explicit S-by-T bandflux result."""
    magnifications = GivenValueList(
        [[2.0, 3.0], [1.0, 4.0]],
        stateful=False,
        node_label="magnification_values",
    )
    delays = GivenValueList(
        [[12.0, 10.0], [1.0, 4.0]],
        stateful=False,
        node_label="delay_values",
    )
    _, lens = _make_bandflux_lens(magnifications, delays)
    state = lens.sample_parameters(num_samples=2)
    times = np.array([10.0, 11.0])
    filters = np.array(["g", "r"])

    result = lens.evaluate_bandfluxes(None, times, filters, state)

    expected = np.array([[51.0, 101.0], [43.0, 93.0]])
    assert result.shape == (2, 2)
    np.testing.assert_allclose(result, expected)

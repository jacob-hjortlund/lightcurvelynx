import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord, SkyOffsetFrame

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

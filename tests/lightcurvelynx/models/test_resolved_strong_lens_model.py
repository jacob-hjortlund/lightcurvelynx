import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord, SkyOffsetFrame

from lightcurvelynx.math_nodes.given_sampler import GivenValueList
from lightcurvelynx.models._resolved_strong_lens import (
    _MacroMagnificationEffect,
    _ResolvedCoordinatesNode,
    _ResolvedImageDataNode,
)


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

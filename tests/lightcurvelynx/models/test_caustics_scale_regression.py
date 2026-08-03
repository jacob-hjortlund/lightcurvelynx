import numpy as np
import pytest

from lightcurvelynx.models.caustics_models import (
    CausticsLensSpec,
    CausticsSourcePositionNode,
)


def test_scale_aware_defaults_certify_observed_micro_caustic():
    """Keep a valid micro-caustic above its realized geometry tolerance."""
    caustics = pytest.importorskip("caustics")
    pytest.importorskip("contourpy")
    pytest.importorskip("shapely")
    pytest.importorskip("torch")

    einstein_radius = 1.1240789823680739e-6
    lens = CausticsLensSpec(
        "SIE",
        {
            "z_l": 1.4719418280655212,
            "x0": 0.0,
            "y0": 0.0,
            "q": 0.5259388787400523,
            "phi": 1.2066416130049746,
            "Rein": einstein_radius,
            "s": 0.0,
        },
    )
    node = CausticsSourcePositionNode(
        lens,
        cosmology=caustics.FlatLambdaCDM(name="cosmology"),
        source_redshift=1.4719595680293527,
        fov=None,
        fov_expansion_factor=1.25,
        pixelscale=0.05,
        pixelscale_fraction=0.02,
        pseudo_caustic_points=512,
        seed=42,
        node_label="micro_caustic_source",
    )
    values = {
        "source_redshift": 1.4719595680293527,
        "lens_z_l": 1.4719418280655212,
        "lens_x0": 0.0,
        "lens_y0": 0.0,
        "lens_q": 0.5259388787400523,
        "lens_phi": 1.2066416130049746,
        "lens_Rein": einstein_radius,
        "lens_s": 0.0,
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
    ) = node._region_for_one_lens(values, sample_index=0)
    (
        characteristic_scale,
        realized_pixelscale,
        realized_epsilon,
        realized_geometry_tolerance,
        realized_boundary_tolerance,
    ) = angular_settings

    assert geometry_adapter is not None
    assert adapter_values
    assert previous_geometry.caustic_curves
    assert geometry.caustic_curves
    assert characteristic_scale == pytest.approx(einstein_radius)
    assert realized_pixelscale == pytest.approx(0.02 * einstein_radius)
    assert realized_epsilon == pytest.approx(1.0e-5 * einstein_radius)
    assert realized_geometry_tolerance == pytest.approx(1.0e-6 * einstein_radius)
    assert realized_boundary_tolerance == pytest.approx(1.0e-4 * einstein_radius)
    assert np.isfinite(region.area)
    assert region.area > 0.0
    assert np.isfinite(uncertainty)
    assert uncertainty <= realized_boundary_tolerance
    assert 1 <= refinements <= node.max_boundary_refinements

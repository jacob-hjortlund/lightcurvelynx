from types import SimpleNamespace

import pytest

from lightcurvelynx.models import caustics_models


class _FakeSIS:
    """Expose a required node-owned source redshift in the SIS signature."""

    def __init__(self, cosmology, z_l, z_s, x0, y0, Rein, s=0.0, name=None):
        pass


@pytest.fixture
def fake_caustics_registry(monkeypatch):
    """Replace the optional Caustics import with an explicit constructor."""
    registry = SimpleNamespace(SIS=_FakeSIS)
    monkeypatch.setattr(caustics_models, "_import_caustics", lambda: registry)
    return registry


def test_lens_spec_accepts_required_node_owned_source_redshift(fake_caustics_registry):
    """Treat required ``z_s`` as supplied later by the consuming graph node."""
    del fake_caustics_registry

    spec = caustics_models.CausticsLensSpec(
        "SIS",
        {
            "z_l": 0.5,
            "x0": 0.0,
            "y0": 0.0,
            "Rein": 1.0,
        },
    )

    assert spec.model == "SIS"
    assert "z_s" not in spec.parameters

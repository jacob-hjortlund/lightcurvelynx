import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest

from lightcurvelynx.graph_state import GraphState
from lightcurvelynx.models import caustics_models
from lightcurvelynx.models._caustics import image_recovery as caustics_image_recovery
from lightcurvelynx.models._caustics import lens_system as caustics_lens_system
from lightcurvelynx.models._caustics import runtime as caustics_runtime
from lightcurvelynx.models._caustics import source_geometry as caustics_source_geometry

_SOURCE_OUTPUTS = (
    "source_x",
    "source_y",
    "strong_lensing_area",
    "sampling_attempts",
    "expected_num_images",
    "critical_curve_fov",
    "boundary_uncertainty",
    "source_boundary_clearance",
    "boundary_refinements",
)

_IMAGE_OUTPUTS = (
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
)


def test_private_caustics_runtime_module_is_available():
    """Keep the private runtime integration importable by its stable path."""
    assert importlib.util.find_spec("lightcurvelynx.models._caustics.runtime") is not None


def test_private_caustics_lens_system_module_is_available():
    """Keep the private lens-system integration importable by its stable path."""
    assert importlib.util.find_spec("lightcurvelynx.models._caustics.lens_system") is not None


def test_private_caustics_source_geometry_module_is_available():
    """Keep the private source-geometry integration importable by its stable path."""
    assert importlib.util.find_spec("lightcurvelynx.models._caustics.source_geometry") is not None


def test_private_caustics_image_recovery_module_is_available():
    """Keep the private image-recovery integration importable by its stable path."""
    assert importlib.util.find_spec("lightcurvelynx.models._caustics.image_recovery") is not None


def test_caustics_public_facade_keeps_identity_without_private_helper_aliases():
    """Expose only stable public identities from the Caustics façade."""
    public_classes = (
        caustics_models.CausticsLensSpec,
        caustics_models.CausticsSourcePositionNode,
        caustics_models.CausticsLensImageNode,
    )

    assert caustics_models.__all__ == [
        "CausticsLensImageNode",
        "CausticsLensSpec",
        "CausticsSourcePositionNode",
    ]
    assert all(cls.__module__ == "lightcurvelynx.models.caustics_models" for cls in public_classes)
    for moved_name in (
        "_import_caustics",
        "_build_lens_system",
        "_find_all_caustics",
        "_recovery_image_seeds",
        "_GeometryAdapter",
        "_BoundaryGeometry",
    ):
        assert not hasattr(caustics_models, moved_name)


class _FakeSIS:
    """Expose a required node-owned source redshift in the SIS signature."""

    def __init__(self, cosmology, z_l, z_s, x0, y0, Rein, s=0.0, name=None):
        pass


class _FakeExternalShear:
    """Mirror the Caustics 1.7 Cartesian signature and angular ``**kwargs``."""

    def __init__(
        self,
        cosmology,
        z_l,
        z_s,
        x0,
        y0,
        gamma_1=None,
        gamma_2=None,
        parametrization="cartesian",
        s=0.0,
        name=None,
        **kwargs,
    ):
        pass


class _FakeSinglePlane:
    """Expose the recursive SinglePlane constructor signature."""

    def __init__(self, cosmology, z_l, z_s, lenses, name=None):
        pass


class _FakeMassSheet:
    """Expose the registered affine MassSheet constructor signature."""

    def __init__(self, cosmology, z_l, z_s, x0, y0, kappa, name=None):
        pass


class _FakeTensor:
    """Provide the tensor-to-NumPy protocol used by recovery certification."""

    def __init__(self, values):
        self.values = np.asarray(values, dtype=float)

    def detach(self):
        """Return this already detached tensor double."""
        return self

    def cpu(self):
        """Return this already CPU-resident tensor double."""
        return self

    def numpy(self):
        """Return the stored NumPy representation."""
        return self.values

    def __getitem__(self, key):
        """Return a tensor double for indexed root-refinement values."""
        return _FakeTensor(self.values[key])


class _FakeTorch:
    """Provide tensor conversion and absolute-value operations under test."""

    float64 = np.float64

    @staticmethod
    def as_tensor(values, dtype=None):
        """Convert values to the minimal tensor double."""
        del dtype
        return _FakeTensor(values)

    @staticmethod
    def abs(values):
        """Return elementwise absolute values through the tensor protocol."""
        return _FakeTensor(np.abs(values.values))


class _FakeGeometryAdapter:
    """Supply deterministic geometry capabilities for node unit tests."""

    def search_center(self, values):
        """Return the fixed numerical grid center."""
        del values
        return (0.0, 0.0)

    def initial_fov(self, values):
        """Return the fixed adapter-derived full field of view."""
        del values
        return 4.0

    def resolution_scale(self, values):
        """Return the fixed characteristic angular scale."""
        del values
        return 2.0

    def jacobian_mask_points(self, values):
        """Return no Jacobian mask points."""
        del values
        return ()

    def root_recovery_points(self, values):
        """Return no targeted image-recovery points."""
        del values
        return ()

    def axisymmetry_center(self, values):
        """Disable the axisymmetric three-snapshot certification branch."""
        del values
        return None

    def expected_num_images(self, source_x, source_y, **kwargs):
        """Return one fixed certified image count."""
        del source_x, source_y, kwargs
        return 4


class _RecoveryGeometryAdapter(_FakeGeometryAdapter):
    """Expose configurable targeted-recovery points."""

    def __init__(self, recovery_points):
        self.recovery_points = tuple(recovery_points)

    def root_recovery_points(self, values):
        """Return configured targeted image-recovery points."""
        del values
        return self.recovery_points


class _FakeLens:
    """Return deterministic image observables."""

    def magnification(self, image_x, image_y):
        """Return signed values so the node must take absolute magnitudes."""
        del image_x, image_y
        return _FakeTensor([-2.0, 3.0])

    def time_delay(self, image_x, image_y):
        """Return absolute arrival times in an order different from the images."""
        del image_x, image_y
        return _FakeTensor([12.0, 10.0])

    def convergence(self, image_x, image_y):
        """Return deterministic macro convergences."""
        del image_x, image_y
        return _FakeTensor([0.25, 0.5])

    def shear(self, image_x, image_y):
        """Return deterministic Cartesian macro-shear components."""
        del image_x, image_y
        return (_FakeTensor([3.0, 5.0]), _FakeTensor([4.0, 12.0]))


class _ArrayLens:
    """Return configurable observables while recording their evaluation count."""

    def __init__(self, magnifications, delays, convergences=None, shear1=None, shear2=None):
        self.magnifications = np.asarray(magnifications, dtype=float)
        self.delays = np.asarray(delays, dtype=float)
        self.convergences = (
            np.zeros_like(self.magnifications)
            if convergences is None
            else np.asarray(convergences, dtype=float)
        )
        self.shear1 = (
            np.zeros_like(self.magnifications) if shear1 is None else np.asarray(shear1, dtype=float)
        )
        self.shear2 = (
            np.zeros_like(self.magnifications) if shear2 is None else np.asarray(shear2, dtype=float)
        )
        self.magnification_calls = 0
        self.delay_calls = 0
        self.convergence_calls = 0
        self.shear_calls = 0

    def magnification(self, image_x, image_y):
        """Return configured signed magnifications."""
        del image_x, image_y
        self.magnification_calls += 1
        return _FakeTensor(self.magnifications)

    def time_delay(self, image_x, image_y):
        """Return configured absolute delays."""
        del image_x, image_y
        self.delay_calls += 1
        return _FakeTensor(self.delays)

    def convergence(self, image_x, image_y):
        """Return configured macro convergences."""
        del image_x, image_y
        self.convergence_calls += 1
        return _FakeTensor(self.convergences)

    def shear(self, image_x, image_y):
        """Return configured Cartesian macro-shear components."""
        del image_x, image_y
        self.shear_calls += 1
        return (_FakeTensor(self.shear1), _FakeTensor(self.shear2))


def _closed_square(center=(0.0, 0.0), half_width=1.0):
    """Return one counterclockwise producer-closed square."""
    center_x, center_y = center
    return np.array(
        [
            [center_x - half_width, center_y - half_width],
            [center_x + half_width, center_y - half_width],
            [center_x + half_width, center_y + half_width],
            [center_x - half_width, center_y + half_width],
            [center_x - half_width, center_y - half_width],
        ],
        dtype=float,
    )


@pytest.fixture
def fake_caustics_registry(monkeypatch):
    """Replace the optional Caustics import with an explicit constructor."""
    registry = SimpleNamespace(
        SIS=_FakeSIS,
        ExternalShear=_FakeExternalShear,
        MassSheet=_FakeMassSheet,
        SinglePlane=_FakeSinglePlane,
    )
    monkeypatch.setattr(caustics_runtime, "_import_caustics", lambda: registry)
    return registry


@pytest.fixture
def valid_sis_spec(fake_caustics_registry):
    """Return one valid root SIS specification for public-node tests."""
    del fake_caustics_registry
    return caustics_models.CausticsLensSpec(
        "SIS",
        {
            "z_l": 0.5,
            "x0": 0.0,
            "y0": 0.0,
            "Rein": 1.0,
        },
    )


@pytest.fixture
def fixed_cosmology():
    """Return an opaque cosmology whose identity must remain node-owned."""
    return object()


def _source_node(lens, cosmology, **overrides):
    """Build a compact valid source node, with explicit settings for unit tests."""
    settings = {
        "cosmology": cosmology,
        "source_redshift": 1.5,
        "fov": 4.0,
        "pixelscale": 1.0,
        "max_fov_expansions": 2,
        "fov_expansion_factor": 2.0,
        "pseudo_caustic_points": 8,
        "pseudo_caustic_epsilon": 0.1,
        "geometry_tolerance": 0.01,
        "boundary_tolerance": 0.1,
        "max_boundary_refinements": 3,
        "max_attempts": 10,
        "seed": 17,
        "node_label": "source_node",
    }
    settings.update(overrides)
    return caustics_models.CausticsSourcePositionNode(lens, **settings)


def _image_node(lens, cosmology, **overrides):
    """Build a compact valid image node, with bounded recovery by default."""
    settings = {
        "cosmology": cosmology,
        "source_redshift": 1.5,
        "source_x": 0.0,
        "source_y": 0.0,
        "max_images": 4,
        "min_images": 2,
        "expected_num_images": 2,
        "fov": 4.0,
        "fov_multiplier": 1.0,
        "pixelscale": 1.0,
        "epsilon": 0.1,
        "max_depth": 5,
        "max_fov_expansions": 0,
        "fov_expansion_factor": 2.0,
        "max_pixelscale_refinements": 0,
        "pixelscale_refinement_factor": 0.5,
        "node_label": "image_node",
    }
    settings.update(overrides)
    return caustics_models.CausticsLensImageNode(lens, **settings)


def _image_values(*, fov=4.0, expected_num_images=2, source_x=0.0, source_y=0.0):
    """Return the per-realization inputs consumed directly by image solving."""
    return {
        "source_redshift": 1.5,
        "source_x": source_x,
        "source_y": source_y,
        "fov": fov,
        "expected_num_images": expected_num_images,
    }


def _graph_state_for(node, num_samples, *, overrides=None, fixed_outputs=None):
    """Populate a GraphState with one node's constant registered inputs."""
    overrides = {} if overrides is None else overrides
    fixed_outputs = {} if fixed_outputs is None else fixed_outputs
    graph_state = GraphState(num_samples)
    for name in node.arg_names:
        graph_state.set(
            node.node_string,
            name,
            overrides.get(name, node.setters[name].value),
        )
    for name, value in fixed_outputs.items():
        graph_state.set(node.node_string, name, value, fixed=True)
    return graph_state


def _boundary_snapshot(*, center_x, fov, pixelscale, pseudo_caustic_points):
    """Return one small regular closed-boundary certification snapshot."""
    return caustics_source_geometry._BoundaryGeometry(
        caustic_curves=(_closed_square(center=(center_x, 0.0)),),
        pseudo_caustic_curves=(),
        critical_curve_fov=fov,
        pixelscale=pixelscale,
        pseudo_caustic_points=pseudo_caustic_points,
    )


def _patch_image_runtime(monkeypatch, lens, geometry_adapter):
    """Patch realization and tensor conversion immediately below image policy."""
    monkeypatch.setattr(
        caustics_lens_system,
        "_build_lens_system",
        lambda *args, **kwargs: (lens, geometry_adapter, {}),
    )
    monkeypatch.setattr(
        caustics_runtime,
        "_import_caustics_dependencies",
        lambda: (object(), _FakeTorch),
    )


def _patch_source_compute_runtime(monkeypatch, node, *, extra_draws=(), events=None):
    """Patch source geometry, sampling, and clearance below orchestration policy."""
    adapter = _FakeGeometryAdapter()
    previous = _boundary_snapshot(
        center_x=0.0,
        fov=4.0,
        pixelscale=0.5,
        pseudo_caustic_points=16,
    )
    geometry = _boundary_snapshot(
        center_x=0.05,
        fov=5.0,
        pixelscale=0.25,
        pseudo_caustic_points=32,
    )
    region = object()

    def region_for_one_lens(values, *, sample_index):
        del values
        if events is not None:
            events.append(("region", sample_index))
        return (
            adapter,
            {},
            previous,
            geometry,
            0.1,
            2,
            region,
            0.5,
        )

    def sample_position(
        current_region,
        rng,
        *,
        max_attempts,
        lens_identifier,
        geometry_settings,
        excluded_points,
    ):
        del max_attempts, geometry_settings, excluded_points
        assert current_region is region
        sample_index = int(lens_identifier.split(" sample ", 1)[1].split(" ", 1)[0])
        sample_extra_draws = extra_draws[sample_index] if sample_index < len(extra_draws) else 0
        for _ in range(sample_extra_draws):
            rng.random()
        return (
            rng.uniform(-1.0, 1.0),
            rng.uniform(-1.0, 1.0),
            10.0 + sample_index,
            sample_extra_draws + 1,
        )

    monkeypatch.setattr(node, "_region_for_one_lens", region_for_one_lens)
    monkeypatch.setattr(caustics_source_geometry, "_sample_position", sample_position)
    monkeypatch.setattr(
        caustics_source_geometry,
        "_source_boundary_clearance",
        lambda *args, **kwargs: 0.5,
    )


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


def test_lens_spec_snapshots_parameters_and_exposes_read_only_mapping(fake_caustics_registry):
    """Snapshot the caller mapping while retaining dependency identities."""
    del fake_caustics_registry
    redshift_dependency = object()
    center_dependency = object()
    parameters = {
        "z_l": redshift_dependency,
        "x0": center_dependency,
        "y0": -0.25,
        "Rein": 1.5,
    }

    spec = caustics_models.CausticsLensSpec("SIS", parameters)
    parameters["z_l"] = object()
    parameters["x0"] = object()
    parameters["extra"] = "later mutation"

    assert tuple(spec.parameters) == ("z_l", "x0", "y0", "Rein")
    assert spec.parameters["z_l"] is redshift_dependency
    assert spec.parameters["x0"] is center_dependency
    assert "extra" not in spec.parameters
    with pytest.raises(TypeError, match="does not support item assignment"):
        spec.parameters["x0"] = 3.0


@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        (["z_l", "x0"], "parameters must be a mapping"),
        ({1: "not a string"}, "parameters keys must be strings"),
    ],
)
def test_lens_spec_rejects_non_mapping_and_non_string_keys(
    fake_caustics_registry,
    parameters,
    message,
):
    """Reject invalid mapping containers and key types before inspection."""
    del fake_caustics_registry
    with pytest.raises(TypeError, match=message):
        caustics_models.CausticsLensSpec("SIS", parameters)


@pytest.mark.parametrize(
    ("reserved_name", "message"),
    [
        ("cosmology", "cosmology is supplied by the consuming node"),
        ("z_s", "z_s is supplied by the consuming node as source_redshift"),
    ],
)
def test_lens_spec_rejects_reserved_node_owned_parameters(
    fake_caustics_registry,
    reserved_name,
    message,
):
    """Reject constructor fields owned by the consuming graph node."""
    del fake_caustics_registry
    parameters = {
        "z_l": 0.5,
        "x0": 0.0,
        "y0": 0.0,
        "Rein": 1.0,
        reserved_name: object(),
    }

    with pytest.raises(ValueError, match=message):
        caustics_models.CausticsLensSpec("SIS", parameters)


def test_lens_spec_rejects_unknown_model_and_unsupported_parameters(fake_caustics_registry):
    """Reject unknown registry keys and unsupported constructor names."""
    del fake_caustics_registry
    with pytest.raises(ValueError, match="not supported currently"):
        caustics_models.CausticsLensSpec("UnknownLens", {})

    with pytest.raises(ValueError, match="keys not supported") as error:
        caustics_models.CausticsLensSpec(
            "SIS",
            {
                "z_l": 0.5,
                "x0": 0.0,
                "y0": 0.0,
                "Rein": 1.0,
                "unsupported_field": 2.0,
            },
        )
    assert "unsupported_field" in str(error.value)


@pytest.mark.parametrize("required_name", ["x0", "y0", "Rein"])
def test_lens_spec_requires_constructor_parameters(fake_caustics_registry, required_name):
    """Require every explicit physical constructor argument without a default."""
    del fake_caustics_registry
    parameters = {
        "z_l": 0.5,
        "x0": 0.0,
        "y0": 0.0,
        "Rein": 1.0,
    }
    parameters.pop(required_name)

    with pytest.raises(
        ValueError,
        match=rf"{required_name} is required by the chosen SIS lens model",
    ):
        caustics_models.CausticsLensSpec("SIS", parameters)


def test_external_shear_validates_cartesian_and_angular_names(fake_caustics_registry):
    """Accept only the shear names selected by a literal parametrization."""
    del fake_caustics_registry
    cartesian = caustics_models.CausticsLensSpec(
        "ExternalShear",
        {
            "z_l": 0.5,
            "x0": 0.0,
            "y0": 0.0,
            "gamma_1": 0.1,
            "gamma_2": -0.2,
        },
    )
    angular = caustics_models.CausticsLensSpec(
        "ExternalShear",
        {
            "z_l": 0.5,
            "x0": 0.0,
            "y0": 0.0,
            "parametrization": "angular",
            "gamma": 0.2,
            "phi": 0.4,
        },
    )

    assert cartesian.parameters["gamma_1"] == 0.1
    assert cartesian.parameters["gamma_2"] == -0.2
    assert angular.parameters["gamma"] == 0.2
    assert angular.parameters["phi"] == 0.4

    with pytest.raises(ValueError, match="keys not supported") as cartesian_error:
        caustics_models.CausticsLensSpec(
            "ExternalShear",
            {
                "z_l": 0.5,
                "x0": 0.0,
                "y0": 0.0,
                "gamma": 0.2,
                "phi": 0.4,
            },
        )
    assert "gamma" in str(cartesian_error.value)
    assert "phi" in str(cartesian_error.value)

    with pytest.raises(ValueError, match="keys not supported") as angular_error:
        caustics_models.CausticsLensSpec(
            "ExternalShear",
            {
                "z_l": 0.5,
                "x0": 0.0,
                "y0": 0.0,
                "parametrization": "angular",
                "gamma_1": 0.1,
                "gamma_2": -0.2,
            },
        )
    assert "gamma_1" in str(angular_error.value)
    assert "gamma_2" in str(angular_error.value)


def test_external_shear_allows_both_name_sets_for_dynamic_selector(fake_caustics_registry):
    """Retain both possible shear branches for a graph-backed selector."""
    del fake_caustics_registry
    selector = object()
    gamma_1 = object()
    gamma_2 = object()
    gamma = object()
    phi = object()

    spec = caustics_models.CausticsLensSpec(
        "ExternalShear",
        {
            "z_l": 0.5,
            "x0": 0.0,
            "y0": 0.0,
            "parametrization": selector,
            "gamma_1": gamma_1,
            "gamma_2": gamma_2,
            "gamma": gamma,
            "phi": phi,
        },
    )

    assert spec.parameters["parametrization"] is selector
    assert spec.parameters["gamma_1"] is gamma_1
    assert spec.parameters["gamma_2"] is gamma_2
    assert spec.parameters["gamma"] is gamma
    assert spec.parameters["phi"] is phi


def test_single_plane_snapshots_lenses_as_tuple(fake_caustics_registry):
    """Snapshot a composite's child sequence while retaining child identities."""
    del fake_caustics_registry
    first = caustics_models.CausticsLensSpec(
        "SIS",
        {"x0": 0.0, "y0": 0.0, "Rein": 1.0},
    )
    second = caustics_models.CausticsLensSpec(
        "MassSheet",
        {"x0": 0.0, "y0": 0.0, "kappa": 0.1},
    )
    lenses = [first, second]

    plane = caustics_models.CausticsLensSpec(
        "SinglePlane",
        {"z_l": 0.5, "lenses": lenses},
    )
    lenses.clear()

    assert isinstance(plane.parameters["lenses"], tuple)
    assert plane.parameters["lenses"] == (first, second)
    assert plane.parameters["lenses"][0] is first
    assert plane.parameters["lenses"][1] is second


def test_single_plane_rejects_non_specs(fake_caustics_registry):
    """Reject both non-iterable lens containers and invalid child objects."""
    del fake_caustics_registry
    with pytest.raises(TypeError, match="not iterable"):
        caustics_models.CausticsLensSpec(
            "SinglePlane",
            {"z_l": 0.5, "lenses": object()},
        )

    with pytest.raises(TypeError, match="only CausticsLensSpec objects"):
        caustics_models.CausticsLensSpec(
            "SinglePlane",
            {"z_l": 0.5, "lenses": [object()]},
        )


def test_source_node_rejects_non_spec_lens(fixed_cosmology):
    """Reject invalid public lens objects before registering graph inputs."""
    with pytest.raises(TypeError, match="lens must be a CausticsLensSpec"):
        _source_node(object(), fixed_cosmology)


@pytest.mark.parametrize(
    ("overrides", "error_type", "message"),
    [
        ({"fov": object()}, TypeError, "fov must be a scalar number"),
        ({"pixelscale": object()}, TypeError, "pixelscale must be a scalar number"),
        (
            {"pseudo_caustic_epsilon": object()},
            TypeError,
            "pseudo_caustic_epsilon must be a scalar number",
        ),
        (
            {"geometry_tolerance": object()},
            TypeError,
            "geometry_tolerance must be a scalar number",
        ),
        (
            {"boundary_tolerance": object()},
            TypeError,
            "boundary_tolerance must be a scalar number",
        ),
        (
            {"fov_expansion_factor": object()},
            TypeError,
            "fov_expansion_factor must be a scalar number",
        ),
        (
            {"pixelscale_fraction": np.array([0.5])},
            TypeError,
            "pixelscale_fraction must be None or a scalar value convertible to float",
        ),
        ({"fov": 0.0}, ValueError, "fov must be finite and positive"),
        ({"pixelscale": np.nan}, ValueError, "pixelscale must be finite and positive"),
        (
            {"pseudo_caustic_epsilon": 0.0},
            ValueError,
            "pseudo_caustic_epsilon must be finite and positive",
        ),
        (
            {"geometry_tolerance": np.inf},
            ValueError,
            "geometry_tolerance must be finite and positive",
        ),
        (
            {"boundary_tolerance": 0.0},
            ValueError,
            "boundary_tolerance must be finite and positive",
        ),
        (
            {"pixelscale_fraction": 0.0},
            ValueError,
            "pixelscale_fraction must be finite and positive",
        ),
        (
            {"fov_expansion_factor": 1.0},
            ValueError,
            "fov_expansion_factor must be finite and greater than one",
        ),
        (
            {"max_fov_expansions": -1},
            ValueError,
            "max_fov_expansions must be a non-negative integer",
        ),
        (
            {"max_fov_expansions": np.int64(0)},
            ValueError,
            "max_fov_expansions must be a non-negative integer",
        ),
        (
            {"pseudo_caustic_points": 2},
            ValueError,
            "pseudo_caustic_points must be an integer of at least three",
        ),
        (
            {"pseudo_caustic_points": np.int64(3)},
            ValueError,
            "pseudo_caustic_points must be an integer of at least three",
        ),
        (
            {"max_boundary_refinements": 0},
            ValueError,
            "max_boundary_refinements must be a positive integer",
        ),
        (
            {"max_boundary_refinements": np.int64(1)},
            ValueError,
            "max_boundary_refinements must be a positive integer",
        ),
        (
            {"max_attempts": 0},
            ValueError,
            "max_attempts must be a positive integer",
        ),
        (
            {"max_attempts": np.int64(1)},
            ValueError,
            "max_attempts must be a positive integer",
        ),
        (
            {"fov": 1.0, "pixelscale": 1.0},
            ValueError,
            "pixelscale must be smaller than fov",
        ),
        (
            {"pixelscale": 0.1, "geometry_tolerance": 0.1},
            ValueError,
            "geometry_tolerance must be smaller than pixelscale",
        ),
        (
            {"geometry_tolerance": 0.02, "boundary_tolerance": 0.01},
            ValueError,
            "boundary_tolerance must be at least geometry_tolerance",
        ),
    ],
)
def test_source_node_validates_every_constructor_setting(
    valid_sis_spec,
    fixed_cosmology,
    overrides,
    error_type,
    message,
):
    """Assign conversion, domain, integer, and relation failures correctly."""
    with pytest.raises(error_type, match=message):
        _source_node(valid_sis_spec, fixed_cosmology, **overrides)


def test_source_node_accepts_documented_constructor_boundaries(valid_sis_spec, fixed_cosmology):
    """Accept inclusive integer and tolerance boundaries at their minimums."""
    node = _source_node(
        valid_sis_spec,
        fixed_cosmology,
        fov=0.2,
        pixelscale=0.1,
        max_fov_expansions=0,
        pseudo_caustic_points=3,
        geometry_tolerance=0.01,
        boundary_tolerance=0.01,
        max_boundary_refinements=1,
        max_attempts=1,
    )

    assert node.max_fov_expansions == 0
    assert node.pseudo_caustic_points == 3
    assert node.boundary_tolerance == node.geometry_tolerance == 0.01
    assert node.max_boundary_refinements == 1
    assert node.max_attempts == 1


def test_image_node_rejects_non_spec_lens(fixed_cosmology):
    """Reject invalid public lens objects before registering graph inputs."""
    with pytest.raises(TypeError, match="lens must be a CausticsLensSpec"):
        _image_node(object(), fixed_cosmology)


@pytest.mark.parametrize(
    ("overrides", "error_type", "message"),
    [
        ({"max_images": 1}, ValueError, "max_images must be an integer greater than one"),
        ({"max_images": 2.0}, ValueError, "max_images must be an integer greater than one"),
        ({"min_images": 0}, ValueError, "min_images must be between one and max_images"),
        ({"min_images": 5}, ValueError, "min_images must be between one and max_images"),
        (
            {"pixelscale_fraction": np.array([0.5])},
            TypeError,
            "pixelscale_fraction must be None or a scalar value convertible to float",
        ),
        (
            {"epsilon_fraction": object()},
            TypeError,
            "epsilon_fraction must be None or a scalar value convertible to float",
        ),
        (
            {"pixelscale_fraction": 0.0},
            ValueError,
            "pixelscale_fraction must be finite and positive",
        ),
        (
            {"epsilon_fraction": np.inf},
            ValueError,
            "epsilon_fraction must be finite and positive",
        ),
        ({"fov_multiplier": object()}, TypeError, "fov_multiplier must be a scalar number"),
        ({"pixelscale": object()}, TypeError, "pixelscale must be a scalar number"),
        ({"epsilon": object()}, TypeError, "epsilon must be a scalar number"),
        (
            {"fov_expansion_factor": object()},
            TypeError,
            "fov_expansion_factor must be a scalar number",
        ),
        (
            {"pixelscale_refinement_factor": object()},
            TypeError,
            "pixelscale_refinement_factor must be a scalar number",
        ),
        ({"fov_multiplier": 0.0}, ValueError, "fov_multiplier must be finite and positive"),
        ({"pixelscale": np.nan}, ValueError, "pixelscale must be finite and positive"),
        ({"epsilon": 0.0}, ValueError, "epsilon must be finite and positive"),
        (
            {"fov_expansion_factor": 1.0},
            ValueError,
            "fov_expansion_factor must be finite and greater than one",
        ),
        (
            {"pixelscale_refinement_factor": 0.0},
            ValueError,
            "pixelscale_refinement_factor must be finite and strictly between zero and one",
        ),
        (
            {"pixelscale_refinement_factor": 1.0},
            ValueError,
            "pixelscale_refinement_factor must be finite and strictly between zero and one",
        ),
        ({"max_depth": 0}, ValueError, "max_depth must be a positive integer"),
        ({"max_depth": 1.0}, ValueError, "max_depth must be a positive integer"),
        (
            {"max_fov_expansions": -1},
            ValueError,
            "max_fov_expansions must be a non-negative integer",
        ),
        (
            {"max_fov_expansions": 1.0},
            ValueError,
            "max_fov_expansions must be a non-negative integer",
        ),
        (
            {"max_pixelscale_refinements": -1},
            ValueError,
            "max_pixelscale_refinements must be a non-negative integer",
        ),
        (
            {"max_pixelscale_refinements": 1.0},
            ValueError,
            "max_pixelscale_refinements must be a non-negative integer",
        ),
    ],
)
def test_image_node_validates_every_constructor_setting(
    valid_sis_spec,
    fixed_cosmology,
    overrides,
    error_type,
    message,
):
    """Assign conversion, domain, count, depth, and recovery failures correctly."""
    with pytest.raises(error_type, match=message):
        _image_node(valid_sis_spec, fixed_cosmology, **overrides)


def test_image_node_accepts_documented_integer_boundaries(valid_sis_spec, fixed_cosmology):
    """Accept NumPy integers and inclusive lower bounds documented by the node."""
    node = _image_node(
        valid_sis_spec,
        fixed_cosmology,
        max_images=np.int64(2),
        min_images=np.int64(1),
        expected_num_images=None,
        max_depth=np.int64(1),
        max_fov_expansions=np.int64(0),
        max_pixelscale_refinements=np.int64(0),
    )

    assert node.max_images == 2
    assert node.min_images == 1
    assert node.max_depth == 1
    assert node.max_fov_expansions == 0
    assert node.max_pixelscale_refinements == 0


@pytest.mark.parametrize(
    ("factory", "outputs"),
    [
        (_source_node, _SOURCE_OUTPUTS),
        (_image_node, _IMAGE_OUTPUTS),
    ],
)
def test_public_nodes_keep_cosmology_fixed_and_register_exact_output_order(
    valid_sis_spec,
    fixed_cosmology,
    factory,
    outputs,
):
    """Keep cosmology off-graph and preserve the public FunctionNode order."""
    node = factory(valid_sis_spec, fixed_cosmology)

    assert node.cosmology is fixed_cosmology
    assert "cosmology" not in node.list_params()
    assert "cosmology" not in node.setters
    assert tuple(node.outputs) == outputs
    assert tuple(node.list_params()[-len(outputs) :]) == outputs


def test_source_node_resolves_absolute_and_relative_pixelscales(valid_sis_spec, fixed_cosmology):
    """Use the configured scale directly or the smaller relative lens scale."""
    adapter = _FakeGeometryAdapter()
    absolute = _source_node(
        valid_sis_spec,
        fixed_cosmology,
        pixelscale=0.75,
        pixelscale_fraction=None,
    )
    relative = _source_node(
        valid_sis_spec,
        fixed_cosmology,
        pixelscale=0.75,
        pixelscale_fraction=0.25,
    )
    capped = _source_node(
        valid_sis_spec,
        fixed_cosmology,
        pixelscale=0.75,
        pixelscale_fraction=1.0,
    )

    assert absolute._realized_pixelscale_for_one_lens(adapter, {}) == 0.75
    assert relative._realized_pixelscale_for_one_lens(adapter, {}) == 0.5
    assert capped._realized_pixelscale_for_one_lens(adapter, {}) == 0.75


def test_source_node_uses_configured_or_adapter_derived_fov(valid_sis_spec, fixed_cosmology):
    """Give an explicit FOV precedence over the adapter-derived extent."""
    adapter = _FakeGeometryAdapter()
    configured = _source_node(valid_sis_spec, fixed_cosmology, fov=6.0)
    derived = _source_node(valid_sis_spec, fixed_cosmology, fov=None)

    assert configured._initial_fov_for_one_lens(adapter, {}, pixelscale=1.0) == 6.0
    assert derived._initial_fov_for_one_lens(adapter, {}, pixelscale=1.0) == 4.0


@pytest.mark.parametrize(
    ("node_settings", "pixelscale"),
    [
        ({"fov": None, "pixelscale": 4.0}, 4.0),
        ({"fov": 1.0, "pixelscale": 2.0, "pixelscale_fraction": 0.5}, 1.0),
    ],
)
def test_source_node_rejects_realized_fov_not_larger_than_scale(
    valid_sis_spec,
    fixed_cosmology,
    node_settings,
    pixelscale,
):
    """Apply the dynamic FOV/scale relation after adapter realization."""
    node = _source_node(valid_sis_spec, fixed_cosmology, **node_settings)

    with pytest.raises(ValueError, match=rf"Initial fov .* larger than pixelscale={pixelscale}"):
        node._initial_fov_for_one_lens(
            _FakeGeometryAdapter(),
            {},
            pixelscale=pixelscale,
        )


def test_source_caustic_search_expands_fov_with_fixed_pixelscale(
    monkeypatch,
    valid_sis_spec,
    fixed_cosmology,
):
    """Multiply only the FOV until critical-curve extraction succeeds."""
    node = _source_node(
        valid_sis_spec,
        fixed_cosmology,
        fov=2.0,
        pixelscale=0.25,
        max_fov_expansions=2,
        fov_expansion_factor=2.0,
    )
    calls = []
    curve = _closed_square(half_width=0.5)

    def find_all(lens, **kwargs):
        calls.append((lens, kwargs))
        if len(calls) < 3:
            raise caustics_source_geometry._CausticFOVError(f"incomplete {len(calls)}")
        return [curve]

    monkeypatch.setattr(caustics_source_geometry, "_find_all_caustics", find_all)
    lens = object()
    adapter = _FakeGeometryAdapter()

    curves, successful_fov = node._find_all_caustics_for_one_lens(
        lens,
        adapter,
        {},
        sample_index=3,
        pixelscale=0.25,
    )

    assert curves == (curve,)
    assert successful_fov == 8.0
    assert [call[1]["fov"] for call in calls] == [2.0, 4.0, 8.0]
    assert [call[1]["pixelscale"] for call in calls] == [0.25, 0.25, 0.25]
    assert [call[1]["center"] for call in calls] == [(0.0, 0.0)] * 3
    assert [call[1]["jacobian_mask_points"] for call in calls] == [()] * 3


def test_source_caustic_search_reports_bounded_fov_exhaustion(
    monkeypatch,
    valid_sis_spec,
    fixed_cosmology,
):
    """Report the initial/final schedule and retain the last retry as cause."""
    node = _source_node(
        valid_sis_spec,
        fixed_cosmology,
        fov=2.0,
        pixelscale=0.25,
        max_fov_expansions=1,
        fov_expansion_factor=3.0,
    )
    calls = []
    last_error = caustics_source_geometry._CausticFOVError("still incomplete")

    def always_incomplete(lens, **kwargs):
        del lens
        calls.append(kwargs["fov"])
        raise last_error

    monkeypatch.setattr(caustics_source_geometry, "_find_all_caustics", always_incomplete)

    with pytest.raises(RuntimeError) as error:
        node._find_all_caustics_for_one_lens(
            object(),
            _FakeGeometryAdapter(),
            {},
            sample_index=7,
            pixelscale=0.25,
        )

    assert calls == [2.0, 6.0]
    assert error.value.__cause__ is last_error
    message = str(error.value)
    assert "lens model 'SIS' sample 7 at node 'source_node'" in message
    assert "initial fov=2.0 arcsec, final fov=6.0 arcsec" in message
    assert "pixelscale=0.25 arcsec, max_fov_expansions=1" in message


def test_source_boundary_certification_refines_until_displacement_converges(
    monkeypatch,
    valid_sis_spec,
    fixed_cosmology,
):
    """Halve grid scale, double boundary points, and warm-start each FOV."""
    pytest.importorskip("shapely")
    node = _source_node(
        valid_sis_spec,
        fixed_cosmology,
        pixelscale=1.0,
        boundary_tolerance=0.2,
        max_boundary_refinements=3,
    )
    snapshots = [
        _boundary_snapshot(center_x=0.0, fov=4.0, pixelscale=1.0, pseudo_caustic_points=8),
        _boundary_snapshot(center_x=0.5, fov=6.0, pixelscale=0.5, pseudo_caustic_points=16),
        _boundary_snapshot(center_x=0.6, fov=8.0, pixelscale=0.25, pseudo_caustic_points=32),
    ]
    calls = []

    def boundary_snapshot(lens, adapter, values, **kwargs):
        del lens, adapter, values
        calls.append(kwargs)
        return snapshots[len(calls) - 1]

    monkeypatch.setattr(node, "_boundary_geometry_for_one_lens", boundary_snapshot)

    previous, current, uncertainty, refinements = node._certified_boundary_geometry_for_one_lens(
        object(),
        _FakeGeometryAdapter(),
        {},
        sample_index=0,
        pixelscale=1.0,
    )

    assert [call["pixelscale"] for call in calls] == [1.0, 0.5, 0.25]
    assert [call["pseudo_caustic_points"] for call in calls] == [8, 16, 32]
    assert [call["initial_fov"] for call in calls] == [None, 4.0, 6.0]
    assert previous.critical_curve_fov == 6.0
    assert current.critical_curve_fov == 8.0
    assert uncertainty == pytest.approx(0.1)
    assert refinements == 2


def test_source_boundary_certification_reports_nonconverging_snapshots(
    monkeypatch,
    valid_sis_spec,
    fixed_cosmology,
):
    """Report the final adjacent snapshot settings after bounded exhaustion."""
    pytest.importorskip("shapely")
    node = _source_node(
        valid_sis_spec,
        fixed_cosmology,
        pixelscale=1.0,
        boundary_tolerance=0.1,
        max_boundary_refinements=2,
    )
    snapshots = [
        _boundary_snapshot(center_x=0.0, fov=4.0, pixelscale=1.0, pseudo_caustic_points=8),
        _boundary_snapshot(center_x=1.0, fov=5.0, pixelscale=0.5, pseudo_caustic_points=16),
        _boundary_snapshot(center_x=2.0, fov=6.0, pixelscale=0.25, pseudo_caustic_points=32),
    ]
    calls = []

    def boundary_snapshot(lens, adapter, values, **kwargs):
        del lens, adapter, values
        calls.append(kwargs)
        return snapshots[len(calls) - 1]

    monkeypatch.setattr(node, "_boundary_geometry_for_one_lens", boundary_snapshot)

    with pytest.raises(RuntimeError) as error:
        node._certified_boundary_geometry_for_one_lens(
            object(),
            _FakeGeometryAdapter(),
            {},
            sample_index=4,
            pixelscale=1.0,
        )

    assert len(calls) == 3
    message = str(error.value)
    assert "lens model 'SIS' sample 4 at node 'source_node'" in message
    assert "last displacement=1.0 arcsec, topology_stable=True" in message
    assert "max_boundary_refinements=2" in message
    assert "previous_pixelscale=0.5 arcsec" in message
    assert "current_pixelscale=0.25 arcsec" in message
    assert "previous_critical_curve_fov=5.0 arcsec" in message
    assert "current_critical_curve_fov=6.0 arcsec" in message


def test_axisymmetric_boundary_certification_partitions_stable_point_and_converges(
    monkeypatch,
    valid_sis_spec,
    fixed_cosmology,
):
    """Use three snapshots to separate a stable contraction from regular area."""
    pytest.importorskip("shapely")
    node = _source_node(
        valid_sis_spec,
        fixed_cosmology,
        pixelscale=1.0,
        boundary_tolerance=0.1,
        max_boundary_refinements=3,
    )

    class AxisymmetricAdapter(_FakeGeometryAdapter):
        def axisymmetry_center(self, values):
            assert values is adapter_values
            return (100.0, -100.0)

    point_curves = (
        _closed_square(center=(0.0, 0.0), half_width=4.0),
        _closed_square(center=(0.04, 0.0), half_width=2.0),
        _closed_square(center=(0.08, 0.0), half_width=1.0),
    )
    regular_curves = (
        _closed_square(center=(10.0, 0.0), half_width=1.0),
        _closed_square(center=(10.02, 0.0), half_width=1.0),
        _closed_square(center=(10.05, 0.0), half_width=1.0),
    )
    snapshots = [
        caustics_source_geometry._BoundaryGeometry(
            caustic_curves=(point_curve, regular_curve),
            pseudo_caustic_curves=(),
            critical_curve_fov=4.0 + index,
            pixelscale=1.0 / 2**index,
            pseudo_caustic_points=8 * 2**index,
        )
        for index, (point_curve, regular_curve) in enumerate(zip(point_curves, regular_curves, strict=True))
    ]
    calls = []
    lens = object()
    adapter = AxisymmetricAdapter()
    adapter_values = object()

    def boundary_snapshot(passed_lens, passed_adapter, passed_values, **kwargs):
        assert passed_lens is lens
        assert passed_adapter is adapter
        assert passed_values is adapter_values
        calls.append(kwargs)
        return snapshots[len(calls) - 1]

    monkeypatch.setattr(node, "_boundary_geometry_for_one_lens", boundary_snapshot)

    previous, current, uncertainty, refinements = node._certified_boundary_geometry_for_one_lens(
        lens,
        adapter,
        adapter_values,
        sample_index=2,
        pixelscale=1.0,
    )

    assert [call["pixelscale"] for call in calls] == [1.0, 0.5, 0.25]
    assert [call["pseudo_caustic_points"] for call in calls] == [8, 16, 32]
    assert [call["initial_fov"] for call in calls] == [None, 4.0, 5.0]
    assert previous.caustic_curves == (regular_curves[1],)
    assert current.caustic_curves == (regular_curves[2],)
    np.testing.assert_allclose(previous.point_caustics, [[0.04, 0.0]])
    np.testing.assert_allclose(current.point_caustics, [[0.08, 0.0]])
    assert uncertainty == pytest.approx(0.03)
    assert refinements == 2


def test_axisymmetric_boundary_certification_exhausts_on_partition_count_change(
    monkeypatch,
    valid_sis_spec,
    fixed_cosmology,
):
    """Treat a three-snapshot curve-count mismatch as unstable until exhaustion."""
    node = _source_node(
        valid_sis_spec,
        fixed_cosmology,
        pixelscale=1.0,
        boundary_tolerance=0.1,
        max_boundary_refinements=2,
    )

    class AxisymmetricAdapter(_FakeGeometryAdapter):
        def axisymmetry_center(self, values):
            del values
            return (0.0, 0.0)

    snapshots = [
        caustics_source_geometry._BoundaryGeometry(
            caustic_curves=curves,
            pseudo_caustic_curves=(),
            critical_curve_fov=4.0 + index,
            pixelscale=1.0 / 2**index,
            pseudo_caustic_points=8 * 2**index,
        )
        for index, curves in enumerate(
            (
                (_closed_square(),),
                (_closed_square(), _closed_square(center=(10.0, 0.0))),
                (_closed_square(),),
            )
        )
    ]
    calls = []

    def boundary_snapshot(*args, **kwargs):
        del args
        calls.append(kwargs)
        return snapshots[len(calls) - 1]

    monkeypatch.setattr(node, "_boundary_geometry_for_one_lens", boundary_snapshot)

    with pytest.raises(RuntimeError) as error:
        node._certified_boundary_geometry_for_one_lens(
            object(),
            AxisymmetricAdapter(),
            {},
            sample_index=6,
            pixelscale=1.0,
        )

    assert len(calls) == 3
    message = str(error.value)
    assert "lens model 'SIS' sample 6 at node 'source_node'" in message
    assert "last displacement=inf arcsec, topology_stable=False" in message
    assert "previous_pixelscale=0.5 arcsec" in message
    assert "current_pixelscale=0.25 arcsec" in message


def test_source_compute_draws_all_subseeds_before_sample_work(
    monkeypatch,
    valid_sis_spec,
    fixed_cosmology,
):
    """Partition caller randomness once before beginning any realization."""
    node = _source_node(valid_sis_spec, fixed_cosmology)
    events = []
    _patch_source_compute_runtime(monkeypatch, node, events=events)

    class RecordingRNG:
        """Record the one parent sub-seed draw."""

        def integers(self, low, high, *, size, dtype):
            events.append(("subseeds", low, high, size, dtype))
            return np.array([11, 22, 33], dtype=np.uint64)

    node.compute(_graph_state_for(node, 3), rng_info=RecordingRNG())

    assert events == [
        ("subseeds", 0, 2**63, 3, np.uint64),
        ("region", 0),
        ("region", 1),
        ("region", 2),
    ]


def test_source_compute_caller_rng_takes_precedence_and_reproduces(
    monkeypatch,
    valid_sis_spec,
    fixed_cosmology,
):
    """Equal caller seeds dominate different node fallback seeds."""
    first = _source_node(valid_sis_spec, fixed_cosmology, seed=1, node_label="source_first")
    _patch_source_compute_runtime(monkeypatch, first)
    first_results = first.compute(
        _graph_state_for(first, 3),
        rng_info=np.random.default_rng(1234),
    )
    assert first._rng.random() == np.random.default_rng(1).random()

    second = _source_node(valid_sis_spec, fixed_cosmology, seed=999, node_label="source_second")
    _patch_source_compute_runtime(monkeypatch, second)
    second_results = second.compute(
        _graph_state_for(second, 3),
        rng_info=np.random.default_rng(1234),
    )

    for first_result, second_result in zip(first_results, second_results, strict=True):
        np.testing.assert_array_equal(first_result, second_result)


def test_source_compute_isolates_later_samples_from_first_rejection_count(
    monkeypatch,
    valid_sis_spec,
    fixed_cosmology,
):
    """Use sample-local RNG streams so extra first-sample draws do not cascade."""
    node = _source_node(valid_sis_spec, fixed_cosmology)
    _patch_source_compute_runtime(monkeypatch, node, extra_draws=(0, 0, 0))
    baseline = node.compute(
        _graph_state_for(node, 3),
        rng_info=np.random.default_rng(77),
    )

    _patch_source_compute_runtime(monkeypatch, node, extra_draws=(7, 0, 0))
    perturbed = node.compute(
        _graph_state_for(node, 3),
        rng_info=np.random.default_rng(77),
    )

    assert baseline[0][0] != perturbed[0][0]
    assert baseline[1][0] != perturbed[1][0]
    assert baseline[3][0] == 1
    assert perturbed[3][0] == 8
    for baseline_result, perturbed_result in zip(baseline, perturbed, strict=True):
        np.testing.assert_array_equal(baseline_result[1:], perturbed_result[1:])


def test_source_compute_retry_keeps_later_sample_rng_isolated(
    monkeypatch,
    valid_sis_spec,
    fixed_cosmology,
):
    """Keep later outputs fixed when a narrow first-sample draw is retried."""
    node = _source_node(valid_sis_spec, fixed_cosmology)
    _patch_source_compute_runtime(monkeypatch, node)
    baseline_clearances = iter([0.5, 0.5])
    monkeypatch.setattr(
        caustics_source_geometry,
        "_source_boundary_clearance",
        lambda *args, **kwargs: next(baseline_clearances),
    )
    baseline = node.compute(
        _graph_state_for(node, 2),
        rng_info=np.random.default_rng(77),
    )

    retried_clearances = iter([0.1, 0.5, 0.5])
    monkeypatch.setattr(
        caustics_source_geometry,
        "_source_boundary_clearance",
        lambda *args, **kwargs: next(retried_clearances),
    )
    retried = node.compute(
        _graph_state_for(node, 2),
        rng_info=np.random.default_rng(77),
    )

    assert baseline[0][0] != retried[0][0]
    assert baseline[3][0] == 1
    assert retried[3][0] == 2
    for baseline_result, retried_result in zip(baseline, retried, strict=True):
        np.testing.assert_array_equal(baseline_result[1:], retried_result[1:])


@pytest.mark.parametrize("num_samples", [1, 2])
def test_source_compute_shapes_output_order_persistence_and_fixed_semantics(
    monkeypatch,
    valid_sis_spec,
    fixed_cosmology,
    num_samples,
):
    """Pack all nine outputs while preserving an already fixed GraphState value."""
    node = _source_node(valid_sis_spec, fixed_cosmology)
    _patch_source_compute_runtime(monkeypatch, node)
    sample_results = iter(
        [
            (1.25, -0.75, 10.0, 3),
            (2.5, -1.5, 20.0, 4),
        ]
    )

    def deterministic_sample(*args, **kwargs):
        del args, kwargs
        return next(sample_results)

    monkeypatch.setattr(caustics_source_geometry, "_sample_position", deterministic_sample)
    fixed_outputs = {"source_x": 99.0} if num_samples == 1 else None
    graph_state = _graph_state_for(
        node,
        num_samples,
        fixed_outputs=fixed_outputs,
    )

    results = node.compute(graph_state, rng_info=np.random.default_rng(123))
    saved = graph_state[node.node_string]

    assert tuple(name for name in saved if name in _SOURCE_OUTPUTS) == _SOURCE_OUTPUTS
    assert "cosmology" not in saved
    if num_samples == 1:
        assert results[0] == 1.25
        assert results[1] == -0.75
        assert results[2] == 10.0
        assert results[3] == 3
        assert results[4] == 4
        assert results[5] == 5.0
        assert results[6] == 0.1
        assert results[7] == 0.5
        assert results[8] == 2
        assert saved["source_x"] == 99.0
        assert saved["source_y"] == -0.75
        assert saved["strong_lensing_area"] == 10.0
        assert saved["sampling_attempts"] == 3
        assert saved["expected_num_images"] == 4
        assert saved["critical_curve_fov"] == 5.0
        assert saved["boundary_uncertainty"] == 0.1
        assert saved["source_boundary_clearance"] == 0.5
        assert saved["boundary_refinements"] == 2
        assert all(np.isscalar(result) for result in results)
    else:
        np.testing.assert_array_equal(results[0], [1.25, 2.5])
        np.testing.assert_array_equal(results[1], [-0.75, -1.5])
        np.testing.assert_array_equal(results[2], [10.0, 20.0])
        np.testing.assert_array_equal(results[3], [3, 4])
        np.testing.assert_array_equal(results[4], [4, 4])
        np.testing.assert_array_equal(results[5], [5.0, 5.0])
        np.testing.assert_array_equal(results[6], [0.1, 0.1])
        np.testing.assert_array_equal(results[7], [0.5, 0.5])
        np.testing.assert_array_equal(results[8], [2, 2])
        np.testing.assert_array_equal(saved["source_x"], [1.25, 2.5])
        np.testing.assert_array_equal(saved["source_y"], [-0.75, -1.5])
        np.testing.assert_array_equal(saved["strong_lensing_area"], [10.0, 20.0])
        np.testing.assert_array_equal(saved["sampling_attempts"], [3, 4])
        np.testing.assert_array_equal(saved["expected_num_images"], [4, 4])
        np.testing.assert_array_equal(saved["critical_curve_fov"], [5.0, 5.0])
        np.testing.assert_array_equal(saved["boundary_uncertainty"], [0.1, 0.1])
        np.testing.assert_array_equal(saved["source_boundary_clearance"], [0.5, 0.5])
        np.testing.assert_array_equal(saved["boundary_refinements"], [2, 2])
        assert all(result.shape == (2,) for result in results)


def test_source_compute_redraws_narrow_boundary_candidate_with_remaining_budget(
    monkeypatch,
    valid_sis_spec,
    fixed_cosmology,
):
    """Retry equality-clearance candidates within one cumulative draw budget."""
    node = _source_node(valid_sis_spec, fixed_cosmology)
    _patch_source_compute_runtime(monkeypatch, node)
    candidates = iter(
        [
            (0.0, 0.0, 10.0, 2),
            (1.25, -0.75, 10.0, 3),
        ]
    )
    budgets = []

    def sample_position(*args, max_attempts, **kwargs):
        del args, kwargs
        budgets.append(max_attempts)
        return next(candidates)

    clearances = iter([0.1, 0.5])
    monkeypatch.setattr(caustics_source_geometry, "_sample_position", sample_position)
    monkeypatch.setattr(
        caustics_source_geometry,
        "_source_boundary_clearance",
        lambda *args, **kwargs: next(clearances),
    )

    results = node.compute(
        _graph_state_for(node, 1),
        rng_info=np.random.default_rng(5),
    )

    assert results[0:5] == [1.25, -0.75, 10.0, 5, 4]
    assert results[7] == 0.5
    assert budgets == [10, 8]


def test_source_compute_exhausts_narrow_boundary_retry_budget(
    monkeypatch,
    valid_sis_spec,
    fixed_cosmology,
):
    """Report cumulative exhaustion after every candidate is too narrow."""
    node = _source_node(valid_sis_spec, fixed_cosmology, max_attempts=3)
    _patch_source_compute_runtime(monkeypatch, node)
    budgets = []

    def sample_position(*args, max_attempts, **kwargs):
        del args, kwargs
        budgets.append(max_attempts)
        return (0.0, 0.0, 10.0, 1)

    monkeypatch.setattr(caustics_source_geometry, "_sample_position", sample_position)
    monkeypatch.setattr(
        caustics_source_geometry,
        "_source_boundary_clearance",
        lambda *args, **kwargs: 0.1,
    )
    graph_state = _graph_state_for(node, 1)

    with pytest.raises(RuntimeError) as error:
        node.compute(graph_state, rng_info=np.random.default_rng(5))

    assert budgets == [3, 2, 1]
    assert "after 3 attempts" in str(error.value)
    assert "narrow_boundary_rejections=3" in str(error.value)
    assert "last_source_boundary_clearance=0.1 arcsec" in str(error.value)
    assert "boundary_uncertainty=0.1 arcsec" in str(error.value)
    assert not set(_SOURCE_OUTPUTS).intersection(graph_state[node.node_string])


def test_source_compute_wraps_sampling_exhaustion_after_narrow_boundary_rejection(
    monkeypatch,
    valid_sis_spec,
    fixed_cosmology,
):
    """Preserve proposal exhaustion as the cause of total-budget exhaustion."""
    node = _source_node(valid_sis_spec, fixed_cosmology, max_attempts=3)
    _patch_source_compute_runtime(monkeypatch, node)
    inner_error = RuntimeError("unique proposal exhaustion")
    budgets = []

    def sample_position(*args, max_attempts, **kwargs):
        del args, kwargs
        budgets.append(max_attempts)
        if len(budgets) == 1:
            return (0.0, 0.0, 10.0, 1)
        raise inner_error

    monkeypatch.setattr(caustics_source_geometry, "_sample_position", sample_position)
    monkeypatch.setattr(
        caustics_source_geometry,
        "_source_boundary_clearance",
        lambda *args, **kwargs: 0.1,
    )

    with pytest.raises(RuntimeError) as error:
        node.compute(_graph_state_for(node, 1), rng_info=np.random.default_rng(5))

    assert budgets == [3, 2]
    assert "after 3 attempts" in str(error.value)
    assert "narrow_boundary_rejections=1" in str(error.value)
    assert "last_source_boundary_clearance=0.1 arcsec" in str(error.value)
    assert "boundary_uncertainty=0.1 arcsec" in str(error.value)
    assert error.value.__cause__ is inner_error


def test_source_compute_count_mismatch_remains_nonretryable(
    monkeypatch,
    valid_sis_spec,
    fixed_cosmology,
):
    """Raise immediately when penultimate and final image counts disagree."""
    node = _source_node(valid_sis_spec, fixed_cosmology)
    previous = _boundary_snapshot(
        center_x=0.0,
        fov=4.0,
        pixelscale=0.5,
        pseudo_caustic_points=16,
    )
    geometry = _boundary_snapshot(
        center_x=0.05,
        fov=5.0,
        pixelscale=0.25,
        pseudo_caustic_points=32,
    )

    class CountAdapter(_FakeGeometryAdapter):
        def __init__(self):
            self.counts = iter((3, 4))

        def expected_num_images(self, source_x, source_y, **kwargs):
            del source_x, source_y, kwargs
            return next(self.counts)

    monkeypatch.setattr(
        node,
        "_region_for_one_lens",
        lambda values, sample_index: (
            CountAdapter(),
            {},
            previous,
            geometry,
            0.1,
            2,
            object(),
            0.5,
        ),
    )

    sampler_calls = 0

    def sample_position(*args, **kwargs):
        nonlocal sampler_calls
        del args, kwargs
        sampler_calls += 1
        return (0.0, 0.0, 1.0, 1)

    monkeypatch.setattr(
        caustics_source_geometry,
        "_sample_position",
        sample_position,
    )
    monkeypatch.setattr(
        caustics_source_geometry,
        "_source_boundary_clearance",
        lambda *args, **kwargs: 0.5,
    )
    graph_state = _graph_state_for(node, 1)

    with pytest.raises(
        RuntimeError,
        match="penultimate expected_num_images=3, final expected_num_images=4",
    ):
        node.compute(graph_state, rng_info=np.random.default_rng(5))

    assert sampler_calls == 1
    assert not set(_SOURCE_OUTPUTS).intersection(graph_state[node.node_string])


def test_image_node_resolves_absolute_and_relative_angular_settings(
    valid_sis_spec,
    fixed_cosmology,
):
    """Apply one characteristic scale only to enabled numerical fractions."""
    adapter = _FakeGeometryAdapter()
    absolute = _image_node(
        valid_sis_spec,
        fixed_cosmology,
        pixelscale=0.8,
        epsilon=0.3,
    )
    relative = _image_node(
        valid_sis_spec,
        fixed_cosmology,
        pixelscale=0.8,
        pixelscale_fraction=0.25,
        epsilon=0.3,
        epsilon_fraction=0.1,
    )
    capped = _image_node(
        valid_sis_spec,
        fixed_cosmology,
        pixelscale=0.8,
        pixelscale_fraction=1.0,
        epsilon=0.3,
        epsilon_fraction=1.0,
    )

    assert absolute._realized_angular_settings(adapter, {}) == (0.8, 0.3)
    assert relative._realized_angular_settings(adapter, {}) == (0.5, 0.2)
    assert capped._realized_angular_settings(adapter, {}) == (0.8, 0.3)


@pytest.mark.parametrize(
    ("name", "value", "error_type", "message"),
    [
        (
            "source_x",
            "not-numeric",
            TypeError,
            "source_x must realize to a scalar numeric value in arcseconds",
        ),
        (
            "source_y",
            object(),
            TypeError,
            "source_y must realize to a scalar numeric value in arcseconds",
        ),
        (
            "source_x",
            np.inf,
            ValueError,
            "source_x must realize to a finite value in arcseconds",
        ),
        (
            "source_y",
            np.nan,
            ValueError,
            "source_y must realize to a finite value in arcseconds",
        ),
    ],
)
def test_image_solve_validates_realized_source_coordinates_before_lens_build(
    monkeypatch,
    valid_sis_spec,
    fixed_cosmology,
    name,
    value,
    error_type,
    message,
):
    """Assign conversion and finiteness failures to each source coordinate."""
    node = _image_node(valid_sis_spec, fixed_cosmology)
    values = _image_values()
    values[name] = value
    monkeypatch.setattr(
        caustics_lens_system,
        "_build_lens_system",
        lambda *args, **kwargs: pytest.fail("invalid coordinates must fail before lens build"),
    )

    with pytest.raises(error_type, match=message):
        node._solve_one(values)


@pytest.mark.parametrize("expected_num_images", [1.5, np.int64(1), 5])
def test_image_solve_rejects_invalid_realized_expected_counts_before_lens_build(
    monkeypatch,
    valid_sis_spec,
    fixed_cosmology,
    expected_num_images,
):
    """Require a realized integer count within the configured inclusive range."""
    node = _image_node(valid_sis_spec, fixed_cosmology)
    monkeypatch.setattr(
        caustics_lens_system,
        "_build_lens_system",
        lambda *args, **kwargs: pytest.fail("invalid count must fail before lens build"),
    )

    with pytest.raises(
        ValueError,
        match="expected_num_images must be None or an integer between min_images and max_images",
    ):
        node._solve_one(_image_values(expected_num_images=expected_num_images))


@pytest.mark.parametrize(
    ("realized_fov", "expected_initial_fov", "uses_adapter"),
    [(None, 9.0, True), ("5.0", 7.5, False)],
)
def test_image_solve_derives_none_fov_or_converts_explicit_fov(
    monkeypatch,
    valid_sis_spec,
    fixed_cosmology,
    realized_fov,
    expected_initial_fov,
    uses_adapter,
):
    """Derive only a realized ``None`` while accepting float-convertible FOVs."""
    node = _image_node(
        valid_sis_spec,
        fixed_cosmology,
        fov_multiplier=1.5,
        pixelscale=1.0,
    )
    lens = _ArrayLens([1.0, 2.0], [0.0, 1.0])
    adapter_values = object()

    class RecordingAdapter(_FakeGeometryAdapter):
        def __init__(self):
            self.initial_fov_calls = []

        def initial_fov(self, values):
            self.initial_fov_calls.append(values)
            return 6.0

    adapter = RecordingAdapter()
    monkeypatch.setattr(
        caustics_lens_system,
        "_build_lens_system",
        lambda *args, **kwargs: (lens, adapter, adapter_values),
    )
    monkeypatch.setattr(
        caustics_runtime,
        "_import_caustics_dependencies",
        lambda: (object(), _FakeTorch),
    )
    forward_calls = []

    def forward(*args, **kwargs):
        del args
        forward_calls.append(kwargs)
        return np.array([[0.0, 0.0], [1.0, 0.0]])

    monkeypatch.setattr(node, "_forward_raytrace_images", forward)

    result = node._solve_one(_image_values(fov=realized_fov))

    assert adapter.initial_fov_calls == ([adapter_values] if uses_adapter else [])
    assert len(forward_calls) == 1
    assert forward_calls[0]["current_fov"] == expected_initial_fov
    assert result[6]["solver_fov"] == expected_initial_fov


@pytest.mark.parametrize(
    ("realized_fov", "error_type", "message"),
    [
        (object(), TypeError, "fov must realize to None or a scalar numeric value"),
        (0.0, ValueError, "fov must realize to None or a positive finite value"),
        (-1.0, ValueError, "fov must realize to None or a positive finite value"),
        (np.inf, ValueError, "fov must realize to None or a positive finite value"),
    ],
)
def test_image_solve_rejects_invalid_explicit_realized_fov(
    monkeypatch,
    valid_sis_spec,
    fixed_cosmology,
    realized_fov,
    error_type,
    message,
):
    """Reject explicit FOV conversion and positive-finite domain failures."""
    node = _image_node(valid_sis_spec, fixed_cosmology)
    _patch_image_runtime(monkeypatch, _FakeLens(), _FakeGeometryAdapter())
    monkeypatch.setattr(
        node,
        "_forward_raytrace_images",
        lambda *args, **kwargs: pytest.fail("invalid FOV must fail before solving"),
    )

    with pytest.raises(error_type, match=message):
        node._solve_one(_image_values(fov=realized_fov))


def test_image_solve_rejects_initial_fov_not_larger_than_realized_pixelscale(
    monkeypatch,
    valid_sis_spec,
    fixed_cosmology,
):
    """Apply the realized FOV/scale relation before creating source tensors."""
    node = _image_node(
        valid_sis_spec,
        fixed_cosmology,
        pixelscale=2.0,
        fov_multiplier=0.5,
    )
    _patch_image_runtime(monkeypatch, _FakeLens(), _FakeGeometryAdapter())

    with pytest.raises(
        ValueError,
        match="Initial solver fov=2.0 arcsec must be larger than pixelscale=2.0 arcsec",
    ):
        node._solve_one(_image_values(fov=4.0))


def test_image_one_attempt_uses_actual_spacing_and_sorts_all_observables(
    monkeypatch,
    valid_sis_spec,
    fixed_cosmology,
):
    """Accept one global solve with exact delay/x/y ordering and diagnostics."""
    node = _image_node(
        valid_sis_spec,
        fixed_cosmology,
        max_images=5,
        min_images=2,
        expected_num_images=4,
        fov=5.0,
        fov_multiplier=2.0,
        pixelscale=0.9,
        pixelscale_fraction=0.2,
        epsilon=0.3,
        epsilon_fraction=0.1,
    )
    lens = _ArrayLens(
        [-2.0, 3.0, -4.0, 5.0],
        [10.0, 10.0, 10.0, 11.0],
        convergences=[0.1, 0.2, 0.3, 0.4],
        shear1=[3.0, 5.0, 8.0, 7.0],
        shear2=[4.0, 12.0, 15.0, 24.0],
    )
    _patch_image_runtime(monkeypatch, lens, _FakeGeometryAdapter())
    calls = []
    coordinates = np.array(
        [
            [2.0, 2.0],
            [1.0, 3.0],
            [1.0, 2.0],
            [0.0, 9.0],
        ]
    )

    def forward(*args, **kwargs):
        del args
        calls.append(kwargs)
        return coordinates

    monkeypatch.setattr(node, "_forward_raytrace_images", forward)

    (
        image_x,
        image_y,
        magnifications,
        delays,
        convergences,
        shears,
        diagnostics,
    ) = node._solve_one(_image_values(fov=5.0, expected_num_images=4))

    assert len(calls) == 1
    assert calls[0] == {
        "center_x": 0.0,
        "center_y": 0.0,
        "current_fov": 10.0,
        "divisions": 25,
        "epsilon": 0.2,
    }
    np.testing.assert_array_equal(image_x, [1.0, 1.0, 2.0, 0.0])
    np.testing.assert_array_equal(image_y, [2.0, 3.0, 2.0, 9.0])
    np.testing.assert_array_equal(magnifications, [4.0, 3.0, 2.0, 5.0])
    np.testing.assert_array_equal(delays, [0.0, 0.0, 0.0, 1.0])
    np.testing.assert_array_equal(convergences, [0.3, 0.2, 0.1, 0.4])
    np.testing.assert_array_equal(shears, [17.0, 13.0, 5.0, 25.0])
    assert diagnostics == {
        "image_count_deficit": 0,
        "solver_fov": 10.0,
        "solver_pixelscale": 0.4,
        "solver_attempts": 1,
        "solver_fov_expansions": 0,
        "solver_pixelscale_refinements": 0,
    }
    assert lens.magnification_calls == 1
    assert lens.delay_calls == 1
    assert lens.convergence_calls == 1
    assert lens.shear_calls == 1


def test_image_singular_base_uses_divisions_plus_one_variant(
    monkeypatch,
    valid_sis_spec,
    fixed_cosmology,
):
    """Advance one singular base grid to the parity-changing variant."""
    node = _image_node(
        valid_sis_spec,
        fixed_cosmology,
        fov=4.0,
        pixelscale=1.5,
    )
    _patch_image_runtime(monkeypatch, _FakeLens(), _FakeGeometryAdapter())
    calls = []

    def forward(*args, **kwargs):
        del args
        calls.append(kwargs)
        if len(calls) == 1:
            raise RuntimeError("torch.linalg.solve: input matrix is singular")
        return np.array([[3.0, 0.0], [1.0, 0.0]])

    monkeypatch.setattr(node, "_forward_raytrace_images", forward)

    _, _, _, _, _, _, diagnostics = node._solve_one(_image_values())

    assert [(call["center_x"], call["center_y"], call["divisions"]) for call in calls] == [
        (0.0, 0.0, 3),
        (0.0, 0.0, 4),
    ]
    assert diagnostics["solver_attempts"] == 2
    assert diagnostics["solver_pixelscale"] == 1.0
    assert diagnostics["solver_fov_expansions"] == 0


def test_image_three_singular_variants_then_expand_fov(
    monkeypatch,
    valid_sis_spec,
    fixed_cosmology,
):
    """Exhaust base, parity, and half-cell variants before the outer schedule."""
    node = _image_node(
        valid_sis_spec,
        fixed_cosmology,
        fov=4.0,
        pixelscale=1.0,
        max_fov_expansions=1,
        fov_expansion_factor=2.0,
    )
    _patch_image_runtime(monkeypatch, _FakeLens(), _FakeGeometryAdapter())
    calls = []

    def forward(*args, **kwargs):
        del args
        calls.append(kwargs)
        if len(calls) <= 3:
            raise RuntimeError("linalg.solve failed with singular U")
        return np.array([[2.0, 0.0], [1.0, 0.0]])

    monkeypatch.setattr(node, "_forward_raytrace_images", forward)

    _, _, _, _, _, _, diagnostics = node._solve_one(_image_values())

    assert [call["current_fov"] for call in calls] == [4.0, 4.0, 4.0, 8.0]
    assert [(call["center_x"], call["center_y"], call["divisions"]) for call in calls] == [
        (0.0, 0.0, 4),
        (0.0, 0.0, 5),
        (0.5, 0.5, 4),
        (0.0, 0.0, 8),
    ]
    assert diagnostics["solver_attempts"] == 4
    assert diagnostics["solver_fov"] == 8.0
    assert diagnostics["solver_pixelscale"] == 1.0
    assert diagnostics["solver_fov_expansions"] == 1


def test_image_empty_candidate_error_moves_directly_to_outer_schedule(
    monkeypatch,
    valid_sis_spec,
    fixed_cosmology,
):
    """Skip remaining inner variants for the exact empty-candidate failure."""
    node = _image_node(
        valid_sis_spec,
        fixed_cosmology,
        max_fov_expansions=1,
        fov_expansion_factor=2.0,
    )
    _patch_image_runtime(monkeypatch, _FakeLens(), _FakeGeometryAdapter())
    calls = []

    def forward(*args, **kwargs):
        del args
        calls.append(kwargs)
        if len(calls) == 1:
            raise IndexError("index 0 is out of bounds for dimension 0")
        return np.array([[2.0, 0.0], [1.0, 0.0]])

    monkeypatch.setattr(node, "_forward_raytrace_images", forward)

    _, _, _, _, _, _, diagnostics = node._solve_one(_image_values())

    assert [(call["current_fov"], call["divisions"]) for call in calls] == [
        (4.0, 4),
        (8.0, 8),
    ]
    assert diagnostics["solver_attempts"] == 2
    assert diagnostics["solver_fov_expansions"] == 1


def test_image_unrelated_global_error_propagates_without_retry(
    monkeypatch,
    valid_sis_spec,
    fixed_cosmology,
):
    """Leave exceptions outside both narrow classifiers untouched."""
    node = _image_node(valid_sis_spec, fixed_cosmology)
    _patch_image_runtime(monkeypatch, _FakeLens(), _FakeGeometryAdapter())
    calls = []
    unexpected = RuntimeError("backend exploded for another reason")

    def forward(*args, **kwargs):
        del args, kwargs
        calls.append("called")
        raise unexpected

    monkeypatch.setattr(node, "_forward_raytrace_images", forward)

    with pytest.raises(RuntimeError) as error:
        node._solve_one(_image_values())

    assert error.value is unexpected
    assert calls == ["called"]


def test_image_outer_schedule_finishes_fov_before_pixelscale_refinement(
    monkeypatch,
    valid_sis_spec,
    fixed_cosmology,
):
    """Run every expansion before reducing the requested pixel scale."""
    node = _image_node(
        valid_sis_spec,
        fixed_cosmology,
        expected_num_images=3,
        fov=4.0,
        pixelscale=1.5,
        max_fov_expansions=2,
        fov_expansion_factor=2.0,
        max_pixelscale_refinements=2,
        pixelscale_refinement_factor=0.5,
    )
    lens = _ArrayLens([2.0, -3.0], [1.0, 0.0])
    _patch_image_runtime(monkeypatch, lens, _FakeGeometryAdapter())
    calls = []

    def forward(*args, **kwargs):
        del args
        calls.append(kwargs)
        return np.array([[2.0, 0.0], [1.0, 0.0]])

    monkeypatch.setattr(node, "_forward_raytrace_images", forward)

    _, _, _, _, _, _, diagnostics = node._solve_one(_image_values(expected_num_images=3))

    assert [call["current_fov"] for call in calls] == [4.0, 8.0, 16.0, 16.0, 16.0]
    assert [call["divisions"] for call in calls] == [3, 6, 11, 22, 43]
    assert diagnostics == {
        "image_count_deficit": 1,
        "solver_fov": 16.0,
        "solver_pixelscale": 16.0 / 43.0,
        "solver_attempts": 5,
        "solver_fov_expansions": 2,
        "solver_pixelscale_refinements": 2,
    }


def test_image_retry_exhaustion_reports_latest_error_and_variant(
    monkeypatch,
    valid_sis_spec,
    fixed_cosmology,
):
    """Retain the final classified failure in below-minimum diagnostics."""
    node = _image_node(valid_sis_spec, fixed_cosmology)
    _patch_image_runtime(monkeypatch, _FakeLens(), _FakeGeometryAdapter())
    calls = []
    errors = [
        RuntimeError("torch.linalg.solve: input matrix is singular"),
        RuntimeError("linalg.solve failed with singular U on parity grid"),
        RuntimeError("torch.linalg.solve failed with singular U on shifted grid"),
    ]

    def forward(*args, **kwargs):
        del args
        calls.append(kwargs)
        raise errors[len(calls) - 1]

    monkeypatch.setattr(node, "_forward_raytrace_images", forward)

    with pytest.raises(RuntimeError) as error:
        node._solve_one(_image_values())

    assert [call["divisions"] for call in calls] == [4, 5, 4]
    message = str(error.value)
    assert "Caustics image recovery exhausted; target_count=2" in message
    assert "grid_variant=half_cell_shift" in message
    assert "solver_attempts=3" in message
    assert (
        "latest_retryable_error=RuntimeError: torch.linalg.solve failed with singular U on shifted grid"
        in message
    )


def test_image_targeted_recovery_adds_one_batch_and_keeps_duplicates(
    monkeypatch,
    valid_sis_spec,
    fixed_cosmology,
):
    """Seed only empty neighborhoods and append certified roots without deduplication."""
    node = _image_node(
        valid_sis_spec,
        fixed_cosmology,
        max_images=4,
        min_images=2,
        expected_num_images=3,
        epsilon=0.2,
    )
    lens = _ArrayLens([-1.0, 2.0, -3.0], [0.0, 1.0, 1.0])
    adapter = _RecoveryGeometryAdapter(((0.0, 0.0), (2.0, 0.0), (4.0, 0.0)))
    _patch_image_runtime(monkeypatch, lens, adapter)
    monkeypatch.setattr(
        node,
        "_forward_raytrace_images",
        lambda *args, **kwargs: np.array([[0.0, 0.0], [4.0, 0.0]]),
    )
    seed_calls = []
    refine_calls = []

    def recovery_seeds(lens_arg, recovery_points, *, source_x, source_y, radius):
        seed_calls.append((lens_arg, recovery_points.copy(), source_x, source_y, radius))
        return np.array([[10.0, 0.0], [20.0, 0.0], [30.0, 0.0]])

    def refine_seeds(
        lens_arg,
        torch,
        seeds,
        recovery_points,
        beta_x,
        beta_y,
        epsilon,
        neighborhood_radius,
    ):
        refine_calls.append(
            (
                lens_arg,
                torch,
                seeds.copy(),
                recovery_points.copy(),
                float(beta_x.values),
                float(beta_y.values),
                epsilon,
                neighborhood_radius,
            )
        )
        return np.array([[4.0, 0.0]])

    monkeypatch.setattr(caustics_image_recovery, "_recovery_image_seeds", recovery_seeds)
    monkeypatch.setattr(caustics_image_recovery, "_refine_image_seeds", refine_seeds)

    image_x, image_y, _, _, _, _, diagnostics = node._solve_one(_image_values(expected_num_images=3))

    assert len(seed_calls) == 1
    assert seed_calls[0][0] is lens
    np.testing.assert_array_equal(seed_calls[0][1], [[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]])
    assert seed_calls[0][2:] == (0.0, 0.0, 0.2)
    assert len(refine_calls) == 1
    assert refine_calls[0][0] is lens
    assert refine_calls[0][1] is _FakeTorch
    np.testing.assert_array_equal(refine_calls[0][2], [[20.0, 0.0]])
    np.testing.assert_array_equal(refine_calls[0][3], [[2.0, 0.0]])
    assert refine_calls[0][4:] == (0.0, 0.0, 0.2, 1.0)
    np.testing.assert_array_equal(image_x, [0.0, 4.0, 4.0])
    np.testing.assert_array_equal(image_y, [0.0, 0.0, 0.0])
    assert diagnostics["image_count_deficit"] == 0
    assert diagnostics["solver_attempts"] == 2


@pytest.mark.parametrize(
    ("coordinates", "expected_num_images", "min_images", "recovery_points", "raises"),
    [
        (np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]), 3, 2, ((5.0, 0.0),), False),
        (np.empty((0, 2)), 2, 1, ((5.0, 0.0),), True),
        (np.array([[0.0, 0.0]]), 2, 1, ((0.5, 0.0),), False),
        (np.array([[0.0, 0.0]]), 2, 1, (), False),
    ],
)
def test_image_targeted_recovery_policy_gates(
    monkeypatch,
    valid_sis_spec,
    fixed_cosmology,
    coordinates,
    expected_num_images,
    min_images,
    recovery_points,
    raises,
):
    """Skip targeted work for complete, empty, occupied, or incapable results."""
    node = _image_node(
        valid_sis_spec,
        fixed_cosmology,
        min_images=min_images,
        expected_num_images=expected_num_images,
    )
    count = len(coordinates)
    lens = _ArrayLens(np.arange(1, count + 1), np.arange(count, dtype=float))
    _patch_image_runtime(monkeypatch, lens, _RecoveryGeometryAdapter(recovery_points))
    monkeypatch.setattr(
        node,
        "_forward_raytrace_images",
        lambda *args, **kwargs: coordinates,
    )

    def unexpected_recovery(*args, **kwargs):
        raise AssertionError("targeted recovery should not run")

    monkeypatch.setattr(caustics_image_recovery, "_recovery_image_seeds", unexpected_recovery)
    monkeypatch.setattr(caustics_image_recovery, "_refine_image_seeds", unexpected_recovery)

    if raises:
        with pytest.raises(RuntimeError, match="Caustics image recovery exhausted"):
            node._solve_one(_image_values(expected_num_images=expected_num_images))
    else:
        result = node._solve_one(_image_values(expected_num_images=expected_num_images))
        assert len(result[0]) == count
        assert result[6]["solver_attempts"] == 1


def test_image_retryable_targeted_failure_retains_deficient_global_result(
    monkeypatch,
    valid_sis_spec,
    fixed_cosmology,
):
    """Keep a sufficient global deficit when its one targeted batch is retryable."""
    node = _image_node(
        valid_sis_spec,
        fixed_cosmology,
        min_images=2,
        expected_num_images=3,
    )
    lens = _ArrayLens([2.0, -3.0], [1.0, 0.0])
    _patch_image_runtime(
        monkeypatch,
        lens,
        _RecoveryGeometryAdapter(((2.0, 0.0),)),
    )
    monkeypatch.setattr(
        node,
        "_forward_raytrace_images",
        lambda *args, **kwargs: np.array([[0.0, 0.0], [4.0, 0.0]]),
    )
    monkeypatch.setattr(
        caustics_image_recovery,
        "_recovery_image_seeds",
        lambda *args, **kwargs: np.array([[2.0, 0.0]]),
    )
    monkeypatch.setattr(
        caustics_image_recovery,
        "_refine_image_seeds",
        lambda *args, **kwargs: (_ for _ in ()).throw(IndexError("index 0 is out of bounds for dimension 0")),
    )

    image_x, _, _, _, _, _, diagnostics = node._solve_one(_image_values(expected_num_images=3))

    np.testing.assert_array_equal(image_x, [4.0, 0.0])
    assert diagnostics["image_count_deficit"] == 1
    assert diagnostics["solver_attempts"] == 2


def test_image_unclassified_targeted_refinement_error_propagates_without_outer_retry(
    monkeypatch,
    valid_sis_spec,
    fixed_cosmology,
):
    """Propagate a targeted exception outside both exact retry classifiers."""
    node = _image_node(
        valid_sis_spec,
        fixed_cosmology,
        min_images=2,
        expected_num_images=3,
        max_fov_expansions=2,
        max_pixelscale_refinements=2,
    )
    lens = _ArrayLens([1.0, 2.0], [0.0, 1.0])
    _patch_image_runtime(
        monkeypatch,
        lens,
        _RecoveryGeometryAdapter(((2.0, 0.0),)),
    )
    global_calls = []

    def forward(*args, **kwargs):
        del args
        global_calls.append(kwargs)
        return np.array([[0.0, 0.0], [4.0, 0.0]])

    monkeypatch.setattr(node, "_forward_raytrace_images", forward)
    monkeypatch.setattr(
        caustics_image_recovery,
        "_recovery_image_seeds",
        lambda *args, **kwargs: np.array([[2.0, 0.0]]),
    )
    unexpected = ArithmeticError("targeted refinement produced invalid state")
    refinement_calls = []

    def refine(*args, **kwargs):
        refinement_calls.append((args, kwargs))
        raise unexpected

    monkeypatch.setattr(caustics_image_recovery, "_refine_image_seeds", refine)

    with pytest.raises(ArithmeticError) as error:
        node._solve_one(_image_values(expected_num_images=3))

    assert error.value is unexpected
    assert len(global_calls) == 1
    assert len(refinement_calls) == 1


@pytest.mark.parametrize(
    ("node_settings", "expected_num_images", "coordinates", "message"),
    [
        (
            {"max_images": 4, "min_images": 2, "expected_num_images": 2},
            2,
            np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
            "exceeded expected_num_images",
        ),
        (
            {"max_images": 2, "min_images": 1, "expected_num_images": None},
            None,
            np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
            "exceeded max_images",
        ),
    ],
)
def test_image_count_overflow_policy(
    monkeypatch,
    valid_sis_spec,
    fixed_cosmology,
    node_settings,
    expected_num_images,
    coordinates,
    message,
):
    """Apply expected-count overflow before the independent maximum policy."""
    node = _image_node(valid_sis_spec, fixed_cosmology, **node_settings)
    lens = _ArrayLens([1.0] * len(coordinates), np.arange(len(coordinates)))
    _patch_image_runtime(monkeypatch, lens, _FakeGeometryAdapter())
    monkeypatch.setattr(
        node,
        "_forward_raytrace_images",
        lambda *args, **kwargs: coordinates,
    )

    with pytest.raises(RuntimeError, match=message) as error:
        node._solve_one(_image_values(expected_num_images=expected_num_images))

    assert "recovered_num_images=3" in str(error.value)
    assert "solver_attempts=1" in str(error.value)


@pytest.mark.parametrize(
    ("coordinates", "raises", "deficit"),
    [
        (np.array([[0.0, 0.0], [1.0, 0.0]]), False, 1),
        (np.array([[0.0, 0.0]]), True, None),
    ],
)
def test_image_bounded_deficit_requires_at_least_min_images(
    monkeypatch,
    valid_sis_spec,
    fixed_cosmology,
    coordinates,
    raises,
    deficit,
):
    """Return an above-minimum deficit but reject exhaustion below the floor."""
    node = _image_node(
        valid_sis_spec,
        fixed_cosmology,
        min_images=2,
        expected_num_images=3,
    )
    lens = _ArrayLens([1.0] * len(coordinates), np.arange(len(coordinates)))
    _patch_image_runtime(monkeypatch, lens, _FakeGeometryAdapter())
    monkeypatch.setattr(
        node,
        "_forward_raytrace_images",
        lambda *args, **kwargs: coordinates,
    )

    if raises:
        with pytest.raises(RuntimeError) as error:
            node._solve_one(_image_values(expected_num_images=3))
        assert "target_count=3" in str(error.value)
        assert "recovered_num_images=1" in str(error.value)
    else:
        result = node._solve_one(_image_values(expected_num_images=3))
        assert len(result[0]) == 2
        assert result[6]["image_count_deficit"] == deficit


def test_image_no_expectation_completes_at_minimum_with_deficit_sentinel(
    monkeypatch,
    valid_sis_spec,
    fixed_cosmology,
):
    """Complete the real solve policy at min_images and emit the no-target sentinel."""
    node = _image_node(
        valid_sis_spec,
        fixed_cosmology,
        min_images=2,
        expected_num_images=None,
        max_fov_expansions=2,
        max_pixelscale_refinements=2,
    )
    lens = _ArrayLens([-2.0, 3.0], [4.0, 1.0])
    _patch_image_runtime(monkeypatch, lens, _FakeGeometryAdapter())
    calls = []

    def forward(*args, **kwargs):
        del args
        calls.append(kwargs)
        return np.array([[2.0, 0.0], [1.0, 0.0]])

    monkeypatch.setattr(node, "_forward_raytrace_images", forward)

    image_x, image_y, magnifications, delays, convergences, shears, diagnostics = node._solve_one(
        _image_values(expected_num_images=None)
    )

    assert len(calls) == 1
    assert calls[0]["current_fov"] == 4.0
    assert calls[0]["divisions"] == 4
    np.testing.assert_array_equal(image_x, [1.0, 2.0])
    np.testing.assert_array_equal(image_y, [0.0, 0.0])
    np.testing.assert_array_equal(magnifications, [3.0, 2.0])
    np.testing.assert_array_equal(delays, [0.0, 3.0])
    np.testing.assert_array_equal(convergences, [0.0, 0.0])
    np.testing.assert_array_equal(shears, [0.0, 0.0])
    assert diagnostics == {
        "image_count_deficit": -1,
        "solver_fov": 4.0,
        "solver_pixelscale": 1.0,
        "solver_attempts": 1,
        "solver_fov_expansions": 0,
        "solver_pixelscale_refinements": 0,
    }


def test_image_compute_single_sample_packs_padding_and_preserves_fixed_output(
    monkeypatch,
    valid_sis_spec,
    fixed_cosmology,
):
    """Pack one active set to width M and leave a pre-fixed count unchanged."""
    node = _image_node(
        valid_sis_spec,
        fixed_cosmology,
        max_images=4,
        expected_num_images=None,
        pixelscale=0.25,
        max_fov_expansions=1,
    )
    solved = (
        np.array([2.0, 1.0]),
        np.array([-2.0, -1.0]),
        np.array([5.0, 6.0]),
        np.array([0.0, 3.0]),
        np.array([0.2, 0.4]),
        np.array([0.5, 0.7]),
        {
            "image_count_deficit": -1,
            "solver_fov": 8.0,
            "solver_pixelscale": 0.25,
            "solver_attempts": 3,
            "solver_fov_expansions": 1,
            "solver_pixelscale_refinements": 0,
        },
    )
    monkeypatch.setattr(node, "_solve_one", lambda values: solved)
    graph_state = _graph_state_for(
        node,
        1,
        fixed_outputs={"num_images": 99},
    )
    caller_rng = np.random.default_rng(91)
    expected_next_draw = np.random.default_rng(91).random()

    results = node.compute(graph_state, rng_info=caller_rng)
    saved = graph_state[node.node_string]

    assert caller_rng.random() == expected_next_draw
    assert tuple(name for name in saved if name in _IMAGE_OUTPUTS) == _IMAGE_OUTPUTS
    assert "cosmology" not in saved
    assert np.isscalar(results[0])
    assert all(np.isscalar(result) for result in results[7:])
    assert all(result.shape == (4,) for result in results[1:7])
    assert results[0] == 2
    np.testing.assert_array_equal(results[1], [2.0, 1.0, np.nan, np.nan])
    np.testing.assert_array_equal(results[2], [-2.0, -1.0, np.nan, np.nan])
    np.testing.assert_array_equal(results[3], [5.0, 6.0, 0.0, 0.0])
    np.testing.assert_array_equal(results[4], [0.0, 3.0, np.nan, np.nan])
    np.testing.assert_array_equal(results[5], [0.2, 0.4, np.nan, np.nan])
    np.testing.assert_array_equal(results[6], [0.5, 0.7, np.nan, np.nan])
    assert results[7] == -1
    assert results[8] == 8.0
    assert results[9] == 0.25
    assert results[10] == 3
    assert results[11] == 1
    assert results[12] == 0
    assert saved["num_images"] == 99
    np.testing.assert_array_equal(saved["image_x"], [2.0, 1.0, np.nan, np.nan])
    np.testing.assert_array_equal(saved["image_y"], [-2.0, -1.0, np.nan, np.nan])
    np.testing.assert_array_equal(saved["macro_magnifications"], [5.0, 6.0, 0.0, 0.0])
    np.testing.assert_array_equal(saved["time_delays"], [0.0, 3.0, np.nan, np.nan])
    np.testing.assert_array_equal(saved["macro_convergences"], [0.2, 0.4, np.nan, np.nan])
    np.testing.assert_array_equal(saved["macro_shear"], [0.5, 0.7, np.nan, np.nan])
    assert saved["image_count_deficit"] == -1
    assert saved["solver_fov"] == 8.0
    assert saved["solver_pixelscale"] == 0.25
    assert saved["solver_attempts"] == 3
    assert saved["solver_fov_expansions"] == 1
    assert saved["solver_pixelscale_refinements"] == 0


def test_image_compute_multiple_samples_packs_rows_and_persists_all_outputs(
    monkeypatch,
    valid_sis_spec,
    fixed_cosmology,
):
    """Pack independent active sets into S-by-M arrays in sample order."""
    node = _image_node(
        valid_sis_spec,
        fixed_cosmology,
        max_images=4,
        min_images=1,
        expected_num_images=None,
        pixelscale=0.25,
        max_fov_expansions=2,
        max_pixelscale_refinements=1,
    )
    solved = iter(
        [
            (
                np.array([2.0, 1.0]),
                np.array([-2.0, -1.0]),
                np.array([5.0, 6.0]),
                np.array([0.0, 3.0]),
                np.array([0.2, 0.4]),
                np.array([0.5, 0.7]),
                {
                    "image_count_deficit": -1,
                    "solver_fov": 8.0,
                    "solver_pixelscale": 0.25,
                    "solver_attempts": 3,
                    "solver_fov_expansions": 1,
                    "solver_pixelscale_refinements": 0,
                },
            ),
            (
                np.array([7.0]),
                np.array([-7.0]),
                np.array([9.0]),
                np.array([0.0]),
                np.array([0.9]),
                np.array([1.1]),
                {
                    "image_count_deficit": -1,
                    "solver_fov": 16.0,
                    "solver_pixelscale": 0.125,
                    "solver_attempts": 5,
                    "solver_fov_expansions": 2,
                    "solver_pixelscale_refinements": 1,
                },
            ),
        ]
    )
    seen_source_x = []

    def solve_one(values):
        seen_source_x.append(values["source_x"])
        return next(solved)

    monkeypatch.setattr(node, "_solve_one", solve_one)
    graph_state = _graph_state_for(
        node,
        2,
        overrides={"source_x": np.array([0.1, 0.2])},
    )

    results = node.compute(graph_state, rng_info=np.random.default_rng(12))
    saved = graph_state[node.node_string]

    np.testing.assert_array_equal(seen_source_x, [0.1, 0.2])
    assert tuple(name for name in saved if name in _IMAGE_OUTPUTS) == _IMAGE_OUTPUTS
    assert all(result.shape == (2,) for result in (results[0], *results[7:]))
    assert all(result.shape == (2, 4) for result in results[1:7])
    np.testing.assert_array_equal(results[0], [2, 1])
    np.testing.assert_array_equal(
        results[1],
        [[2.0, 1.0, np.nan, np.nan], [7.0, np.nan, np.nan, np.nan]],
    )
    np.testing.assert_array_equal(
        results[2],
        [[-2.0, -1.0, np.nan, np.nan], [-7.0, np.nan, np.nan, np.nan]],
    )
    np.testing.assert_array_equal(results[3], [[5.0, 6.0, 0.0, 0.0], [9.0, 0.0, 0.0, 0.0]])
    np.testing.assert_array_equal(
        results[4],
        [[0.0, 3.0, np.nan, np.nan], [0.0, np.nan, np.nan, np.nan]],
    )
    np.testing.assert_array_equal(
        results[5],
        [[0.2, 0.4, np.nan, np.nan], [0.9, np.nan, np.nan, np.nan]],
    )
    np.testing.assert_array_equal(
        results[6],
        [[0.5, 0.7, np.nan, np.nan], [1.1, np.nan, np.nan, np.nan]],
    )
    np.testing.assert_array_equal(results[7], [-1, -1])
    np.testing.assert_array_equal(results[8], [8.0, 16.0])
    np.testing.assert_array_equal(results[9], [0.25, 0.125])
    np.testing.assert_array_equal(results[10], [3, 5])
    np.testing.assert_array_equal(results[11], [1, 2])
    np.testing.assert_array_equal(results[12], [0, 1])
    np.testing.assert_array_equal(saved["num_images"], [2, 1])
    np.testing.assert_array_equal(
        saved["image_x"],
        [[2.0, 1.0, np.nan, np.nan], [7.0, np.nan, np.nan, np.nan]],
    )
    np.testing.assert_array_equal(
        saved["image_y"],
        [[-2.0, -1.0, np.nan, np.nan], [-7.0, np.nan, np.nan, np.nan]],
    )
    np.testing.assert_array_equal(
        saved["macro_magnifications"],
        [[5.0, 6.0, 0.0, 0.0], [9.0, 0.0, 0.0, 0.0]],
    )
    np.testing.assert_array_equal(
        saved["time_delays"],
        [[0.0, 3.0, np.nan, np.nan], [0.0, np.nan, np.nan, np.nan]],
    )
    np.testing.assert_array_equal(
        saved["macro_convergences"],
        [[0.2, 0.4, np.nan, np.nan], [0.9, np.nan, np.nan, np.nan]],
    )
    np.testing.assert_array_equal(
        saved["macro_shear"],
        [[0.5, 0.7, np.nan, np.nan], [1.1, np.nan, np.nan, np.nan]],
    )
    np.testing.assert_array_equal(saved["image_count_deficit"], [-1, -1])
    np.testing.assert_array_equal(saved["solver_fov"], [8.0, 16.0])
    np.testing.assert_array_equal(saved["solver_pixelscale"], [0.25, 0.125])
    np.testing.assert_array_equal(saved["solver_attempts"], [3, 5])
    np.testing.assert_array_equal(saved["solver_fov_expansions"], [1, 2])
    np.testing.assert_array_equal(saved["solver_pixelscale_refinements"], [0, 1])

from types import SimpleNamespace

import numpy as np
import pytest

from lightcurvelynx.models import caustics_models
from lightcurvelynx.models._caustics import lens_system, runtime


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


class _ValueCountPointGeometryAdapter(lens_system._PointSingularityGeometryAdapter):
    """Read a controlled reference image count from component values."""

    def reference_num_images(self, values):
        """Return the component's configured reference image count."""
        return values["reference_count"]


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


class _RecordingTorch(_FakeTorch):
    """Record scalar values and dtypes used during lens-system realization."""

    events = None

    @classmethod
    def as_tensor(cls, values, dtype=None):
        """Record tensorization before returning the minimal tensor double."""
        normalized = np.asarray(values)
        recorded_value = normalized.item() if normalized.ndim == 0 else normalized.copy()
        if cls.events is not None:
            cls.events.append(("as_tensor", recorded_value, dtype))
        return _FakeTensor(values)


class _RecordingParameter:
    """Record dtype staticization for one fake Caustics parameter."""

    def __init__(self, value, *, events=None, owner=None, name=None):
        self.value = _FakeTensor(value)
        self.dtype_calls = []
        self.events = events
        self.owner = owner
        self.name = name

    def to(self, *, dtype):
        """Record the requested dtype and preserve parameter identity."""
        self.dtype_calls.append(dtype)
        if self.events is not None:
            self.events.append(("parameter_to", self.owner, self.name, dtype))
        return self


class _RecordingSIS:
    """Record recursive construction and atomic staticization for an SIS."""

    events = None

    def __init__(self, cosmology, z_l, z_s, x0, y0, Rein, s=0.0, name=None):
        self.cosmology = cosmology
        self.z_l = z_l
        self.z_s = z_s
        self.x0 = _RecordingParameter(
            x0,
            events=self.events,
            owner=name,
            name="x0",
        )
        self.y0 = _RecordingParameter(
            y0,
            events=self.events,
            owner=name,
            name="y0",
        )
        self.Rein = _RecordingParameter(
            Rein,
            events=self.events,
            owner=name,
            name="Rein",
        )
        self.s = s
        self.name = name
        self.static_calls = 0

    def to_static(self):
        """Record conversion of the realized atomic lens to static mode."""
        self.static_calls += 1
        if self.events is not None:
            self.events.append(("to_static", self.name))
        return self


class _RecordingSinglePlane:
    """Record redshift ownership and recursive names for a fake lens plane."""

    def __init__(self, cosmology, z_l, z_s, lenses, name=None):
        self.cosmology = cosmology
        self.z_l = z_l
        self.z_s = z_s
        self.lenses = tuple(lenses)
        self.name = name


@pytest.fixture
def fake_caustics_registry(monkeypatch):
    """Replace the optional Caustics import with an explicit constructor."""
    registry = SimpleNamespace(
        SIS=_FakeSIS,
        ExternalShear=_FakeExternalShear,
        MassSheet=_FakeMassSheet,
        SinglePlane=_FakeSinglePlane,
    )
    monkeypatch.setattr(runtime, "_import_caustics", lambda: registry)
    return registry


@pytest.fixture
def recording_caustics_runtime(monkeypatch):
    """Install recording constructors for full recursive lens realization."""
    events = []
    monkeypatch.setattr(_RecordingSIS, "events", events)
    monkeypatch.setattr(_RecordingTorch, "events", events)
    module = SimpleNamespace(
        SIS=_RecordingSIS,
        SinglePlane=_RecordingSinglePlane,
    )
    monkeypatch.setattr(runtime, "_import_caustics", lambda: module)
    monkeypatch.setattr(
        runtime,
        "_import_caustics_dependencies",
        lambda: (module, _RecordingTorch),
    )
    return SimpleNamespace(module=module, events=events)


@pytest.fixture
def fixed_cosmology():
    """Return an opaque cosmology whose identity must remain node-owned."""
    return object()


def test_root_validation_requires_redshift_and_rejects_nested_redshift(fake_caustics_registry):
    """Enforce root ownership of the shared lens-plane redshift."""
    del fake_caustics_registry
    missing_root = caustics_models.CausticsLensSpec(
        "SIS",
        {"x0": 0.0, "y0": 0.0, "Rein": 1.0},
    )
    with pytest.raises(ValueError, match="root lens specification requires: z_l"):
        lens_system._validate_root_lens_spec(missing_root)

    child = caustics_models.CausticsLensSpec(
        "SIS",
        {"z_l": 0.4, "x0": 0.0, "y0": 0.0, "Rein": 1.0},
    )
    root = caustics_models.CausticsLensSpec(
        "SinglePlane",
        {"z_l": 0.4, "lenses": [child]},
    )
    with pytest.raises(ValueError, match="inherit rather than define: z_l"):
        lens_system._validate_root_lens_spec(root)


def test_root_validation_rejects_atomic_and_composite_affine_only_lenses(fake_caustics_registry):
    """Reject every root tree that contains only affine perturbations."""
    del fake_caustics_registry
    mass_sheet = caustics_models.CausticsLensSpec(
        "MassSheet",
        {"z_l": 0.5, "x0": 0.0, "y0": 0.0, "kappa": 0.1},
    )
    shear = caustics_models.CausticsLensSpec(
        "ExternalShear",
        {"x0": 0.0, "y0": 0.0, "gamma_1": 0.1, "gamma_2": 0.0},
    )
    nested_mass_sheet = caustics_models.CausticsLensSpec(
        "MassSheet",
        {"x0": 0.0, "y0": 0.0, "kappa": 0.2},
    )
    plane = caustics_models.CausticsLensSpec(
        "SinglePlane",
        {"z_l": 0.5, "lenses": [shear, nested_mass_sheet]},
    )

    for affine_lens in (mass_sheet, plane):
        with pytest.raises(ValueError, match="at least one non-affine lens model"):
            lens_system._validate_root_lens_spec(affine_lens)


def test_root_validation_accepts_atomic_mixed_and_nested_non_affine_lenses(fake_caustics_registry):
    """Accept valid roots containing a non-affine atomic lens at any depth."""
    del fake_caustics_registry
    atomic = caustics_models.CausticsLensSpec(
        "SIS",
        {"z_l": 0.5, "x0": 0.0, "y0": 0.0, "Rein": 1.0},
    )
    nested_sis = caustics_models.CausticsLensSpec(
        "SIS",
        {"x0": 1.0, "y0": -1.0, "Rein": 0.8},
    )
    mass_sheet = caustics_models.CausticsLensSpec(
        "MassSheet",
        {"x0": 0.0, "y0": 0.0, "kappa": 0.1},
    )
    mixed = caustics_models.CausticsLensSpec(
        "SinglePlane",
        {"z_l": 0.5, "lenses": [nested_sis, mass_sheet]},
    )
    inner = caustics_models.CausticsLensSpec(
        "SinglePlane",
        {"lenses": [nested_sis, mass_sheet]},
    )
    nested = caustics_models.CausticsLensSpec(
        "SinglePlane",
        {"z_l": 0.5, "lenses": [inner]},
    )

    assert lens_system._validate_root_lens_spec(atomic) is None
    assert lens_system._validate_root_lens_spec(mixed) is None
    assert lens_system._validate_root_lens_spec(nested) is None


def test_lens_graph_inputs_preserve_mapping_order_and_recursive_paths(fake_caustics_registry):
    """Flatten recursive inputs in mapping order under positional path names."""
    del fake_caustics_registry
    values = {name: object() for name in ("root_z", "first_rein", "first_x", "first_y")}
    values.update({name: object() for name in ("nested_x", "nested_y", "nested_rein", "mass_kappa")})
    values.update({name: object() for name in ("mass_x", "mass_y")})
    first = caustics_models.CausticsLensSpec(
        "SIS",
        {
            "Rein": values["first_rein"],
            "x0": values["first_x"],
            "y0": values["first_y"],
        },
    )
    nested_atomic = caustics_models.CausticsLensSpec(
        "SIS",
        {
            "x0": values["nested_x"],
            "y0": values["nested_y"],
            "Rein": values["nested_rein"],
        },
    )
    nested_affine = caustics_models.CausticsLensSpec(
        "MassSheet",
        {
            "kappa": values["mass_kappa"],
            "x0": values["mass_x"],
            "y0": values["mass_y"],
        },
    )
    inner = caustics_models.CausticsLensSpec(
        "SinglePlane",
        {"lenses": [nested_atomic, nested_affine]},
    )
    root = caustics_models.CausticsLensSpec(
        "SinglePlane",
        {"lenses": [first, inner], "z_l": values["root_z"]},
    )

    expected = (
        ("lens_0_Rein", values["first_rein"]),
        ("lens_0_x0", values["first_x"]),
        ("lens_0_y0", values["first_y"]),
        ("lens_1_0_x0", values["nested_x"]),
        ("lens_1_0_y0", values["nested_y"]),
        ("lens_1_0_Rein", values["nested_rein"]),
        ("lens_1_1_kappa", values["mass_kappa"]),
        ("lens_1_1_x0", values["mass_x"]),
        ("lens_1_1_y0", values["mass_y"]),
        ("lens_z_l", values["root_z"]),
    )
    flattened = lens_system._lens_graph_inputs(root)

    assert tuple(name for name, _ in flattened) == tuple(name for name, _ in expected)
    assert all(actual is wanted for (_, actual), (_, wanted) in zip(flattened, expected, strict=True))
    assert "lens_cosmology" not in tuple(name for name, _ in flattened)
    assert "lens_lenses" not in tuple(name for name, _ in flattened)


def test_build_lens_system_owns_root_redshifts_names_staticization_and_adapter_values(
    recording_caustics_runtime,
    fixed_cosmology,
):
    """Realize one nested tree with node-owned redshifts and positional metadata."""
    events = recording_caustics_runtime.events
    first = caustics_models.CausticsLensSpec(
        "SIS",
        {"x0": -1.0, "y0": 0.25, "Rein": 1.5, "s": 0.0},
    )
    nested_atomic = caustics_models.CausticsLensSpec(
        "SIS",
        {"x0": 2.0, "y0": -0.5, "Rein": 0.75, "s": 0.1},
    )
    inner = caustics_models.CausticsLensSpec(
        "SinglePlane",
        {"lenses": (nested_atomic,), "name": "configured-inner"},
    )
    root = caustics_models.CausticsLensSpec(
        "SinglePlane",
        {"z_l": 0.45, "lenses": (first, inner)},
    )
    values = {
        "lens_z_l": 0.45,
        "lens_0_x0": -1.0,
        "lens_0_y0": 0.25,
        "lens_0_Rein": 1.5,
        "lens_0_s": 0.0,
        "lens_1_name": "configured-inner",
        "lens_1_0_x0": 2.0,
        "lens_1_0_y0": -0.5,
        "lens_1_0_Rein": 0.75,
        "lens_1_0_s": 0.1,
        "source_redshift": 1.8,
    }

    lens, adapter, adapter_values = lens_system._build_lens_system(
        root,
        cosmology=fixed_cosmology,
        values=values,
    )

    first_lens = lens.lenses[0]
    inner_lens = lens.lenses[1]
    nested_lens = inner_lens.lenses[0]
    assert lens.name == "lens"
    assert first_lens.name == "lens_0"
    assert inner_lens.name == "configured-inner"
    assert nested_lens.name == "lens_1_0"
    assert all(
        realized.cosmology is fixed_cosmology for realized in (lens, first_lens, inner_lens, nested_lens)
    )
    np.testing.assert_array_equal(lens.z_l.values, 0.45)
    np.testing.assert_array_equal(lens.z_s.values, 1.8)
    assert all("z_s" not in spec.parameters for spec in (root, first, inner, nested_atomic))
    assert first_lens.z_l is first_lens.z_s is None
    assert inner_lens.z_l is inner_lens.z_s is None
    assert nested_lens.z_l is nested_lens.z_s is None
    assert first_lens.static_calls == nested_lens.static_calls == 1
    for atomic_lens in (first_lens, nested_lens):
        for parameter in (atomic_lens.x0, atomic_lens.y0, atomic_lens.Rein):
            assert parameter.dtype_calls == [np.float64]
    assert events == [
        ("as_tensor", 1.8, np.float64),
        ("as_tensor", 0.45, np.float64),
        ("parameter_to", "lens_0", "x0", np.float64),
        ("parameter_to", "lens_0", "y0", np.float64),
        ("parameter_to", "lens_0", "Rein", np.float64),
        ("to_static", "lens_0"),
        ("parameter_to", "lens_1_0", "x0", np.float64),
        ("parameter_to", "lens_1_0", "y0", np.float64),
        ("parameter_to", "lens_1_0", "Rein", np.float64),
        ("to_static", "lens_1_0"),
    ]

    assert isinstance(adapter, lens_system._SinglePlaneGeometryAdapter)
    assert tuple(component.name for component in adapter.components) == (
        "lens_0",
        "lens_1",
    )
    first_adapter = adapter.components[0].adapter
    nested_adapter = adapter.components[1].adapter
    assert tuple(component.name for component in nested_adapter.components) == ("lens_1_0",)
    assert first_adapter.jacobian_mask_points(adapter_values["lens_0"]) == ((-1.0, 0.25),)
    assert first_adapter.root_recovery_points(adapter_values["lens_0"]) == ((-1.0, 0.25),)
    assert first_adapter.pseudo_caustic_generators(adapter_values["lens_0"]) == (
        lens_system._PseudoCausticGenerator((-1.0, 0.25)),
    )
    nested_atomic_adapter = nested_adapter.components[0].adapter
    nested_atomic_values = adapter_values["lens_1"]["lens_1_0"]
    assert nested_atomic_adapter.jacobian_mask_points(nested_atomic_values) == ()
    assert nested_atomic_adapter.root_recovery_points(nested_atomic_values) == ()
    assert nested_atomic_adapter.pseudo_caustic_generators(nested_atomic_values) == ()
    assert tuple(adapter_values) == ("lens_0", "lens_1")
    assert tuple(adapter_values["lens_0"]) == ("x0", "y0", "Rein", "s")
    assert tuple(adapter_values["lens_1"]) == ("lens_1_0",)
    assert tuple(nested_atomic_values) == ("x0", "y0", "Rein", "s")
    assert dict(adapter_values["lens_0"]) == {
        "x0": -1.0,
        "y0": 0.25,
        "Rein": 1.5,
        "s": 0.0,
    }
    assert dict(adapter_values["lens_1"]["lens_1_0"]) == {
        "x0": 2.0,
        "y0": -0.5,
        "Rein": 0.75,
        "s": 0.1,
    }
    with pytest.raises(TypeError, match="does not support item assignment"):
        adapter_values["new"] = {}
    with pytest.raises(TypeError, match="does not support item assignment"):
        adapter_values["lens_0"]["x0"] = 99.0
    with pytest.raises(TypeError, match="does not support item assignment"):
        adapter_values["lens_1"]["new"] = {}
    with pytest.raises(TypeError, match="does not support item assignment"):
        nested_atomic_values["x0"] = 99.0


def test_build_lens_system_rejects_exact_non_affine_center_collision(
    recording_caustics_runtime,
    fixed_cosmology,
):
    """Name both generated components when realized non-affine centers collide."""
    del recording_caustics_runtime
    children = tuple(
        caustics_models.CausticsLensSpec(
            "SIS",
            {"x0": 0.0, "y0": 0.0, "Rein": radius},
        )
        for radius in (1.0, 2.0)
    )
    root = caustics_models.CausticsLensSpec(
        "SinglePlane",
        {"z_l": 0.5, "lenses": children},
    )
    values = {
        "lens_z_l": 0.5,
        "lens_0_x0": 1.25,
        "lens_0_y0": -0.75,
        "lens_0_Rein": 1.0,
        "lens_1_x0": 1.25,
        "lens_1_y0": -0.75,
        "lens_1_Rein": 2.0,
        "source_redshift": 2.0,
    }

    with pytest.raises(ValueError) as error:
        lens_system._build_lens_system(
            root,
            cosmology=fixed_cosmology,
            values=values,
        )

    message = str(error.value)
    assert "Non-affine lens components 'lens_0' and 'lens_1'" in message
    assert "exactly coincident centers" in message


def test_winding_number_reverses_sign_and_expected_counts_are_typed():
    """Apply signed true-caustic and pseudo-caustic image-count increments."""
    square = np.array(
        [
            [-1.0, -1.0],
            [1.0, -1.0],
            [1.0, 1.0],
            [-1.0, 1.0],
            [-1.0, -1.0],
        ]
    )
    adapter = lens_system._PointSingularityGeometryAdapter(
        center=(0.0, 0.0),
        resolution=1.0,
        extent=2.0,
        mask_center=True,
        recover_center=True,
        generate_pseudo_caustic=True,
    )

    assert adapter.winding_number(square, 0.0, 0.0) == 1
    assert adapter.winding_number(square[::-1], 0.0, 0.0) == -1
    assert adapter.winding_number(square, 3.0, 0.0) == 0
    assert (
        adapter.expected_num_images(
            0.0,
            0.0,
            values={},
            caustic_curves=(square,),
            pseudo_caustic_curves=(0.5 * square,),
        )
        == 4
    )
    assert (
        adapter.expected_num_images(
            0.0,
            0.0,
            values={},
            caustic_curves=(square[::-1],),
            pseudo_caustic_curves=(0.5 * square,),
        )
        == 0
    )


def test_point_singularity_geometry_adapter_exposes_atomic_capabilities():
    """Expose stored finite geometry and independently enabled point policies."""
    adapter = lens_system._PointSingularityGeometryAdapter(
        center=(1.5, -2.0),
        resolution=0.25,
        extent=6.0,
        mask_center=True,
        recover_center=False,
        generate_pseudo_caustic=True,
        axisymmetric=True,
    )

    assert adapter.search_center({}) == (1.5, -2.0)
    assert adapter.initial_fov({}) == 6.0
    assert adapter.resolution_scale({}) == 0.25
    assert adapter.jacobian_mask_points({}) == ((1.5, -2.0),)
    assert adapter.root_recovery_points({}) == ()
    assert adapter.pseudo_caustic_generators({}) == (lens_system._PseudoCausticGenerator((1.5, -2.0)),)
    assert adapter.axisymmetry_center({}) == (1.5, -2.0)
    assert adapter.preserves_axisymmetry({}) is False
    assert adapter.reference_num_images({}) == 1


def test_smooth_cusp_geometry_adapter_exposes_atomic_capabilities():
    """Expose smooth-cusp geometry without a pseudo-caustic generator."""
    adapter = lens_system._SmoothCuspGeometryAdapter(
        center=(-1.0, 3.0),
        resolution=0.4,
        extent=8.0,
        mask_center=False,
        recover_center=True,
        axisymmetric=False,
    )

    assert adapter.search_center({}) == (-1.0, 3.0)
    assert adapter.initial_fov({}) == 8.0
    assert adapter.resolution_scale({}) == 0.4
    assert adapter.jacobian_mask_points({}) == ()
    assert adapter.root_recovery_points({}) == ((-1.0, 3.0),)
    assert adapter.pseudo_caustic_generators({}) == ()
    assert adapter.axisymmetry_center({}) is None
    assert adapter.preserves_axisymmetry({}) is False
    assert adapter.reference_num_images({}) == 1


def test_registered_sie_factory_derives_extent_and_unsoftened_policy():
    """Map realized SIE tensors to ellipticity, scale, and singular policies."""
    lens = SimpleNamespace(
        x0=_RecordingParameter(1.0),
        y0=_RecordingParameter(-2.0),
        Rein=_RecordingParameter(2.0),
        q=_RecordingParameter(0.25),
        s=0.0,
    )
    registration = lens_system._LENS_MODEL_REGISTRY["SIE"]

    adapter = registration.geometry_factory(lens, {})

    assert registration.affine is False
    assert adapter.search_center({}) == (1.0, -2.0)
    assert adapter.resolution_scale({}) == 2.0
    assert adapter.initial_fov({}) == pytest.approx(8.8)
    assert adapter.jacobian_mask_points({}) == ((1.0, -2.0),)
    assert adapter.root_recovery_points({}) == ((1.0, -2.0),)
    assert adapter.pseudo_caustic_generators({}) == (lens_system._PseudoCausticGenerator((1.0, -2.0)),)
    assert adapter.axisymmetry_center({}) is None


@pytest.mark.parametrize(
    ("slope", "singular", "pseudo_caustic"),
    [(0.8, True, False), (1.0, True, True), (1.2, False, False)],
)
def test_registered_epl_factory_applies_slope_policy(
    slope,
    singular,
    pseudo_caustic,
):
    """Distinguish EPL center recovery from its exact isothermal pseudo-loop."""
    lens = SimpleNamespace(
        x0=_RecordingParameter(0.5),
        y0=_RecordingParameter(-0.25),
        Rein=_RecordingParameter(2.0),
        q=_RecordingParameter(1.0),
        t=_RecordingParameter(slope),
    )
    registration = lens_system._LENS_MODEL_REGISTRY["EPL"]

    adapter = registration.geometry_factory(lens, {})

    assert registration.affine is False
    assert adapter.initial_fov({}) == pytest.approx(4.4)
    assert bool(adapter.jacobian_mask_points({})) is singular
    assert bool(adapter.root_recovery_points({})) is singular
    assert bool(adapter.pseudo_caustic_generators({})) is pseudo_caustic
    assert adapter.axisymmetry_center({}) == (0.5, -0.25)


@pytest.mark.parametrize(
    ("softening", "singular", "pseudo_caustic"),
    [(0.0, True, False), (0.2, False, True)],
)
def test_registered_tnfw_factory_switches_softening_policy(
    softening,
    singular,
    pseudo_caustic,
):
    """Invert TNFW singular-center and pseudo-caustic capabilities at softening."""
    lens = SimpleNamespace(
        x0=_RecordingParameter(-1.0),
        y0=_RecordingParameter(3.0),
        Rs=_RecordingParameter(0.5),
        tau=_RecordingParameter(4.0),
        s=softening,
    )
    registration = lens_system._LENS_MODEL_REGISTRY["TNFW"]

    adapter = registration.geometry_factory(lens, {})

    assert registration.affine is False
    assert adapter.resolution_scale({}) == 0.5
    assert adapter.initial_fov({}) == pytest.approx(4.4)
    assert bool(adapter.jacobian_mask_points({})) is singular
    assert bool(adapter.root_recovery_points({})) is singular
    assert bool(adapter.pseudo_caustic_generators({})) is pseudo_caustic
    assert adapter.axisymmetry_center({}) == (-1.0, 3.0)


@pytest.mark.parametrize(
    ("gamma_1", "gamma_2", "preserves_axisymmetry"),
    [(0.0, 0.0, True), (0.0, 0.1, False), (-0.2, 0.0, False)],
)
def test_registered_affine_factories_encode_exact_symmetry_policy(
    gamma_1,
    gamma_2,
    preserves_axisymmetry,
):
    """Require exactly zero shear while every mass sheet preserves symmetry."""
    shear_registration = lens_system._LENS_MODEL_REGISTRY["ExternalShear"]
    sheet_registration = lens_system._LENS_MODEL_REGISTRY["MassSheet"]
    shear_lens = SimpleNamespace(
        gamma_1=_RecordingParameter(gamma_1),
        gamma_2=_RecordingParameter(gamma_2),
    )

    shear = shear_registration.geometry_factory(shear_lens, {})
    sheet = sheet_registration.geometry_factory(object(), {})

    assert shear_registration.affine is sheet_registration.affine is True
    assert shear.preserves_axisymmetry({}) is preserves_axisymmetry
    assert sheet.preserves_axisymmetry({}) is True


def test_affine_geometry_adapter_exposes_only_center_and_symmetry_policy():
    """Exclude affine perturbations from finite search and recovery geometry."""
    adapter = lens_system._AffinePerturbationGeometryAdapter(
        axisymmetry_preserving=True,
    )
    values = {"x0": "2.5", "y0": -4}

    assert adapter.search_center(values) == (2.5, -4.0)
    assert adapter.initial_fov(values) is None
    assert adapter.resolution_scale(values) is None
    assert adapter.jacobian_mask_points(values) == ()
    assert adapter.root_recovery_points(values) == ()
    assert adapter.pseudo_caustic_generators(values) == ()
    assert adapter.axisymmetry_center(values) is None
    assert adapter.preserves_axisymmetry(values) is True
    assert adapter.reference_num_images(values) == 1


def test_single_plane_geometry_adapter_aggregates_ordered_component_geometry():
    """Aggregate non-affine envelopes, points, scales, and peer radius caps."""
    point = lens_system._PointSingularityGeometryAdapter(
        center=(0.0, 0.0),
        resolution=0.25,
        extent=2.0,
        mask_center=True,
        recover_center=True,
        generate_pseudo_caustic=True,
        axisymmetric=True,
    )
    affine = lens_system._AffinePerturbationGeometryAdapter(
        axisymmetry_preserving=True,
    )
    second = lens_system._PointSingularityGeometryAdapter(
        center=(4.0, 0.0),
        resolution=0.1,
        extent=4.0,
        mask_center=True,
        recover_center=True,
        generate_pseudo_caustic=True,
        axisymmetric=True,
    )
    plane = lens_system._SinglePlaneGeometryAdapter(
        (
            lens_system._GeometryComponent("point", point, False),
            lens_system._GeometryComponent("sheet", affine, True),
            lens_system._GeometryComponent("second", second, False),
        )
    )
    values = {
        "point": {},
        "sheet": {"x0": 100.0, "y0": -100.0},
        "second": {},
    }

    assert plane.search_center(values) == (2.5, 0.0)
    assert plane.initial_fov(values) == 7.0
    assert plane.resolution_scale(values) == 0.1
    assert plane.jacobian_mask_points(values) == ((0.0, 0.0), (4.0, 0.0))
    assert plane.root_recovery_points(values) == ((0.0, 0.0), (4.0, 0.0))
    assert plane.pseudo_caustic_generators(values) == (
        lens_system._PseudoCausticGenerator(
            center=(0.0, 0.0),
            max_initial_radius=1.0,
        ),
        lens_system._PseudoCausticGenerator(
            center=(4.0, 0.0),
            max_initial_radius=1.0,
        ),
    )
    assert plane.axisymmetry_center(values) is None
    assert plane.preserves_axisymmetry(values) is False
    assert plane.reference_num_images(values) == 1


def test_single_plane_geometry_adapter_certifies_only_preserved_axisymmetry():
    """Require one axisymmetric non-affine child and preserving affine peers."""
    point = lens_system._PointSingularityGeometryAdapter(
        center=(1.0, -1.0),
        resolution=0.2,
        extent=2.0,
        mask_center=False,
        recover_center=False,
        generate_pseudo_caustic=False,
        axisymmetric=True,
    )
    preserving = lens_system._AffinePerturbationGeometryAdapter(
        axisymmetry_preserving=True,
    )
    breaking = lens_system._AffinePerturbationGeometryAdapter(
        axisymmetry_preserving=False,
    )
    values = {
        "point": {},
        "affine": {"x0": 20.0, "y0": 30.0},
    }
    preserved_plane = lens_system._SinglePlaneGeometryAdapter(
        (
            lens_system._GeometryComponent("point", point, False),
            lens_system._GeometryComponent("affine", preserving, True),
        )
    )
    broken_plane = lens_system._SinglePlaneGeometryAdapter(
        (
            lens_system._GeometryComponent("point", point, False),
            lens_system._GeometryComponent("affine", breaking, True),
        )
    )
    affine_plane = lens_system._SinglePlaneGeometryAdapter(
        (lens_system._GeometryComponent("affine", preserving, True),)
    )

    assert preserved_plane.axisymmetry_center(values) == (1.0, -1.0)
    assert broken_plane.axisymmetry_center(values) is None
    assert affine_plane.preserves_axisymmetry(values) is True


def test_single_plane_geometry_adapter_aggregates_reference_count_excesses():
    """Add each component's excess over the shared single-plane image."""
    first = _ValueCountPointGeometryAdapter(
        center=(-1.0, 0.0),
        resolution=0.2,
        extent=2.0,
        mask_center=False,
        recover_center=False,
        generate_pseudo_caustic=False,
    )
    second = _ValueCountPointGeometryAdapter(
        center=(1.0, 0.0),
        resolution=0.1,
        extent=2.0,
        mask_center=False,
        recover_center=False,
        generate_pseudo_caustic=False,
    )
    affine = lens_system._AffinePerturbationGeometryAdapter()
    plane = lens_system._SinglePlaneGeometryAdapter(
        (
            lens_system._GeometryComponent("first", first, False),
            lens_system._GeometryComponent("affine", affine, True),
            lens_system._GeometryComponent("second", second, False),
        )
    )
    values = {
        "first": {"reference_count": 3},
        "affine": {"x0": 0.0, "y0": 0.0},
        "second": {"reference_count": 2},
    }

    assert plane.reference_num_images(values) == 4

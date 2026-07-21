from types import SimpleNamespace

import numpy as np
import pytest

from lightcurvelynx.models import caustics_models


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


class _ValueCountPointGeometryAdapter(caustics_models._PointSingularityGeometryAdapter):
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


class _FakeTorch:
    """Provide float64 tensor conversion for recovery certification."""

    float64 = np.float64

    @staticmethod
    def as_tensor(values, dtype=None):
        """Convert values to the minimal tensor double."""
        del dtype
        return _FakeTensor(values)


class _MappedRecoveryLens:
    """Return predetermined source positions for candidate recovery roots."""

    def __init__(self, mapped_x, mapped_y):
        self.mapped_x = mapped_x
        self.mapped_y = mapped_y

    def raytrace(self, image_x, image_y):
        """Return the configured mapped positions in candidate order."""
        del image_x, image_y
        return _FakeTensor(self.mapped_x), _FakeTensor(self.mapped_y)


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


def test_root_validation_requires_redshift_and_rejects_nested_redshift(fake_caustics_registry):
    """Enforce root ownership of the shared lens-plane redshift."""
    del fake_caustics_registry
    missing_root = caustics_models.CausticsLensSpec(
        "SIS",
        {"x0": 0.0, "y0": 0.0, "Rein": 1.0},
    )
    with pytest.raises(ValueError, match="root lens specification requires: z_l"):
        caustics_models._validate_root_lens_spec(missing_root)

    child = caustics_models.CausticsLensSpec(
        "SIS",
        {"z_l": 0.4, "x0": 0.0, "y0": 0.0, "Rein": 1.0},
    )
    root = caustics_models.CausticsLensSpec(
        "SinglePlane",
        {"z_l": 0.4, "lenses": [child]},
    )
    with pytest.raises(ValueError, match="inherit rather than define: z_l"):
        caustics_models._validate_root_lens_spec(root)


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
            caustics_models._validate_root_lens_spec(affine_lens)


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

    assert caustics_models._validate_root_lens_spec(atomic) is None
    assert caustics_models._validate_root_lens_spec(mixed) is None
    assert caustics_models._validate_root_lens_spec(nested) is None


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
    flattened = caustics_models._lens_graph_inputs(root)

    assert tuple(name for name, _ in flattened) == tuple(name for name, _ in expected)
    assert all(actual is wanted for (_, actual), (_, wanted) in zip(flattened, expected, strict=True))
    assert "lens_cosmology" not in tuple(name for name, _ in flattened)
    assert "lens_lenses" not in tuple(name for name, _ in flattened)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        (2, 2.0),
        (0.25, 0.25),
        (np.array(0.75), 0.75),
        ("0.5", 0.5),
    ],
)
def test_validate_optional_positive_fraction_accepts_scalar_values(value, expected):
    """Normalize every supported optional scalar representation."""
    result = caustics_models._validate_optional_positive_fraction("fraction", value)
    assert result == expected


@pytest.mark.parametrize("value", [np.array([0.5]), "not-a-number", object()])
def test_validate_optional_positive_fraction_rejects_non_scalars(value):
    """Reject arrays and objects that cannot represent one floating scalar."""
    with pytest.raises(TypeError, match="None or a scalar value convertible to float"):
        caustics_models._validate_optional_positive_fraction("fraction", value)


@pytest.mark.parametrize("value", [0.0, -0.1, np.nan, np.inf, -np.inf])
def test_validate_optional_positive_fraction_rejects_non_positive_or_non_finite(value):
    """Reject normalized fractions outside the finite positive domain."""
    with pytest.raises(ValueError, match="fraction must be finite and positive"):
        caustics_models._validate_optional_positive_fraction("fraction", value)


def test_sample_value_preserves_single_sample_and_indexes_multiple_samples():
    """Keep a single-system object intact and index only multi-sample values."""
    single = np.array([1.0, 2.0])
    first = object()
    second = object()

    assert caustics_models._sample_value(single, 0, 1) is single
    assert caustics_models._sample_value([first, second], 1, 2) is second
    np.testing.assert_array_equal(
        caustics_models._sample_value(np.array([[1.0, 2.0], [3.0, 4.0]]), 1, 2),
        [3.0, 4.0],
    )


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (RuntimeError("torch.linalg.solve: input matrix is singular"), True),
        (RuntimeError("LINALG.SOLVE failed with SINGULAR U"), True),
        (ValueError("linalg.solve: input matrix is singular"), False),
        (RuntimeError("input matrix is singular"), False),
        (RuntimeError("linalg.solve failed"), False),
    ],
)
def test_singular_forward_raytrace_error_classifier_is_exact(error, expected):
    """Recognize only the two documented singular linear-solve messages."""
    assert caustics_models._is_singular_forward_raytrace_error(error) is expected


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (IndexError("index 0 is out of bounds for dimension 0"), True),
        (IndexError("INDEX 0 IS OUT OF BOUNDS"), True),
        (RuntimeError("index 0 is out of bounds"), False),
        (IndexError("index 1 is out of bounds"), False),
        (IndexError("index 0 was outside bounds"), False),
    ],
)
def test_retryable_forward_raytrace_error_classifier_is_exact(error, expected):
    """Recognize only the documented empty-candidate IndexError."""
    assert caustics_models._is_retryable_forward_raytrace_error(error) is expected


def test_recovery_neighborhoods_handle_empty_and_occupied_inputs():
    """Mark recovery neighborhoods empty only when no image lies within radius."""
    recovery_points = np.array([[0.0, 0.0], [2.0, 0.0]])
    np.testing.assert_array_equal(
        caustics_models._recovery_neighborhoods_are_empty(
            np.empty((0, 2)),
            recovery_points,
            1.0,
        ),
        [True, True],
    )
    assert caustics_models._recovery_neighborhoods_are_empty(
        np.array([[0.0, 0.0]]),
        np.empty((0, 2)),
        1.0,
    ).shape == (0,)

    np.testing.assert_array_equal(
        caustics_models._recovery_neighborhoods_are_empty(
            np.array([[0.0, 0.0], [3.0, 0.0]]),
            np.array([[0.5, 0.0], [1.0, 0.0], [2.0, 0.0], [5.0, 0.0]]),
            1.0,
        ),
        [False, False, False, True],
    )


def test_validated_recovery_images_applies_strict_residual_and_inclusive_locality():
    """Retain ordered duplicates only when both recovery certifications pass."""
    lens = _MappedRecoveryLens(
        mapped_x=[0.09, 0.1, 0.05, 0.0, 0.05],
        mapped_y=[0.0, 0.0, 0.0, 0.0, 0.0],
    )
    image_x = _FakeTensor([0.0, 1.0, 2.0, 3.0, 2.0])
    image_y = _FakeTensor([0.0, 0.0, 0.0, 0.0, 0.0])
    recovery_points = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.5, 0.0],
            [3.5001, 0.0],
            [2.5, 0.0],
        ]
    )

    certified = caustics_models._validated_recovery_images(
        lens,
        _FakeTorch,
        image_x,
        image_y,
        recovery_points,
        _FakeTensor(0.0),
        _FakeTensor(0.0),
        epsilon=0.1,
        neighborhood_radius=0.5,
    )

    np.testing.assert_array_equal(
        certified,
        [[0.0, 0.0], [2.0, 0.0], [2.0, 0.0]],
    )


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
    adapter = caustics_models._PointSingularityGeometryAdapter(
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
    adapter = caustics_models._PointSingularityGeometryAdapter(
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
    assert adapter.pseudo_caustic_generators({}) == (caustics_models._PseudoCausticGenerator((1.5, -2.0)),)
    assert adapter.axisymmetry_center({}) == (1.5, -2.0)
    assert adapter.preserves_axisymmetry({}) is False
    assert adapter.reference_num_images({}) == 1


def test_smooth_cusp_geometry_adapter_exposes_atomic_capabilities():
    """Expose smooth-cusp geometry without a pseudo-caustic generator."""
    adapter = caustics_models._SmoothCuspGeometryAdapter(
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


def test_affine_geometry_adapter_exposes_only_center_and_symmetry_policy():
    """Exclude affine perturbations from finite search and recovery geometry."""
    adapter = caustics_models._AffinePerturbationGeometryAdapter(
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
    point = caustics_models._PointSingularityGeometryAdapter(
        center=(0.0, 0.0),
        resolution=0.25,
        extent=2.0,
        mask_center=True,
        recover_center=True,
        generate_pseudo_caustic=True,
        axisymmetric=True,
    )
    affine = caustics_models._AffinePerturbationGeometryAdapter(
        axisymmetry_preserving=True,
    )
    second = caustics_models._PointSingularityGeometryAdapter(
        center=(4.0, 0.0),
        resolution=0.1,
        extent=4.0,
        mask_center=True,
        recover_center=True,
        generate_pseudo_caustic=True,
        axisymmetric=True,
    )
    plane = caustics_models._SinglePlaneGeometryAdapter(
        (
            caustics_models._GeometryComponent("point", point, False),
            caustics_models._GeometryComponent("sheet", affine, True),
            caustics_models._GeometryComponent("second", second, False),
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
        caustics_models._PseudoCausticGenerator(
            center=(0.0, 0.0),
            max_initial_radius=1.0,
        ),
        caustics_models._PseudoCausticGenerator(
            center=(4.0, 0.0),
            max_initial_radius=1.0,
        ),
    )
    assert plane.axisymmetry_center(values) is None
    assert plane.preserves_axisymmetry(values) is False
    assert plane.reference_num_images(values) == 1


def test_single_plane_geometry_adapter_certifies_only_preserved_axisymmetry():
    """Require one axisymmetric non-affine child and preserving affine peers."""
    point = caustics_models._PointSingularityGeometryAdapter(
        center=(1.0, -1.0),
        resolution=0.2,
        extent=2.0,
        mask_center=False,
        recover_center=False,
        generate_pseudo_caustic=False,
        axisymmetric=True,
    )
    preserving = caustics_models._AffinePerturbationGeometryAdapter(
        axisymmetry_preserving=True,
    )
    breaking = caustics_models._AffinePerturbationGeometryAdapter(
        axisymmetry_preserving=False,
    )
    values = {
        "point": {},
        "affine": {"x0": 20.0, "y0": 30.0},
    }
    preserved_plane = caustics_models._SinglePlaneGeometryAdapter(
        (
            caustics_models._GeometryComponent("point", point, False),
            caustics_models._GeometryComponent("affine", preserving, True),
        )
    )
    broken_plane = caustics_models._SinglePlaneGeometryAdapter(
        (
            caustics_models._GeometryComponent("point", point, False),
            caustics_models._GeometryComponent("affine", breaking, True),
        )
    )
    affine_plane = caustics_models._SinglePlaneGeometryAdapter(
        (caustics_models._GeometryComponent("affine", preserving, True),)
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
    affine = caustics_models._AffinePerturbationGeometryAdapter()
    plane = caustics_models._SinglePlaneGeometryAdapter(
        (
            caustics_models._GeometryComponent("first", first, False),
            caustics_models._GeometryComponent("affine", affine, True),
            caustics_models._GeometryComponent("second", second, False),
        )
    )
    values = {
        "first": {"reference_count": 3},
        "affine": {"x0": 0.0, "y0": 0.0},
        "second": {"reference_count": 2},
    }

    assert plane.reference_num_images(values) == 4


def test_outer_grid_boundary_uses_rows_then_side_interiors():
    """Return every square-grid boundary value once in documented order."""
    grid = np.arange(9).reshape(3, 3)
    np.testing.assert_array_equal(
        caustics_models._outer_grid_boundary(grid),
        [0, 1, 2, 6, 7, 8, 3, 5],
    )


def test_close_curve_removes_near_duplicates_and_recloses_exactly():
    """Remove consecutive and cyclic near-duplicates before exact closure."""
    curve = np.array(
        [
            [0.0, 0.0],
            [0.0001, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.0, 0.0001],
            [0.0, 0.0],
        ]
    )
    expected = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ]
    )

    np.testing.assert_array_equal(
        caustics_models._close_curve(curve, tolerance=0.001),
        expected,
    )
    with pytest.raises(RuntimeError, match="fewer than three unique vertices"):
        caustics_models._close_curve(
            np.array([[0.0, 0.0], [0.0001, 0.0], [0.0, 0.0001], [0.0, 0.0]]),
            tolerance=0.001,
        )


def test_resample_closed_curve_returns_equal_arclength_square_points():
    """Sample a square at fixed half-edge arclength increments."""
    expected = np.array(
        [
            [-1.0, -1.0],
            [0.0, -1.0],
            [1.0, -1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [-1.0, 1.0],
            [-1.0, 0.0],
        ]
    )
    np.testing.assert_allclose(
        caustics_models._resample_closed_curve(_closed_square(), num_points=8),
        expected,
    )


def test_curve_orientation_detects_reversal_after_translation_and_cyclic_shift():
    """Ignore start point and translation while detecting traversal reversal."""
    reference = _closed_square()
    shifted_vertices = np.roll(reference[:-1], 2, axis=0) + np.array([3.0, -2.0])
    shifted = np.concatenate((shifted_vertices, shifted_vertices[:1]), axis=0)

    assert caustics_models._curve_orientation_is_preserved(reference, shifted) is True
    assert caustics_models._curve_orientation_is_preserved(reference, shifted[::-1]) is False


def test_curve_center_diameter_and_raw_hausdorff_use_vertex_geometry():
    """Measure bounding-box geometry and bidirectional nearest-vertex distance."""
    rectangle = np.array([[2.0, -1.0], [6.0, -1.0], [6.0, 2.0], [2.0, 2.0], [2.0, -1.0]])
    center, diameter = caustics_models._curve_center_and_diameter(rectangle)

    np.testing.assert_array_equal(center, [4.0, 0.5])
    assert diameter == 5.0
    assert caustics_models._raw_curve_hausdorff_distance(
        _closed_square(),
        _closed_square(center=(3.0, 0.0)),
    ) == pytest.approx(3.0)


def test_match_raw_caustic_curves_handles_empty_unequal_and_reordered_sets():
    """Return count sentinels and reorder equal sets by minimum displacement."""
    assert caustics_models._match_raw_caustic_curves((), ()) == ((), True)

    unmatched = _closed_square(center=(20.0, 0.0))
    candidates, counts_match = caustics_models._match_raw_caustic_curves(
        (_closed_square(),),
        (unmatched, _closed_square(center=(40.0, 0.0))),
    )
    assert candidates[0] is unmatched
    assert counts_match is False

    reference_first = _closed_square()
    reference_second = _closed_square(center=(10.0, 0.0))
    candidate_first = reference_first.copy()
    candidate_second = reference_second.copy()
    ordered, counts_match = caustics_models._match_raw_caustic_curves(
        (reference_first, reference_second),
        (candidate_second, candidate_first),
    )
    assert ordered[0] is candidate_first
    assert ordered[1] is candidate_second
    assert counts_match is True


def test_partition_axisymmetric_point_caustics_returns_count_mismatch_sentinel():
    """Return ``None`` when either consecutive curve count differs."""
    assert (
        caustics_models._partition_axisymmetric_point_caustics(
            (_closed_square(), _closed_square(center=(10.0, 0.0))),
            (_closed_square(),),
            (_closed_square(),),
            boundary_tolerance=0.1,
        )
        is None
    )


def test_partition_axisymmetric_point_caustics_extracts_contracting_point():
    """Classify a centered factor-of-two contraction as a point caustic."""
    older = _closed_square(half_width=4.0)
    previous = _closed_square(center=(0.05, 0.0), half_width=2.0)
    current = _closed_square(center=(0.08, 0.0), half_width=1.0)

    partition = caustics_models._partition_axisymmetric_point_caustics(
        (older,),
        (previous,),
        (current,),
        boundary_tolerance=0.1,
    )

    assert partition.previous_curves == ()
    assert partition.current_curves == ()
    assert len(partition.previous_points) == 1
    assert len(partition.current_points) == 1
    np.testing.assert_allclose(partition.previous_points[0], [0.05, 0.0])
    np.testing.assert_allclose(partition.current_points[0], [0.08, 0.0])


def test_partition_axisymmetric_point_caustics_retains_noncontracting_curve():
    """Keep a curve regular when its diameter does not contract fast enough."""
    older = _closed_square(half_width=1.0)
    previous = _closed_square(half_width=0.75)
    current = _closed_square(half_width=0.5)

    partition = caustics_models._partition_axisymmetric_point_caustics(
        (older,),
        (previous,),
        (current,),
        boundary_tolerance=0.1,
    )

    assert partition.previous_curves[0] is previous
    assert partition.current_curves[0] is current
    assert partition.previous_points == ()
    assert partition.current_points == ()


def test_boundary_regions_repairs_invalid_curves_and_rejects_zero_area():
    """Repair a bow-tie while rejecting a collinear non-area boundary."""
    pytest.importorskip("shapely")
    bow_tie = np.array([[0.0, 0.0], [2.0, 2.0], [0.0, 2.0], [2.0, 0.0], [0.0, 0.0]])

    (region,) = caustics_models._boundary_regions(
        (bow_tie,),
        geometry_tolerance=1.0e-6,
    )

    assert region.is_valid
    assert not region.is_empty
    assert np.isfinite(float(region.area))
    assert float(region.area) > 0.0

    collinear = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 0.0]])
    with pytest.raises(
        RuntimeError,
        match="Caustic boundary 0 did not enclose a finite positive-area polygonal region",
    ):
        caustics_models._boundary_regions(
            (collinear,),
            geometry_tolerance=1.0e-6,
        )


def test_match_boundary_curves_reports_assignment_displacement_counts_and_orientation():
    """Match geometry by distance while retaining typed count and direction flags."""
    pytest.importorskip("shapely")
    assert caustics_models._match_boundary_curves(
        (),
        (),
        geometry_tolerance=1.0e-6,
    ) == ((), 0.0, True, True)

    reference_first = _closed_square()
    reference_second = _closed_square(center=(10.0, 0.0))
    candidate_first = reference_first + np.array([0.1, 0.0])
    candidate_second = reference_second + np.array([0.0, 0.25])
    ordered, displacement, counts_stable, orientation_stable = caustics_models._match_boundary_curves(
        (reference_first, reference_second),
        (candidate_second, candidate_first),
        geometry_tolerance=1.0e-6,
    )

    assert ordered[0] is candidate_first
    assert ordered[1] is candidate_second
    assert displacement == pytest.approx(0.25)
    assert counts_stable is True
    assert orientation_stable is True

    reversed_ordered, reversed_displacement, counts_stable, orientation_stable = (
        caustics_models._match_boundary_curves(
            (reference_first,),
            (reference_first[::-1],),
            geometry_tolerance=1.0e-6,
        )
    )
    np.testing.assert_array_equal(reversed_ordered[0], reference_first[::-1])
    assert reversed_displacement == pytest.approx(0.0)
    assert counts_stable is True
    assert orientation_stable is False

    unmatched = _closed_square(center=(20.0, 0.0))
    candidates, displacement, counts_stable, orientation_stable = caustics_models._match_boundary_curves(
        (reference_first,),
        (candidate_first, unmatched),
        geometry_tolerance=1.0e-6,
    )
    assert candidates[0] is candidate_first
    assert candidates[1] is unmatched
    assert np.isinf(displacement)
    assert counts_stable is False
    assert orientation_stable is False


def test_boundary_topology_signature_is_stable_under_translation_and_point_changes():
    """Retain typed containment topology while ignoring positions and points."""
    pytest.importorskip("shapely")
    true_curve = _closed_square(half_width=2.0)
    pseudo_curve = _closed_square(half_width=0.5)
    geometry = caustics_models._BoundaryGeometry(
        caustic_curves=(true_curve,),
        pseudo_caustic_curves=(pseudo_curve,),
        critical_curve_fov=4.0,
        pixelscale=0.1,
        pseudo_caustic_points=64,
        point_caustics=(np.array([0.0, 0.0]),),
    )
    translated = caustics_models._BoundaryGeometry(
        caustic_curves=(true_curve + np.array([7.0, -3.0]),),
        pseudo_caustic_curves=(pseudo_curve + np.array([7.0, -3.0]),),
        critical_curve_fov=8.0,
        pixelscale=0.05,
        pseudo_caustic_points=128,
        point_caustics=(np.array([100.0, 100.0]),),
    )

    signature = caustics_models._boundary_topology_signature(
        geometry,
        geometry_tolerance=1.0e-6,
    )

    assert signature == (
        (((1, 0),), ((1, 0),)),
        ((False, False, True, False, False),),
    )
    assert signature == caustics_models._boundary_topology_signature(
        translated,
        geometry_tolerance=1.0e-6,
    )


def test_compare_boundary_geometry_reorders_stable_topology_and_preserves_metadata():
    """Reorder matched curves while forwarding the current snapshot metadata."""
    pytest.importorskip("shapely")
    reference_first = _closed_square()
    reference_second = _closed_square(center=(10.0, 0.0))
    candidate_first = reference_first + np.array([0.1, 0.0])
    candidate_second = reference_second + np.array([0.0, 0.25])
    previous = caustics_models._BoundaryGeometry(
        caustic_curves=(reference_first, reference_second),
        pseudo_caustic_curves=(),
        critical_curve_fov=4.0,
        pixelscale=0.1,
        pseudo_caustic_points=64,
    )
    point_caustic = np.array([0.0, 0.0])
    current = caustics_models._BoundaryGeometry(
        caustic_curves=(candidate_second, candidate_first),
        pseudo_caustic_curves=(),
        critical_curve_fov=8.0,
        pixelscale=0.05,
        pseudo_caustic_points=128,
        point_caustics=(point_caustic,),
    )

    reordered, displacement, topology_stable = caustics_models._compare_boundary_geometry(
        previous,
        current,
        1.0e-6,
    )

    assert reordered.caustic_curves[0] is candidate_first
    assert reordered.caustic_curves[1] is candidate_second
    assert reordered.critical_curve_fov == 8.0
    assert reordered.pixelscale == 0.05
    assert reordered.pseudo_caustic_points == 128
    assert reordered.point_caustics[0] is point_caustic
    assert displacement == pytest.approx(0.25)
    assert topology_stable is True


def test_build_strong_lensing_region_unions_boundaries_and_rejects_empty_input():
    """Union overlapping typed interiors into one finite positive-area region."""
    pytest.importorskip("shapely")
    left = _closed_square(center=(1.0, 1.0), half_width=1.0)
    right = _closed_square(center=(2.0, 1.0), half_width=1.0)

    region = caustics_models._build_strong_lensing_region(
        (left,),
        (right,),
        geometry_tolerance=1.0e-6,
    )

    assert region.is_valid
    assert tuple(float(value) for value in region.bounds) == (0.0, 0.0, 3.0, 2.0)
    assert float(region.area) == pytest.approx(6.0)

    with pytest.raises(RuntimeError, match="strong-lensing region has no finite positive area"):
        caustics_models._build_strong_lensing_region(
            (),
            (),
            geometry_tolerance=1.0e-6,
        )


def test_source_boundary_clearance_uses_regular_curves_and_ignores_point_caustics():
    """Measure to the nearest line without assigning radius to point caustics."""
    pytest.importorskip("shapely")
    geometry = caustics_models._BoundaryGeometry(
        caustic_curves=(_closed_square(),),
        pseudo_caustic_curves=(_closed_square(center=(10.0, 0.0)),),
        critical_curve_fov=4.0,
        pixelscale=0.1,
        pseudo_caustic_points=64,
        point_caustics=(np.array([0.0, 0.0]),),
    )

    assert caustics_models._source_boundary_clearance(
        0.0,
        0.0,
        geometry,
        1.0e-6,
    ) == pytest.approx(1.0)


def test_sample_position_rejects_outside_draws_with_fixed_rng():
    """Return the third fixed draw pair after two triangle rejections."""
    shapely = pytest.importorskip("shapely")
    region = shapely.Polygon([(0.0, 0.0), (2.0, 0.0), (0.0, 2.0)])

    source_x, source_y, area, attempts = caustics_models._sample_position(
        region,
        np.random.default_rng(1),
        max_attempts=4,
        lens_identifier="lens_0",
        geometry_settings={"pixelscale": 0.1},
    )

    assert source_x == pytest.approx(0.6236629040209709)
    assert source_y == pytest.approx(0.8466528979451513)
    assert area == pytest.approx(2.0)
    assert attempts == 3


def test_sample_position_rejects_only_exact_excluded_points():
    """Reject an exactly sampled point while accepting a nearby exclusion."""
    shapely = pytest.importorskip("shapely")
    region = shapely.box(0.0, 0.0, 1.0, 1.0)
    first_draw = np.array([0.2616121342493164, 0.2984911434141233])

    source_x, source_y, area, attempts = caustics_models._sample_position(
        region,
        np.random.default_rng(2),
        max_attempts=2,
        lens_identifier="lens_1",
        geometry_settings={"pixelscale": 0.1},
        excluded_points=(first_draw,),
    )

    assert source_x == pytest.approx(0.8142257405942803)
    assert source_y == pytest.approx(0.0919159421350969)
    assert area == pytest.approx(1.0)
    assert attempts == 2

    nearby = np.array([np.nextafter(first_draw[0], np.inf), first_draw[1]])
    source_x, source_y, _, attempts = caustics_models._sample_position(
        region,
        np.random.default_rng(2),
        max_attempts=1,
        lens_identifier="lens_1",
        geometry_settings={"pixelscale": 0.1},
        excluded_points=(nearby,),
    )
    assert source_x == first_draw[0]
    assert source_y == first_draw[1]
    assert attempts == 1


def test_sample_position_reports_bounded_attempt_exhaustion_context():
    """Report lens, area, attempt, and geometry context after bounded failure."""
    shapely = pytest.importorskip("shapely")
    region = shapely.box(0.0, 0.0, 1.0, 1.0)
    preview = np.random.default_rng(3)
    excluded_points = tuple(
        np.array([preview.uniform(0.0, 1.0), preview.uniform(0.0, 1.0)]) for _ in range(2)
    )

    with pytest.raises(RuntimeError) as error:
        caustics_models._sample_position(
            region,
            np.random.default_rng(3),
            max_attempts=2,
            lens_identifier="lens[3]",
            geometry_settings={"pixelscale": 0.1, "fov": 4.0},
            excluded_points=excluded_points,
        )

    message = str(error.value)
    assert "Unable to sample the strong-lensing region for lens[3] after 2 attempts" in message
    assert "bounding-box area=1.0 arcsec^2" in message
    assert "polygon area=1.0 arcsec^2" in message
    assert "pixelscale=0.1" in message
    assert "fov=4.0" in message

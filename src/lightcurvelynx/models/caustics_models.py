"""Caustics-backed nodes for strong-lens image configurations."""

from collections.abc import Mapping

import numpy as np
from citation_compass import CiteClass

from lightcurvelynx.base_models import FunctionNode

_RESERVED_LENS_PARAMETERS = {
    "cosmology",
    "name",
    "z_l",
    "z_s",
}


def _validate_lens_configuration(lens_model, lens_parameters):
    """Validate configuration shared by Caustics-backed lens nodes."""
    if not isinstance(lens_model, str) or not lens_model:
        raise TypeError("lens_model must be a non-empty Caustics class name.")
    if not isinstance(lens_parameters, Mapping):
        raise TypeError("lens_parameters must be a mapping.")

    collisions = _RESERVED_LENS_PARAMETERS.intersection(lens_parameters)
    if collisions:
        names = ", ".join(sorted(collisions))
        raise ValueError(f"Reserved lens parameter name(s): {names}.")


def _import_caustics_dependencies():
    """Lazily import the optional Caustics runtime dependencies."""
    try:
        import caustics
        import torch
    except ImportError as err:  # pragma: no cover
        raise ImportError(
            "Caustics-backed lens nodes require the optional 'caustics' package. "
            "Install it with `pip install caustics`."
        ) from err
    return caustics, torch


def _to_numpy(tensor):
    """Detach a backend tensor and return a CPU float array."""
    return tensor.detach().cpu().numpy().astype(float, copy=False)


def _sample_value(value, sample_index, num_samples):
    """Extract one realized value from a scalar or sample-first array."""
    if num_samples == 1:
        return value
    return value[sample_index]


def _validate_lens_redshifts(values):
    """Return validated lens and source redshifts for one realization."""
    z_l = float(values["lens_redshift"])
    z_s = float(values["source_redshift"])
    if not np.isfinite(z_l) or not np.isfinite(z_s):
        raise ValueError("Lens and source redshifts must be finite.")
    if z_l < 0.0 or z_s <= z_l:
        raise ValueError(
            f"Expected 0 <= lens_redshift < source_redshift; got {z_l} and {z_s}."
        )
    return z_l, z_s


def _construct_caustics_lens(
    *,
    lens_model,
    cosmology,
    values,
    lens_parameter_names,
):
    """Construct one Caustics lens from realized numeric graph values."""
    caustics, torch = _import_caustics_dependencies()
    z_l, z_s = _validate_lens_redshifts(values)

    # TODO: Currently only supports base lens classes in Caustics. Implement a
    # helper for compound lens configurations such as SIE plus external shear.
    try:
        lens_class = getattr(caustics, lens_model)
    except AttributeError as err:
        raise ValueError(f"Unknown Caustics lens model '{lens_model}'.") from err

    dtype = torch.float64
    lens_kwargs = {
        name: torch.as_tensor(values[f"lens_{name}"], dtype=dtype)
        for name in lens_parameter_names
    }
    lens = lens_class(
        name="lens",
        cosmology=cosmology,
        z_l=torch.as_tensor(z_l, dtype=dtype),
        z_s=torch.as_tensor(z_s, dtype=dtype),
        **lens_kwargs,
    )
    return lens, torch


class CausticsLensImageNode(FunctionNode, CiteClass):
    """Compute point-source macro-images with the optional Caustics package.

    References
    ----------
    * Caustics - https://github.com/Ciela-Institute/caustics
    """

    _OUTPUTS = [
        "num_images",
        "image_x",
        "image_y",
        "macro_magnifications",
        "time_delays",
    ]

    def __init__(
        self,
        lens_model,
        *,
        cosmology,
        lens_redshift,
        source_redshift,
        source_x,
        source_y,
        lens_parameters,
        max_images,
        min_images=2,
        fov=5.0,
        divisions=100,
        epsilon=1.0e-3,
        max_depth=25,
        residual_tolerance=1.0e-4,
        node_label=None,
    ):
        _validate_lens_configuration(lens_model, lens_parameters)
        if not isinstance(max_images, int) or max_images < 2:
            raise ValueError("max_images must be an integer greater than one.")
        if not isinstance(min_images, int) or not 1 <= min_images <= max_images:
            raise ValueError("min_images must be between one and max_images.")
        if fov <= 0.0 or divisions < 2 or epsilon <= 0.0 or max_depth < 1:
            raise ValueError("Invalid forward-raytrace solver configuration.")
        if residual_tolerance <= 0.0:
            raise ValueError("residual_tolerance must be positive.")

        self.lens_model = lens_model
        self.cosmology = cosmology
        self.max_images = max_images
        self.min_images = min_images
        self.fov = float(fov)
        self.divisions = int(divisions)
        self.epsilon = float(epsilon)
        self.max_depth = int(max_depth)
        self.residual_tolerance = float(residual_tolerance)
        self._lens_parameter_names = tuple(lens_parameters)

        # Register every lens parameter independently so AttributeIndicatorNode
        # dependencies inside the mapping remain visible to the graph.
        node_inputs = {
            "lens_redshift": lens_redshift,
            "source_redshift": source_redshift,
            "source_x": source_x,
            "source_y": source_y,
        }
        for name, setter in lens_parameters.items():
            node_inputs[f"lens_{name}"] = setter

        super().__init__(
            self._non_func,
            node_label=node_label,
            outputs=self._OUTPUTS,
            **node_inputs,
        )

    def _solve_one(self, values):
        lens, torch = _construct_caustics_lens(
            lens_model=self.lens_model,
            cosmology=self.cosmology,
            values=values,
            lens_parameter_names=self._lens_parameter_names,
        )

        for method_name in (
            "forward_raytrace",
            "raytrace",
            "magnification",
            "time_delay",
        ):
            if not hasattr(lens, method_name):
                raise TypeError(
                    f"Caustics lens model '{self.lens_model}' does not implement "
                    f"required method '{method_name}'."
                )

        beta_x = torch.as_tensor(values["source_x"], dtype=torch.float64)
        beta_y = torch.as_tensor(values["source_y"], dtype=torch.float64)
        image_x, image_y = lens.forward_raytrace(
            beta_x,
            beta_y,
            epsilon=self.epsilon,
            fov=self.fov,
            divisions=self.divisions,
            max_depth=self.max_depth,
        )

        # Validate that every returned image maps back to the requested source.
        traced_x, traced_y = lens.raytrace(image_x, image_y)
        residual = torch.sqrt(
            (traced_x - beta_x) ** 2 + (traced_y - beta_y) ** 2
        )
        if len(residual) == 0 or bool(torch.any(residual > self.residual_tolerance)):
            max_residual = float(torch.max(residual)) if len(residual) else np.inf
            raise RuntimeError(
                "Caustics returned an invalid image solution; maximum source-plane "
                f"residual was {max_residual} arcsec."
            )

        magnifications = torch.abs(lens.magnification(image_x, image_y))
        time_delays = lens.time_delay(image_x, image_y)

        image_x = _to_numpy(image_x)
        image_y = _to_numpy(image_y)
        magnifications = _to_numpy(magnifications)
        time_delays = _to_numpy(time_delays)

        num_images = len(image_x)
        if num_images < self.min_images:
            raise RuntimeError(
                f"Expected at least {self.min_images} images, found {num_images}."
            )
        if num_images > self.max_images:
            raise RuntimeError(
                f"Found {num_images} images, exceeding max_images={self.max_images}."
            )
        if not all(
            np.all(np.isfinite(values_array))
            for values_array in (
                image_x,
                image_y,
                magnifications,
                time_delays,
            )
        ):
            raise RuntimeError("Caustics returned non-finite active image values.")

        time_delays = time_delays - np.min(time_delays)
        # np.lexsort uses the final key as the primary key: delay, then x, then y.
        order = np.lexsort((image_y, image_x, time_delays))
        return (
            image_x[order],
            image_y[order],
            magnifications[order],
            time_delays[order],
        )

    def compute(self, graph_state, rng_info=None, **kwargs):
        """Solve every sampled lens and save fixed-width numeric outputs."""
        del rng_info  # The solver is deterministic for realized input parameters.
        input_values = self._build_inputs(graph_state, **kwargs)
        num_samples = graph_state.num_samples

        counts = np.empty(num_samples, dtype=int)
        image_x = np.full((num_samples, self.max_images), np.nan)
        image_y = np.full((num_samples, self.max_images), np.nan)
        magnifications = np.zeros((num_samples, self.max_images), dtype=float)
        time_delays = np.full((num_samples, self.max_images), np.nan)

        for sample_index in range(num_samples):
            current_values = {
                name: _sample_value(value, sample_index, num_samples)
                for name, value in input_values.items()
            }
            current_x, current_y, current_mu, current_delay = self._solve_one(
                current_values
            )
            count = len(current_x)
            counts[sample_index] = count
            image_x[sample_index, :count] = current_x
            image_y[sample_index, :count] = current_y
            magnifications[sample_index, :count] = current_mu
            time_delays[sample_index, :count] = current_delay

        if num_samples == 1:
            results = [
                counts[0],
                image_x[0],
                image_y[0],
                magnifications[0],
                time_delays[0],
            ]
        else:
            results = [
                counts,
                image_x,
                image_y,
                magnifications,
                time_delays,
            ]

        self._save_results(results, graph_state)
        return results

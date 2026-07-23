import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord, SkyOffsetFrame
from scipy.optimize import least_squares

from lightcurvelynx.base_models import FunctionNode
from lightcurvelynx.effects.effect_model import EffectModel
from lightcurvelynx.graph_state import GraphState
from lightcurvelynx.models.physical_model import BasePhysicalModel


def _build_dependency_graph_without_mutation(source_model):
    """Inspect dependencies while preserving every reachable node identity."""
    reachable_nodes = []
    pending_nodes = [source_model]
    seen_nodes = set()
    while pending_nodes:
        node = pending_nodes.pop()
        if node in seen_nodes:
            continue
        seen_nodes.add(node)
        reachable_nodes.append(node)
        pending_nodes.extend(getattr(node, "objects", ()))
        pending_nodes.extend(
            setter.dependency for setter in node.setters.values() if setter.dependency is not None
        )

    snapshots = [
        (
            node,
            node.node_pos,
            node.node_string,
            tuple((setter, setter.node_name) for setter in node.setters.values()),
        )
        for node in reachable_nodes
    ]
    try:
        dependency_graph = source_model.build_dependency_graph()
        source_node_string = str(source_model)
    finally:
        for node, node_pos, node_string, setter_snapshots in snapshots:
            node.node_pos = node_pos
            node.node_string = node_string
            for setter, node_name in setter_snapshots:
                setter.node_name = node_name
    return dependency_graph, source_node_string


def _validate_source_for_resolved_lensing(source_model):
    if not isinstance(source_model, BasePhysicalModel):
        raise TypeError("source_model must be a BasePhysicalModel.")

    reserved = [
        name
        for name in ("base_ra", "base_dec", "base_t0", "macro_magnification")
        if name in source_model.setters
    ]
    if reserved:
        raise ValueError(
            "source_model is already decorated or uses reserved resolved-lens "
            f"parameters: {', '.join(reserved)}."
        )

    dependency_graph, source_node_string = _build_dependency_graph_without_mutation(source_model)
    for parameter_name in ("ra", "dec", "t0"):
        full_name = GraphState.extended_param_name(
            source_node_string,
            parameter_name,
        )
        dependents = dependency_graph.outgoing[full_name]
        if dependents:
            raise ValueError(
                "Cannot decorate source_model because parameter "
                f"{parameter_name} has dependent parameters: "
                f"{', '.join(sorted(dependents))}."
            )


def _coerce_scalar_samples(value, num_samples, name):
    try:
        values = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be float-coercible.") from exc
    if num_samples == 1:
        if values.ndim != 0:
            raise ValueError(f"{name} must be scalar for one sample.")
        return values.reshape(1)
    if values.shape != (num_samples,):
        raise ValueError(f"{name} must have shape ({num_samples},).")
    return values


def _coerce_image_samples(value, num_samples, name):
    try:
        values = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be float-coercible.") from exc
    if num_samples == 1:
        if values.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional for one sample.")
        return values[np.newaxis, :]
    if values.ndim != 2 or values.shape[0] != num_samples:
        raise ValueError(f"{name} must have shape ({num_samples}, M) for multiple samples.")
    return values


def _coerce_count_samples(value, num_samples):
    if value is None:
        return [None] * num_samples
    values = np.asarray(value, dtype=object)
    if num_samples == 1:
        if values.ndim != 0:
            raise ValueError("num_images must be scalar for one sample.")
        return [values.item()]
    if values.shape != (num_samples,):
        raise ValueError(f"num_images must have shape ({num_samples},).")
    return values.tolist()


def _active_count(raw_count, width):
    if raw_count is None:
        count = width
    else:
        if np.ndim(raw_count) != 0:
            raise ValueError("num_images must be scalar for one sample.")
        try:
            count = int(raw_count)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("num_images must be an integer.") from exc
        if count != raw_count:
            raise ValueError("num_images must be an integer.")
    if count < 2:
        raise ValueError("A resolved strong lens system must contain at least two images.")
    if count > width:
        raise ValueError(f"num_images={count} exceeds the image-array width {width}.")
    return count


def _normalized_image_rows(
    *,
    num_samples,
    source_t0,
    source_x,
    source_y,
    image_x,
    image_y,
    macro_magnifications,
    time_delays,
    num_images,
):
    source_t0_values = _coerce_scalar_samples(source_t0, num_samples, "source_t0")
    source_x_values = _coerce_scalar_samples(source_x, num_samples, "source_x")
    source_y_values = _coerce_scalar_samples(source_y, num_samples, "source_y")
    if not np.all(np.isfinite(source_t0_values)):
        raise ValueError("source_t0 must be finite.")
    if not np.all(np.isfinite(source_x_values)):
        raise ValueError("source_x must be finite.")
    if not np.all(np.isfinite(source_y_values)):
        raise ValueError("source_y must be finite.")

    arrays = {
        name: _coerce_image_samples(value, num_samples, name)
        for name, value in {
            "image_x": image_x,
            "image_y": image_y,
            "macro_magnifications": macro_magnifications,
            "time_delays": time_delays,
        }.items()
    }
    widths = {values.shape[1] for values in arrays.values()}
    if len(widths) != 1:
        raise ValueError("All resolved image arrays must have the same fixed width.")
    width = widths.pop()
    count_values = _coerce_count_samples(num_images, num_samples)

    rows = []
    for sample_index in range(num_samples):
        count = _active_count(count_values[sample_index], width)
        active = {name: values[sample_index, :count] for name, values in arrays.items()}
        if not np.all(np.isfinite(active["image_x"])):
            raise ValueError("Active image_x must be finite.")
        if not np.all(np.isfinite(active["image_y"])):
            raise ValueError("Active image_y must be finite.")
        magnifications = active["macro_magnifications"]
        if not np.all(np.isfinite(magnifications)):
            raise ValueError("Active macro_magnifications must be finite.")
        if np.any(magnifications < 0.0):
            raise ValueError("Active macro_magnifications must be non-negative.")
        if not np.any(magnifications > 0.0):
            raise ValueError("At least one macro_magnification must be positive.")
        delays = active["time_delays"]
        if not np.all(np.isfinite(delays)):
            raise ValueError("Active time_delays must be finite.")

        relative_delays = delays - np.min(delays)
        order = np.argsort(relative_delays, kind="stable")
        rows.append(
            {
                "image_x": active["image_x"][order],
                "image_y": active["image_y"][order],
                "macro_magnification": magnifications[order],
                "time_delay": relative_delays[order],
            }
        )
    return rows


class _ResolvedImageDataNode(FunctionNode):
    def __init__(
        self,
        *,
        source_t0,
        source_x,
        source_y,
        image_x,
        image_y,
        macro_magnifications,
        time_delays,
        num_images,
        **kwargs,
    ):
        super().__init__(
            self._non_func,
            outputs=["image_data"],
            source_t0=source_t0,
            source_x=source_x,
            source_y=source_y,
            image_x=image_x,
            image_y=image_y,
            macro_magnifications=macro_magnifications,
            time_delays=time_delays,
            num_images=num_images,
            **kwargs,
        )

    def compute(self, graph_state, rng_info=None, **kwargs):
        del rng_info
        rows = _normalized_image_rows(
            num_samples=graph_state.num_samples,
            **self._build_inputs(graph_state, **kwargs),
        )
        result = rows[0] if graph_state.num_samples == 1 else np.asarray(rows, dtype=object)
        self._save_results(result, graph_state)
        return result


_COORDINATE_TOLERANCE_ARCSEC = 1.0e-6


def _solve_lens_origin(source_ra, source_dec, source_x, source_y):
    source = SkyCoord(ra=source_ra * u.deg, dec=source_dec * u.deg, frame="icrs")
    initial = source.spherical_offsets_by(-source_x * u.arcsec, -source_y * u.arcsec)

    def residual(candidate):
        origin = SkyCoord(
            ra=candidate[0] * u.deg,
            dec=candidate[1] * u.deg,
            frame="icrs",
        )
        recovered = source.transform_to(SkyOffsetFrame(origin=origin))
        lon_error = (recovered.lon - source_x * u.arcsec).wrap_at(180.0 * u.deg)
        lat_error = recovered.lat - source_y * u.arcsec
        return np.array(
            [
                lon_error.to_value(u.arcsec),
                lat_error.to_value(u.arcsec),
            ]
        )

    solution = least_squares(
        residual,
        [initial.ra.deg, initial.dec.deg],
        bounds=([-np.inf, -90.0], [np.inf, 90.0]),
        xtol=1.0e-13,
        ftol=1.0e-13,
        gtol=1.0e-13,
        max_nfev=100,
    )
    error = np.max(np.abs(residual(solution.x)))
    if not solution.success or error > _COORDINATE_TOLERANCE_ARCSEC:
        raise ValueError(
            "Unable to derive a lens origin consistent with source-plane "
            f"coordinates; maximum residual is {error} arcsec."
        )
    return SkyCoord(
        ra=solution.x[0] * u.deg,
        dec=solution.x[1] * u.deg,
        frame="icrs",
    )


def _resolved_coordinates(
    *,
    num_samples,
    source_ra,
    source_dec,
    source_x,
    source_y,
    image_x,
    image_y,
):
    values = {
        name: _coerce_scalar_samples(value, num_samples, name)
        for name, value in {
            "source_ra": source_ra,
            "source_dec": source_dec,
            "source_x": source_x,
            "source_y": source_y,
            "image_x": image_x,
            "image_y": image_y,
        }.items()
    }
    if not np.all(np.isfinite(values["source_ra"])) or not np.all(np.isfinite(values["source_dec"])):
        raise ValueError("source RA/Dec must be finite.")

    outputs = {name: np.empty(num_samples) for name in ("lens_ra", "lens_dec", "ra_offset", "dec_offset")}
    for sample_index in range(num_samples):
        try:
            lens = _solve_lens_origin(
                values["source_ra"][sample_index],
                values["source_dec"][sample_index],
                values["source_x"][sample_index],
                values["source_y"][sample_index],
            )
            image = lens.spherical_offsets_by(
                values["image_x"][sample_index] * u.arcsec,
                values["image_y"][sample_index] * u.arcsec,
            )
        except Exception as exc:
            raise ValueError(
                f"Unable to transform resolved lens coordinates for sample {sample_index}."
            ) from exc
        outputs["lens_ra"][sample_index] = lens.ra.deg
        outputs["lens_dec"][sample_index] = lens.dec.deg
        outputs["ra_offset"][sample_index] = image.ra.deg - values["source_ra"][sample_index]
        outputs["dec_offset"][sample_index] = image.dec.deg - values["source_dec"][sample_index]

    ordered = [outputs[name] for name in ("lens_ra", "lens_dec", "ra_offset", "dec_offset")]
    if num_samples == 1:
        return [value[0] for value in ordered]
    return ordered


class _ResolvedCoordinatesNode(FunctionNode):
    def __init__(self, *, source_ra, source_dec, source_x, source_y, image_x, image_y, **kwargs):
        super().__init__(
            self._non_func,
            outputs=["lens_ra", "lens_dec", "ra_offset", "dec_offset"],
            source_ra=source_ra,
            source_dec=source_dec,
            source_x=source_x,
            source_y=source_y,
            image_x=image_x,
            image_y=image_y,
            **kwargs,
        )

    def compute(self, graph_state, rng_info=None, **kwargs):
        del rng_info
        results = _resolved_coordinates(
            num_samples=graph_state.num_samples,
            **self._build_inputs(graph_state, **kwargs),
        )
        self._save_results(results, graph_state)
        return results


class _MacroMagnificationEffect(EffectModel):
    def __init__(self, macro_magnification):
        super().__init__(rest_frame=False)
        self.add_effect_parameter("macro_magnification", macro_magnification)

    @staticmethod
    def _scale(values, macro_magnification):
        if macro_magnification is None:
            raise ValueError("macro_magnification must be provided.")
        return np.asarray(values) * macro_magnification

    def apply(
        self,
        flux_density,
        times=None,
        wavelengths=None,
        macro_magnification=None,
        **kwargs,
    ):
        del times, wavelengths, kwargs
        return self._scale(flux_density, macro_magnification)

    def apply_bandflux(
        self,
        bandfluxes,
        *,
        times=None,
        filters=None,
        macro_magnification=None,
        **kwargs,
    ):
        del times, filters, kwargs
        return self._scale(bandfluxes, macro_magnification)

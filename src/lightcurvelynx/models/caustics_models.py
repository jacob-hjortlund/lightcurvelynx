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
    """Validate constructor inputs shared by Caustics-backed lens nodes.

    Parameters
    ----------
    lens_model : str
        Name of a lens class exposed by the top-level ``caustics`` package.
    lens_parameters : Mapping
        Mapping from Caustics constructor parameter names to LightCurveLynx
        parameter setters. Values are not required to be numeric until graph
        sampling realizes them.

    Raises
    ------
    TypeError
        If ``lens_model`` is not a non-empty string or ``lens_parameters`` is
        not a mapping.
    ValueError
        If a lens parameter would collide with a constructor argument managed
        internally by the adapter.
    """
    if not isinstance(lens_model, str) or not lens_model:
        raise TypeError("lens_model must be a non-empty Caustics class name.")
    if not isinstance(lens_parameters, Mapping):
        raise TypeError("lens_parameters must be a mapping.")

    collisions = _RESERVED_LENS_PARAMETERS.intersection(lens_parameters)
    if collisions:
        names = ", ".join(sorted(collisions))
        raise ValueError(f"Reserved lens parameter name(s): {names}.")


def _import_caustics_dependencies():
    """Lazily import the optional Caustics runtime dependencies.

    Returns
    -------
    caustics : module
        Imported Caustics package.
    torch : module
        Imported PyTorch backend used by the installed Caustics package.

    Raises
    ------
    ImportError
        If either Caustics or its PyTorch backend cannot be imported. The
        original import error is retained as the exception cause.
    """
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
    """Convert a Caustics backend tensor to a CPU NumPy float array.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor returned by Caustics. It may reside on any Torch device or be
        attached to an autograd graph.

    Returns
    -------
    numpy.ndarray
        Detached CPU array with a floating dtype. No copy is made when the
        tensor's existing NumPy representation already has the requested dtype.
    """
    return tensor.detach().cpu().numpy().astype(float, copy=False)


def _sample_value(value, sample_index, num_samples):
    """Extract one lens realization from a sampled graph input.

    Parameters
    ----------
    value : object or array-like
        A scalar/single-system value when ``num_samples == 1`` or an object
        whose first axis indexes graph samples otherwise.
    sample_index : int
        Zero-based sample index to extract from a multi-sample value.
    num_samples : int
        Number of samples represented by the current ``GraphState``.

    Returns
    -------
    object
        ``value`` unchanged for a single-sample state, otherwise
        ``value[sample_index]``.
    """
    if num_samples == 1:
        return value
    return value[sample_index]


def _validate_lens_redshifts(values):
    """Validate redshifts from one realized Caustics-node input mapping.

    Parameters
    ----------
    values : Mapping
        Numeric inputs for one lens-system sample. The mapping must contain
        dimensionless ``lens_redshift`` and ``source_redshift`` entries.

    Returns
    -------
    lens_redshift : float
        Finite, non-negative lens redshift.
    source_redshift : float
        Finite source redshift strictly greater than ``lens_redshift``.

    Raises
    ------
    KeyError
        If either required redshift is absent.
    TypeError
        If a redshift cannot be converted to a scalar float.
    ValueError
        If a redshift is non-finite or the lens/source ordering is invalid.
    """
    z_l = float(values["lens_redshift"])
    z_s = float(values["source_redshift"])
    if not np.isfinite(z_l) or not np.isfinite(z_s):
        raise ValueError("Lens and source redshifts must be finite.")
    if z_l < 0.0 or z_s <= z_l:
        raise ValueError(f"Expected 0 <= lens_redshift < source_redshift; got {z_l} and {z_s}.")
    return z_l, z_s


def _construct_caustics_lens(
    *,
    lens_model,
    cosmology,
    values,
    lens_parameter_names,
):
    """Construct one Caustics lens from a single graph realization.

    Parameters
    ----------
    lens_model : str
        Name of the Caustics lens class to instantiate.
    cosmology : caustics.Cosmology
        Fixed cosmology object supplied to the Caustics lens constructor.
    values : Mapping
        Numeric inputs for one lens-system sample. It must contain
        ``lens_redshift``, ``source_redshift``, and one ``lens_<name>`` entry
        for every name in ``lens_parameter_names``.
    lens_parameter_names : iterable of str
        Caustics constructor parameter names without the ``lens_`` graph-input
        prefix.

    Returns
    -------
    lens : object
        Newly constructed Caustics lens for this realization. The object is not
        cached on a LightCurveLynx node.
    torch : module
        Imported PyTorch module, returned so callers can create compatible
        ``float64`` inputs without importing it eagerly.

    Raises
    ------
    ImportError
        If the optional Caustics runtime dependencies are unavailable.
    KeyError
        If a required realized input is missing from ``values``.
    ValueError
        If redshifts are invalid or ``lens_model`` is not exposed by Caustics.
    """
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
        name: torch.as_tensor(values[f"lens_{name}"], dtype=dtype) for name in lens_parameter_names
    }
    lens = lens_class(
        name="lens",
        cosmology=cosmology,
        z_l=torch.as_tensor(z_l, dtype=dtype),
        z_s=torch.as_tensor(z_s, dtype=dtype),
        **lens_kwargs,
    )
    return lens, torch


def _import_contourpy():
    """Lazily import the contour implementation used for critical curves.

    Returns
    -------
    contourpy : module
        Imported ContourPy package.

    Raises
    ------
    ImportError
        If ContourPy is unavailable. The original import error is retained as
        the exception cause.
    """
    try:
        import contourpy
    except ImportError as err:  # pragma: no cover
        raise ImportError(
            "Caustics source-position sampling requires the optional 'contourpy' "
            "package. Install it with `pip install contourpy`."
        ) from err
    return contourpy


def _lens_plane_origin(values):
    """Read the image-plane search origin from one realized input mapping.

    Parameters
    ----------
    values : Mapping
        Numeric inputs for one lens-system sample. ``lens_x0`` and ``lens_y0``
        are interpreted as angular offsets in arcseconds and each defaults to
        zero when absent.

    Returns
    -------
    x0 : float
        Image-plane x origin in arcseconds.
    y0 : float
        Image-plane y origin in arcseconds.

    Raises
    ------
    TypeError
        If either coordinate cannot be converted to a scalar float.
    ValueError
        If either coordinate is non-finite.
    """
    x0 = float(values.get("lens_x0", 0.0))
    y0 = float(values.get("lens_y0", 0.0))
    if not np.isfinite(x0) or not np.isfinite(y0):
        raise ValueError("The lens-plane origin must be finite.")
    return x0, y0


def _raytrace_curve(lens, coordinates):
    """Map one image-plane curve through the Caustics lens equation.

    Parameters
    ----------
    lens : object
        Realized Caustics lens implementing ``raytrace(x, y)``.
    coordinates : array-like, shape (N, 2)
        Image-plane x/y angular offsets in arcseconds.

    Returns
    -------
    numpy.ndarray, shape (N, 2)
        Source-plane x/y angular offsets in arcseconds, stored as finite CPU
        floating-point values.

    Raises
    ------
    ValueError
        If ``coordinates`` does not have shape ``(N, 2)``.
    RuntimeError
        If raytracing changes the expected shape or produces non-finite values.
    """
    _, torch = _import_caustics_dependencies()
    coordinates = np.asarray(coordinates, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("A lens-plane curve must have shape (N, 2).")

    source_x, source_y = lens.raytrace(
        torch.as_tensor(coordinates[:, 0], dtype=torch.float64),
        torch.as_tensor(coordinates[:, 1], dtype=torch.float64),
    )
    source_curve = np.column_stack((_to_numpy(source_x), _to_numpy(source_y)))
    if source_curve.shape != coordinates.shape or not np.all(np.isfinite(source_curve)):
        raise RuntimeError("Caustics returned an invalid source-plane boundary.")
    return source_curve


class _PointSingularityGeometryAdapter:
    """Enumerate pseudo-caustics for a lens with one point singularity.

    This private adapter describes the singular geometry shared by the Caustics
    SIE and SIS implementations. An unsoftened lens (``lens_s == 0``) has one
    singular point at ``(lens_x0, lens_y0)``. Its pseudo-caustic is obtained by
    raytracing successively smaller image-plane circles around that point until
    the source-plane boundary changes by no more than ``geometry_tolerance``.
    A softened lens (``lens_s > 0``) is continuous at its center and therefore
    contributes no pseudo-caustic through this adapter.

    Notes
    -----
    The adapter receives numeric values for a single graph sample. It never
    reads a ``GraphState`` directly and does not retain a realized lens or
    extracted boundary as mutable state.
    """

    _MAX_REFINEMENTS = 32

    @staticmethod
    def singular_points(values):
        """Return singular image-plane locations for one lens realization.

        Parameters
        ----------
        values : Mapping
            Numeric inputs for one lens-system sample. ``lens_s`` is the
            softening radius in arcseconds and defaults to zero. Unsoftened
            lenses must provide ``lens_x0`` and ``lens_y0`` in arcseconds.

        Returns
        -------
        tuple of tuple of float
            Empty for a softened lens, otherwise ``((x0, y0),)`` containing the
            singular image-plane location in arcseconds.

        Raises
        ------
        ValueError
            If the softening radius is negative/non-finite or an unsoftened
            realization omits either center coordinate.
        """
        softening = float(values.get("lens_s", 0.0))
        if not np.isfinite(softening) or softening < 0.0:
            raise ValueError("The lens softening radius must be finite and non-negative.")
        if softening > 0.0:
            return ()
        if "lens_x0" not in values or "lens_y0" not in values:
            raise ValueError("Point-singularity geometry requires lens_parameters entries for 'x0' and 'y0'.")
        return (_lens_plane_origin(values),)

    def pseudo_caustics(
        self,
        lens,
        values,
        *,
        num_points,
        epsilon,
        geometry_tolerance,
    ):
        """Trace every converged pseudo-caustic for one lens realization.

        Parameters
        ----------
        lens : object
            Realized Caustics lens implementing ``raytrace(x, y)``.
        values : Mapping
            Numeric inputs for the same single lens-system realization used to
            construct ``lens``. The adapter reads ``lens_s``, ``lens_x0``, and
            ``lens_y0`` to determine its singular geometry; other realized
            entries may remain in the mapping and are ignored here.
        num_points : int
            Number of unique, evenly spaced vertices on each image-plane loop.
        epsilon : float
            Initial loop radius around each singularity in arcseconds.
        geometry_tolerance : float
            Maximum allowed pointwise source-plane change, in arcseconds,
            between successive loop refinements.

        Returns
        -------
        list of numpy.ndarray
            One closed source-plane boundary per singularity. Each array has
            shape ``(num_points + 1, 2)`` in arcseconds, with the first vertex
            repeated at the end. A softened lens returns an empty list.

        Raises
        ------
        ValueError
            If the realized singularity configuration is invalid.
        RuntimeError
            If raytracing produces an invalid boundary or the shrinking-loop
            sequence does not converge within the bounded refinement count.
        """
        singular_points = self.singular_points(values)
        if not singular_points:
            return []

        angles = 2.0 * np.pi * np.arange(num_points, dtype=float) / num_points
        directions = np.column_stack((np.cos(angles), np.sin(angles)))
        pseudo_caustics = []

        for singular_x, singular_y in singular_points:
            center = np.array([singular_x, singular_y], dtype=float)
            radius = float(epsilon)
            previous_curve = _raytrace_curve(lens, center + radius * directions)
            last_change = np.inf

            for _ in range(self._MAX_REFINEMENTS):
                radius *= 0.5
                current_curve = _raytrace_curve(lens, center + radius * directions)
                last_change = float(np.max(np.linalg.norm(current_curve - previous_curve, axis=1)))
                if last_change <= geometry_tolerance:
                    pseudo_caustics.append(np.concatenate((current_curve, current_curve[:1]), axis=0))
                    break
                previous_curve = current_curve
            else:
                raise RuntimeError(
                    "Pseudo-caustic extraction did not converge after "
                    f"{self._MAX_REFINEMENTS} refinements; final boundary change "
                    f"was {last_change} arcsec."
                )

        return pseudo_caustics


_POINT_SINGULARITY_ADAPTER = _PointSingularityGeometryAdapter()
_LENS_GEOMETRY_ADAPTERS = {
    "SIE": _POINT_SINGULARITY_ADAPTER,
    "SIS": _POINT_SINGULARITY_ADAPTER,
}


def _get_lens_geometry_adapter(lens_model):
    """Look up the certified singular-geometry adapter for a lens class.

    Parameters
    ----------
    lens_model : str
        Name of the Caustics lens class used to construct the realized lens.

    Returns
    -------
    object
        Private adapter implementing ``singular_points`` and
        ``pseudo_caustics`` for every singular boundary of that lens class.

    Raises
    ------
    ValueError
        If no complete geometry adapter is registered. Explicit source
        positions may still use such a model through ``CausticsLensImageNode``.
    """
    try:
        return _LENS_GEOMETRY_ADAPTERS[lens_model]
    except KeyError as err:
        supported = ", ".join(sorted(_LENS_GEOMETRY_ADAPTERS))
        raise ValueError(
            "Source-position sampling has no complete pseudo-caustic geometry "
            f"adapter for Caustics lens model '{lens_model}'. Supported models: "
            f"{supported}."
        ) from err


def _find_all_caustics(
    lens,
    *,
    center,
    fov,
    pixelscale,
    geometry_tolerance,
    singular_points=(),
):
    """Locate all complete critical curves and map them into the source plane.

    The implementation uses the generic Caustics lens-equation Jacobian and
    raytrace protocols; it does not depend on a particular analytic lens class.

    Parameters
    ----------
    lens : object
        Realized Caustics lens implementing ``jacobian_lens_equation`` and
        ``raytrace``.
    center : tuple of float
        Image-plane x/y center of the square search grid in arcseconds.
    fov : float
        Width of the square image-plane search region in arcseconds.
    pixelscale : float
        Requested maximum grid spacing in arcseconds. The actual spacing evenly
        divides ``fov`` and is no larger than this value.
    geometry_tolerance : float
        Maximum allowed endpoint gap for a mapped source-plane caustic, in
        arcseconds.
    singular_points : iterable of tuple of float, optional
        Image-plane singular locations in arcseconds. One grid-spacing radius
        around each point is masked so a discontinuity is not misidentified as
        a true critical curve.

    Returns
    -------
    list of numpy.ndarray
        Closed source-plane caustic curves. Each element has shape ``(N, 2)``
        and contains x/y offsets in arcseconds. Disconnected critical curves
        remain separate list entries.

    Raises
    ------
    ImportError
        If Caustics, Torch, or ContourPy is unavailable.
    TypeError
        If the lens lacks the required protocol or returns a Jacobian with an
        unsupported representation.
    RuntimeError
        If the grid/Jacobian is invalid, a contour reaches the field-of-view
        boundary, a contour is malformed or open, or raytracing yields an
        invalid caustic.
    """
    contourpy = _import_contourpy()
    _, torch = _import_caustics_dependencies()

    for method_name in ("jacobian_lens_equation", "raytrace"):
        if not hasattr(lens, method_name):
            raise TypeError(f"Caustics source-position sampling requires lens method '{method_name}'.")

    center_x, center_y = (float(value) for value in center)
    num_intervals = int(np.ceil(fov / pixelscale))
    actual_pixelscale = float(fov) / num_intervals
    half_fov = 0.5 * float(fov)
    x_axis = torch.linspace(
        center_x - half_fov,
        center_x + half_fov,
        num_intervals + 1,
        dtype=torch.float64,
    )
    y_axis = torch.linspace(
        center_y - half_fov,
        center_y + half_fov,
        num_intervals + 1,
        dtype=torch.float64,
    )
    grid_y, grid_x = torch.meshgrid(y_axis, x_axis, indexing="ij")

    jacobian = lens.jacobian_lens_equation(grid_x, grid_y, method="autograd")
    if not isinstance(jacobian, torch.Tensor) or jacobian.shape[-2:] != (2, 2):
        raise TypeError("Caustics jacobian_lens_equation must return a tensor with trailing shape (2, 2).")
    determinant = _to_numpy(torch.linalg.det(jacobian))
    if determinant.shape != (num_intervals + 1, num_intervals + 1):
        raise RuntimeError("Caustics returned a lens-equation Jacobian with an unexpected grid shape.")

    x_coordinates = _to_numpy(x_axis)
    y_coordinates = _to_numpy(y_axis)
    invalid = ~np.isfinite(determinant)
    for singular_x, singular_y in singular_points:
        squared_distance = (x_coordinates[np.newaxis, :] - singular_x) ** 2 + (
            y_coordinates[:, np.newaxis] - singular_y
        ) ** 2
        invalid |= squared_distance <= actual_pixelscale**2

    if np.all(invalid):
        raise RuntimeError("The lens-equation Jacobian grid contains no finite values.")

    contour_generator = contourpy.contour_generator(
        x=x_coordinates,
        y=y_coordinates,
        z=np.ma.array(determinant, mask=invalid),
        line_type=contourpy.LineType.Separate,
    )
    critical_curves = contour_generator.lines(0.0)
    caustic_curves = []
    boundary_tolerance = max(
        actual_pixelscale * 1.0e-6,
        np.finfo(float).eps * max(1.0, abs(center_x), abs(center_y), fov) * 16.0,
    )

    for critical_curve in critical_curves:
        critical_curve = np.asarray(critical_curve, dtype=float)
        if critical_curve.ndim != 2 or critical_curve.shape[1] != 2:
            raise RuntimeError("Contour extraction returned an invalid critical curve.")
        if len(critical_curve) < 4:
            raise RuntimeError("A critical curve has fewer than three vertices.")
        if not np.all(np.isfinite(critical_curve)):
            raise RuntimeError("A critical curve contains non-finite coordinates.")

        reaches_boundary = (
            np.isclose(critical_curve[:, 0], x_coordinates[0], atol=boundary_tolerance, rtol=0.0)
            | np.isclose(critical_curve[:, 0], x_coordinates[-1], atol=boundary_tolerance, rtol=0.0)
            | np.isclose(critical_curve[:, 1], y_coordinates[0], atol=boundary_tolerance, rtol=0.0)
            | np.isclose(critical_curve[:, 1], y_coordinates[-1], atol=boundary_tolerance, rtol=0.0)
        )
        if np.any(reaches_boundary):
            raise RuntimeError(
                "A critical curve reaches the configured image-plane field-of-view boundary; increase fov."
            )

        critical_gap = float(np.linalg.norm(critical_curve[0] - critical_curve[-1]))
        if critical_gap > boundary_tolerance:
            raise RuntimeError(
                f"Contour extraction returned an open critical curve with endpoint gap {critical_gap} arcsec."
            )
        if critical_gap > 0.0:
            critical_curve = np.concatenate((critical_curve, critical_curve[:1]), axis=0)

        caustic_curve = _raytrace_curve(lens, critical_curve)
        caustic_gap = float(np.linalg.norm(caustic_curve[0] - caustic_curve[-1]))
        if caustic_gap > geometry_tolerance:
            raise RuntimeError(f"A mapped caustic is open with endpoint gap {caustic_gap} arcsec.")
        caustic_curves.append(caustic_curve)

    return caustic_curves


def _find_all_pseudo_caustics(
    lens,
    *,
    lens_model,
    values,
    num_points,
    epsilon,
    geometry_tolerance,
):
    """Extract all certified pseudo-caustics for one lens realization.

    Parameters
    ----------
    lens : object
        Realized Caustics lens used for source-plane raytracing.
    lens_model : str
        Caustics class name used to select a complete singular-geometry adapter.
    values : Mapping
        Numeric inputs for exactly one lens-system sample, normally produced by
        selecting one index from ``FunctionNode._build_inputs``. Keys retain
        their registered graph-input names: ``lens_redshift``,
        ``source_redshift``, and ``lens_<parameter>`` for each Caustics
        constructor parameter. The selected adapter reads the subset needed to
        enumerate singularities, such as ``lens_s``, ``lens_x0``, and
        ``lens_y0`` for SIE/SIS.
    num_points : int
        Number of unique vertices used to trace each singular boundary.
    epsilon : float
        Initial image-plane offset from a singular boundary in arcseconds.
    geometry_tolerance : float
        Required source-plane convergence tolerance in arcseconds.

    Returns
    -------
    list of numpy.ndarray
        Closed pseudo-caustic boundaries in source-plane arcseconds. Boundary
        arrays may have different vertex counts for different future adapters.

    Raises
    ------
    ValueError
        If ``lens_model`` has no complete geometry adapter or the realized
        singular geometry is invalid.
    RuntimeError
        If boundary raytracing is invalid or fails to converge.
    """
    adapter = _get_lens_geometry_adapter(lens_model)
    return adapter.pseudo_caustics(
        lens,
        values,
        num_points=num_points,
        epsilon=epsilon,
        geometry_tolerance=geometry_tolerance,
    )


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
        """Solve and validate the active macro-images for one lens system.

        Parameters
        ----------
        values : Mapping
            Numeric inputs for exactly one graph sample. Required entries are
            ``lens_redshift``, ``source_redshift``, source-plane ``source_x``
            and ``source_y`` in arcseconds, and ``lens_<parameter>`` for every
            configured Caustics constructor parameter.

        Returns
        -------
        image_x : numpy.ndarray, shape (I,)
            Active image-plane x positions in arcseconds.
        image_y : numpy.ndarray, shape (I,)
            Active image-plane y positions in arcseconds.
        macro_magnifications : numpy.ndarray, shape (I,)
            Finite, absolute dimensionless macro-magnifications.
        time_delays : numpy.ndarray, shape (I,)
            Observer-frame relative delays in days, normalized to start at zero.

        Notes
        -----
        All four arrays use the same deterministic ordering: increasing delay,
        then image x, then image y. Padding to ``max_images`` is performed by
        ``compute`` after this method returns.

        Raises
        ------
        ImportError
            If optional Caustics runtime dependencies are unavailable.
        TypeError
            If the selected lens lacks a required point-image method.
        ValueError
            If redshifts or the selected lens model are invalid.
        RuntimeError
            If root residuals, image counts, or active solver outputs fail
            validation.
        """
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
        residual = torch.sqrt((traced_x - beta_x) ** 2 + (traced_y - beta_y) ** 2)
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
            raise RuntimeError(f"Expected at least {self.min_images} images, found {num_images}.")
        if num_images > self.max_images:
            raise RuntimeError(f"Found {num_images} images, exceeding max_images={self.max_images}.")
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
                name: _sample_value(value, sample_index, num_samples) for name, value in input_values.items()
            }
            current_x, current_y, current_mu, current_delay = self._solve_one(current_values)
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

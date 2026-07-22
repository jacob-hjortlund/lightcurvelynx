import numpy as np


def _import_caustics():
    """Lazily import the optional Caustics package.

    Returns
    -------
    caustics : module
        Imported top-level Caustics package.

    Raises
    ------
    ImportError
        If Caustics itself or an import-time dependency is unavailable. The
        original ``ImportError`` is retained as the exception cause.
    """
    try:
        import caustics
    except ImportError as err:  # pragma: no cover
        raise ImportError(
            "Caustics-backed lens nodes require the optional 'caustics' package. "
            "Install it with `pip install caustics`."
        ) from err
    return caustics


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

    Notes
    -----
    A successful call sets Torch's process-wide default dtype to
    ``torch.float64``. The previous default is neither recorded nor restored.
    """
    caustics = _import_caustics()
    try:
        import torch

        torch.set_default_dtype(torch.float64)
    except ImportError as err:  # pragma: no cover
        raise ImportError(
            "Caustics-backed lens nodes require the optional 'caustics' package. "
            "Install it with `pip install caustics`."
        ) from err
    return caustics, torch


def _to_numpy(tensor):
    """Convert a Caustics backend tensor to a detached CPU NumPy array.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor returned by Caustics. It may reside on any Torch device or be
        attached to an autograd graph.

    Returns
    -------
    numpy.ndarray
        Detached CPU array with the same shape and dtype as the tensor's NumPy
        representation.

    Notes
    -----
    A CPU tensor's NumPy array can share its storage; moving a non-CPU tensor to
    the CPU necessarily allocates CPU storage. This helper does not request an
    independent copy after that device transfer.
    """
    return tensor.detach().cpu().numpy()


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

    Notes
    -----
    Single-sample extraction preserves exact object identity, including for an
    array-valued sample. Multi-sample indexing is not copied and may therefore
    return a scalar or a view according to ``value``'s indexing contract.
    """
    if num_samples == 1:
        return value
    return value[sample_index]


def _validate_optional_positive_fraction(name, value):
    """Normalize an optional positive dimensionless fraction.

    Parameters
    ----------
    name : str
        Public argument name used in contextual error messages.
    value : object or None
        Ordinary float-convertible scalar, including a zero-dimensional NumPy
        array, or ``None`` to disable relative scaling.

    Returns
    -------
    float or None
        Finite positive dimensionless fraction, or ``None`` unchanged.

    Raises
    ------
    TypeError
        If ``value`` is a non-scalar NumPy array or cannot be converted to a
        scalar float.
    ValueError
        If the normalized fraction is non-finite or not strictly positive.
    """
    if value is None:
        return None
    if isinstance(value, np.ndarray) and value.ndim != 0:
        raise TypeError(f"{name} must be None or a scalar value convertible to float.")
    try:
        normalized = float(value)
    except (TypeError, ValueError, OverflowError) as err:
        raise TypeError(f"{name} must be None or a scalar value convertible to float.") from err
    if not np.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return normalized


def _import_contourpy():
    """Lazily import the contour implementation used for critical curves.

    Returns
    -------
    contourpy : module
        Imported ContourPy package.

    Raises
    ------
    ImportError
        If ContourPy itself or an import-time dependency is unavailable. The
        original import error is retained as the exception cause.
    """
    try:
        import contourpy
    except ImportError as err:  # pragma: no cover
        raise ImportError(
            "Caustics source-position sampling requires the optional 'contourpy' "
            "package. Install it with `pip install contourpy`."
        ) from err
    return contourpy


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
        Source-plane x/y angular offsets in arcseconds on the CPU.

    Raises
    ------
    ImportError
        If the optional Caustics runtime dependencies are unavailable.

    Notes
    -----
    The production caller supplies a trusted ``(N, 2)`` curve. Caustics'
    paired output shapes, types, and finiteness are trusted. Coordinates are
    coerced through a floating NumPy array and float64 Torch tensors before the
    raytrace, and outputs are detached onto the CPU. Loading the runtime also
    sets Torch's process-wide default dtype to float64.
    """
    _, torch = _import_caustics_dependencies()
    coordinates = np.asarray(coordinates, dtype=float)

    source_x, source_y = lens.raytrace(
        torch.as_tensor(coordinates[:, 0], dtype=torch.float64),
        torch.as_tensor(coordinates[:, 1], dtype=torch.float64),
    )
    return np.column_stack((_to_numpy(source_x), _to_numpy(source_y)))


def _import_shapely():
    """Lazily import Shapely for source-plane topology operations.

    Returns
    -------
    shapely : module
        Imported Shapely package.

    Raises
    ------
    ImportError
        If Shapely is unavailable. The original import error is retained as the
        exception cause.
    """
    try:
        import shapely
    except ImportError as err:  # pragma: no cover
        raise ImportError(
            "Caustics source-position sampling requires the optional 'shapely' "
            "package. Install it with `pip install shapely`."
        ) from err
    return shapely

import importlib
import logging
from pathlib import Path

import numpy as np
import pooch
from astropy.coordinates import SkyCoord
from citation_compass import CiteClass

from lightcurvelynx import _LIGHTCURVELYNX_DOWNLOAD_DATA_DIR
from lightcurvelynx.base_models import FunctionNode


class StrongLens(FunctionNode):

    def __init__(self, model_name, source_redshift, source_ra, source_dec, **kwargs):

        # TODO: Add comment about potential issues with ra/decs and generalization with siblings
        try:
            import caustics
        except ImportError as err:
            raise ImportError(
                "caustics is not installed by default. To use the StrongLens class, "
                "please install caustics with `pip install caustics`"
            )

        try:
            lens_model = getattr(caustics, model_name)
        except:
            raise ValueError(f"Model {model_name} not found in caustics.")

        super().__init__(self._lens_model, **kwargs)

    def _lens_model(self, lens_params):
        pass

import torch
import logging
import caustics
import importlib
import numpy as np

from pathlib import Path
from citation_compass import CiteClass
from astropy.coordinates import SkyCoord
from lightcurvelynx.models.physical_model import SEDModel
from lightcurvelynx import _LIGHTCURVELYNX_DOWNLOAD_DATA_DIR

class StrongLensModel(SEDModel):

    def __init__(self, model_name, lens_redshift, source_redshift, cosmo, **kwargs):

        # TODO: Add comment about potential issues with ra/decs and generalization with siblings
        try:
            import caustics
            import torch
        except ImportError as err:
            raise ImportError(
                "caustics is not installed by default. To use the StrongLens class, "
                "please install caustics with `pip install caustics`"
            )

        try:
            lens_model = getattr(caustics, model_name)
        except:
            raise ValueError(f"Model {model_name} not found in caustics.")

        self.lens = lens_model(name="lens",
                               cosmology=cosmo,
                               z_l=lens_redshift,
                               z_s=source_redshift,
                               **kwargs)

        super().__init__(self._lens_model, **kwargs)

    def _lens_model(self, **kwargs):
        """
        Generate lens configuration by inverting the lens equation, then sampling source position from inside the caustic.
        """
        # TODO: add special cases for when we know the analytical cautics

        # create a grid in source plane
        
        n_pix = 100
        res = 0.05
        upsample_factor = 2
        fov = res * n_pix
        thx, thy = caustics.utils.meshgrid(
            res / upsample_factor,
            upsample_factor * n_pix,
            dtype=torch.float32,
        )

        # invert the lens equation, then ray trace to get caustics (lens plane)
        A = lens_model.jacobian_lens_equation(thx, thy)
        detA = torch.linalg.det(A)

        # transform back to the source plane, sample from within the caustics


        lp_grid_x, lp_grid_y = self.lens.forward_raytrace(sp_x, sp_y)

        self.image_x = lp_x # lens plane x position of images
        self.image_y = lp_y # lens plane y position of images
        self.macro_mag = self.lens.magnification(lp_x, lp_y)
        self.time_delays = self.lens.time_delay(lp_x, lp_y)

    def flux(self, t):

        # only for unresolved right now!
        flux = source.flux(t, )

        return flux

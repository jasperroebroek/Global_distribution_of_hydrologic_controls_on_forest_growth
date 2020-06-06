import thesis
import geomappy as mp
from geomappy.utils import progress_bar
import numpy as np
np.warnings.filterwarnings('ignore')

M_corr_wtd = mp.Raster("data/cmap_2d_significant/wtd_fapar_sign.tif")
M_corr_p_pet = mp.Raster("data/cmap_2d_significant/p_pet_fapar_sign.tif")

CLASSES = "data/cmap_2d_significant/classes_serial.tif"
CLASSES_FLOAT = "data/cmap_2d_significant/classes_serial_float.tif"
CLASSES_DOWNSAMPLED_10 = "data/cmap_2d_significant/classes_serial_downsampled_10.tif"

profile = M_corr_p_pet.profile
profile['dtype'] = np.int8
profile['nodata'] = -1
M_classes = mp.Raster(CLASSES, mode='w', profile=profile, overwrite=True)

profile['dtype'] = np.float64
M_classes_float = mp.Raster(CLASSES_FLOAT, mode='w', profile=profile, overwrite=True)

mp.Raster.set_tiles((4, 4))

classes = np.arange(9, dtype=np.int8).reshape((3, 3))
classes_float = np.arange(9, dtype=np.float64).reshape((3, 3))

for i in M_corr_wtd:
    progress_bar((i + 1) / M_corr_wtd.c_tiles)

    corr_wtd = M_corr_wtd[i] + 1
    corr_p_pet = M_corr_p_pet[i] + 1

    mask = np.logical_or(np.isnan(corr_wtd), np.isnan(corr_p_pet))

    corr_wtd[mask] = 0
    corr_p_pet[mask] = 0

    corr_wtd = corr_wtd.astype(int)
    corr_p_pet = corr_p_pet.astype(int)

    output = classes[corr_p_pet, corr_wtd]
    output[mask] = -1

    output_float = classes_float[corr_p_pet, corr_wtd]
    output_float[mask] = np.nan

    M_classes[i] = output
    M_classes_float[i] = output_float

mp.Raster.close()

M_classes = mp.Raster(CLASSES_FLOAT, tiles=(10, 10))
M_classes.focal_majority(output_file=CLASSES_DOWNSAMPLED_10, window_size=10, reduce=True, fraction_accepted=0, overwrite=True)

mp.Raster.close()

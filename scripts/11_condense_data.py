import thesis
import geomappy as mp
from geomappy.utils import progress_bar
import numpy as np
import pandas as pd

CORR_WTD_FAPAR = "data/correlation_wtd_fapar/correlation_wtd_fapar_ge3_15_downsampled_10.tif"
CORR_P_PET_FAPAR = "data/correlation_p_pet_fapar/correlation_p_pet_fapar_ge3_15_downsampled_10.tif"
CORR_P_PET_WTD = "data/correlation_p_pet_wtd/correlation_p_pet_wtd_15_downsampled_10.tif"
LANDSCAPE = "data/landscape_classes/landscape_downsampled_10.tif"
CLIMATE = "data/climate/climate_downsampled_10.tif"
CLASSES = "data/cmap_2d_significant/classes_serial_downsampled_10.tif"
P_PET = "data/pet/p_pet_downsampled_10.tif"
WTD = "data/wtd/wtd_downsampled_10.tif"
WTD_STD = "data/wtd/wtd_std_5_downsampled_10.tif"
TH = "data/tree_height/tree_height_global_downsampled_10.tif"
FAPAR = "data/fapar/mean_fpar_downsampled_10.tif"

M_corr_wtd_fapar = mp.Raster(CORR_WTD_FAPAR)
M_corr_p_pet_fapar = mp.Raster(CORR_P_PET_FAPAR)
M_corr_p_pet_wtd = mp.Raster(CORR_P_PET_WTD)
M_climate = mp.Raster(CLIMATE)
M_p_pet = mp.Raster(P_PET)
M_wtd = mp.Raster(WTD)
M_wtd_std = mp.Raster(WTD_STD)
M_landscape = mp.Raster(LANDSCAPE)
M_classes = mp.Raster(CLASSES, fill_value=-1)
M_fapar = mp.Raster(FAPAR)
M_th = mp.Raster(TH)

mp.Raster.set_tiles((10, 20))

df = None
for i in M_wtd:
    progress_bar((i + 1) / M_wtd.c_tiles)
    if (i + 1) // 50 == (i + 1) / 50:
        df.dropna().reset_index().to_feather(f'data/df/data_{i}.feather')
        df = None
    wtd = M_wtd[i].flatten()
    if (~np.isnan(wtd)).sum() == 0:
        # print(" empty array")
        continue
    p_pet = M_p_pet[i].flatten()
    if (~np.isnan(p_pet)).sum() == 0:
        # print(" empty array")
        continue

    temp_df = pd.DataFrame(data={'wtd': wtd,
                                 'wtd_std': M_wtd_std[i].flatten(),
                                 'p_pet': p_pet,
                                 'climate': M_climate[i].flatten(),
                                 'landscape': M_landscape[i].flatten(),
                                 'corr_wtd_fapar': M_corr_wtd_fapar[i].flatten(),
                                 'corr_p_pet_fapar': M_corr_p_pet_fapar[i].flatten(),
                                 'corr_p_pet_wtd': M_corr_p_pet_wtd[i].flatten(),
                                 'classes': M_classes[i].flatten(),
                                 'th': M_th[i].flatten(),
                                 'fapar': M_fapar[i].flatten()})

    if isinstance(df, type(None)):
        df = temp_df
    else:
        df = df.append(temp_df)

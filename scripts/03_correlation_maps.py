"""
Create correlation maps: WTD & FAPAR, P & FAPAR, WTD & P
"""
from thesis import *
import geomappy as mp
from geomappy.utils import progress_bar
import numpy as np

WTD = "data/wtd/wtd.tif"
P = "data/precipitation/precipitation.tif"
TH = "data/tree_height/tree_height_global.tif"
FAPAR = "data/fapar/mean_fpar_reprojected.tif"
T = "data/temperature/temperature.tif"
P_PET = "data/pet/p_pet.tif"

CORR_WTD_FAPAR_15 = "data/correlation_wtd_fapar/correlation_wtd_fapar_ge3_15.tif"
CORR_P_FAPAR_15 = "data/correlation_p_fapar/correlation_p_fapar_ge3_15.tif"
CORR_P_WTD_15 = "data/correlation_p_wtd/correlation_p_wtd_15.tif"
CORR_P_PET_WTD_15 = "data/correlation_p_pet_wtd/correlation_p_pet_wtd_15.tif"
CORR_WTD_T_15 = "data/correlation_wtd_t/correlation_wtd_t_15.tif"
CORR_P_T_15 = "/Volumes/Elements SE/Thesis/Data/correlation_p_t/correlation_p_t_15.tif"
CORR_T_FAPAR_15 = "data/correlation_t_fapar/correlation_t_fapar_ge3_15.tif"
CORR_P_PET_FAPAR_15 = "data/correlation_p_pet_fapar/correlation_p_pet_fapar_ge3_15.tif"

CORR_WTD_FAPAR_15_DOWNSAMPLED_10 = "data/correlation_wtd_fapar/correlation_wtd_fapar_ge3_15_downsampled_10.tif"
CORR_P_FAPAR_15_DOWNSAMPLED_10 = "data/correlation_p_fapar/correlation_p_fapar_ge3_15_downsampled_10.tif"
CORR_P_WTD_15_DOWNSAMPLED_10 = "data/correlation_p_wtd/correlation_p_wtd_15_downsampled_10.tif"
CORR_P_PET_WTD_15_DOWNSAMPLED_10 = "data/correlation_p_pet_wtd/correlation_p_pet_wtd_15_downsampled_10.tif"
CORR_WTD_T_15_DOWNSAMPLED_10 = "data/correlation_wtd_t/correlation_wtd_t_15_downsampled_10.tif"
CORR_P_T_15_DOWNSAMPLED_10 = "/Volumes/Elements SE/Thesis/Data/correlation_p_t/correlation_p_t_15_downsampled_10.tif"
CORR_T_FAPAR_15_DOWNSAMPLED_10 = "data/correlation_t_fapar/correlation_t_fapar_ge3_15_downsampled_10.tif"
CORR_P_PET_FAPAR_15_DOWNSAMPLED_10 = "data/correlation_p_pet_fapar/correlation_p_pet_fapar_ge3_15_downsampled_10.tif"

### Correlate WTD and fPAR
M_wtd = mp.Raster(WTD, tiles=(10, 10), window_size=15)
M_th = mp.Raster(TH, tiles=(10, 10), window_size=15)
M_fapar = mp.Raster(FAPAR, tiles=(10, 10), window_size=15)

with mp.Raster(CORR_WTD_FAPAR_15, mode='w', ref_map=WTD, tiles=(10, 10), window_size=15) as M_corr_wtd_fapar_15:
    for i in M_wtd:
        progress_bar((i + 1) / M_wtd.c_tiles)
        wtd = M_wtd[i]
        th = M_th[i]
        fapar = M_fapar[i]
        fapar[th < 3] = np.nan
        M_corr_wtd_fapar_15[i] = mp.correlate_maps(wtd, fapar, window_size=15, fraction_accepted=0.25)

M_corr_wtd_fapar_15 = mp.Raster(CORR_WTD_FAPAR_15, tiles=(10, 10))
M_corr_wtd_fapar_15.focal_mean(output_file=CORR_WTD_FAPAR_15_DOWNSAMPLED_10, window_size=10, reduce=True,
                               fraction_accepted=0)

mp.Raster.close()

### Correlate P and fPAR
M_p = mp.Raster(P, tiles=(10, 10), window_size=15)
M_th = mp.Raster(TH, tiles=(10, 10), window_size=15)
M_fapar = mp.Raster(FAPAR, tiles=(10, 10), window_size=15)

with mp.Raster(CORR_P_FAPAR_15, mode='w', ref_map=P, tiles=(10, 10), window_size=15) as M_corr_p_fapar_15:
    for i in M_p:
        progress_bar((i + 1) / M_p.c_tiles)
        c_p = M_p[i]
        th = M_th[i]
        fapar = M_fapar[i]
        fapar[th < 3] = np.nan
        M_corr_p_fapar_15[i] = mp.correlate_maps(c_p, fapar, window_size=15, verbose=False, fraction_accepted=0.25)

M_corr_p_fapar_15 = mp.Raster(CORR_P_FAPAR_15, tiles=(10, 10))
M_corr_p_fapar_15.focal_mean(output_file=CORR_P_FAPAR_15_DOWNSAMPLED_10, window_size=10, reduce=True,
                             fraction_accepted=0)

mp.Raster.close()

### Correlate P and WTD
M_p = mp.Raster(P)
M_wtd = mp.Raster(WTD)

mp.Raster.set_tiles((10, 10))
mp.Raster.set_window_size(15)

M_p.correlate(M_wtd, output_file=CORR_P_WTD_15, verbose=False, fraction_accepted=0.25)

M_corr_p_wtd = mp.Raster(CORR_P_WTD_15, tiles=(10, 10))
M_corr_p_wtd.focal_mean(output_file=CORR_P_WTD_15_DOWNSAMPLED_10, window_size=10, reduce=True, fraction_accepted=0)

mp.Raster.close()

### Correlate P_PET and WTD
M_p_pet = mp.Raster(P_PET)
M_wtd = mp.Raster(WTD)

mp.Raster.set_tiles((10, 10))
mp.Raster.set_window_size(15)

M_p_pet.correlate(M_wtd, output_file=CORR_P_PET_WTD_15, verbose=False, fraction_accepted=0.25)

M_corr_p_pet_wtd = mp.Raster(CORR_P_PET_WTD_15, tiles=(10, 10))
M_corr_p_pet_wtd.focal_mean(output_file=CORR_P_PET_WTD_15_DOWNSAMPLED_10, window_size=10, reduce=True,
                            fraction_accepted=0)

mp.Raster.close()

### Correlate T and fPAR
M_t = mp.Raster(T, tiles=(10, 10), window_size=15)
M_th = mp.Raster(TH, tiles=(10, 10), window_size=15)
M_fapar = mp.Raster(FAPAR, tiles=(10, 10), window_size=15)

with mp.Raster(CORR_T_FAPAR_15, mode='w', ref_map=WTD, tiles=(10, 10), window_size=15) as M_corr_t_fapar_15:
    for i in M_t:
        progress_bar((i + 1) / M_t.c_tiles)
        t = M_t[i]
        th = M_th[i]
        fapar = M_fapar[i]
        fapar[th < 3] = np.nan
        M_corr_t_fapar_15[i] = mp.correlate_maps(t, fapar, window_size=15, fraction_accepted=0.25)

M_corr_t_fapar_15 = mp.Raster(CORR_T_FAPAR_15, tiles=(10, 10))
M_corr_t_fapar_15.focal_mean(output_file=CORR_T_FAPAR_15_DOWNSAMPLED_10, window_size=10, reduce=True,
                             fraction_accepted=0)

### Correlate P and T
M_p = mp.Raster(P)
M_t = mp.Raster(T)

mp.Raster.set_tiles((10, 10))
mp.Raster.set_window_size(15)

M_p.correlate(M_t, output_file=CORR_P_T_15, window_size=15, fraction_accepted=0.25)

M_corr_p_t = mp.Raster(CORR_P_T_15, tiles=(10, 10))
M_corr_p_t.focal_mean(output_file=CORR_P_T_15_DOWNSAMPLED_10, window_size=10, reduce=True, fraction_accepted=0)

mp.Raster.close()

### Correlate WTD and T
M_wtd = mp.Raster(WTD)
M_t = mp.Raster(T)

mp.Raster.set_tiles((10, 10))
mp.Raster.set_window_size(15)

M_wtd.correlate(M_t, output_file=CORR_WTD_T_15, verbose=False, fraction_accepted=0.25)

M_corr_wtd_t = mp.Raster(CORR_WTD_T_15, tiles=(10, 10))
M_corr_wtd_t.focal_mean(output_file=CORR_WTD_T_15_DOWNSAMPLED_10, window_size=10, reduce=True, fraction_accepted=0)

mp.Raster.close()

### Correlate P_PET and fPAR
M_p_pet = mp.Raster(P_PET, tiles=(10, 10), window_size=15)
M_th = mp.Raster(TH, tiles=(10, 10), window_size=15)
M_fapar = mp.Raster(FAPAR, tiles=(10, 10), window_size=15)

with mp.Raster(CORR_P_PET_FAPAR_15, mode='w', ref_map=P_PET, tiles=(10, 10), window_size=15) as M_corr_p_pet_fapar_15:
    for i in M_p_pet:
        progress_bar((i + 1) / M_p_pet.c_tiles)
        p_pet = M_p_pet[i]
        th = M_th[i]
        fapar = M_fapar[i]
        fapar[th < 3] = np.nan
        M_corr_p_pet_fapar_15[i] = mp.correlate_maps(p_pet, fapar, window_size=15, fraction_accepted=0.25)

M_corr_p_pet_fapar_15 = mp.Raster(CORR_P_PET_FAPAR_15, tiles=(10, 10))
M_corr_p_pet_fapar_15.focal_mean(output_file=CORR_P_PET_FAPAR_15_DOWNSAMPLED_10, window_size=10, reduce=True,
                                 fraction_accepted=0)

mp.Raster.close()

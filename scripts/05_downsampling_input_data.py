#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import geomappy as mp
import numpy as np

CLIMATE = "data/climate/climate.tif"
P = "data/precipitation/precipitation.tif"
P_PET = "data/pet/p_pet.tif"
WTD = "data/wtd/wtd.tif"
WTD_STD = "data/wtd/wtd_std_5.tif"
TH = "data/tree_height/tree_height_global.tif"
FAPAR = "data/fapar/mean_fpar_reprojected.tif"

CLIMATE_DOWNSAMPLED_10 = "data/climate/climate_downsampled_10.tif"
CLIMATE_DOWNSAMPLED_10_DISPLAY = "data/climate/climate_downsampled_10_display.tif"

P_DOWNSAMPLED_10 = "data/precipitation/precipitation_downsampled_10.tif"

P_PET_DOWNSAMPLED_10 = "data/pet/p_pet.tif_downsampled_10.tif"

WTD_DOWNSAMPLED_10 = "data/wtd/wtd_downsampled_10.tif"
WTD_STD_DOWNSAMPLED_10 = "data/wtd/wtd_std_5_downsampled_10.tif"

TH_DOWNSAMPLED_10 = "data/tree_height/tree_height_global_downsampled_10.tif"

FAPAR_DOWNSAMPLED_10 = "data/fapar/mean_fpar_downsampled_10.tif"

M_clim = mp.Map(CLIMATE, tiles=(10, 20))
M_p = mp.Map(P, tiles=(10, 20))
M_p_pet = mp.Map(P_PET, tiles=(10, 20))
M_wtd = mp.Map(WTD, tiles=(10, 20))
M_wtd_std = mp.Map(WTD_STD, tiles=(10, 20))
M_th = mp.Map(TH, tiles=(10, 20))
M_fapar = mp.Map(FAPAR, tiles=(10, 20))

M_clim.focal_majority(output_file=CLIMATE_DOWNSAMPLED_10, window_size=10, reduce=True, fraction_accepted=0,
                      majority_mode='nan', dtype=np.float64)
M_clim.focal_majority(output_file=CLIMATE_DOWNSAMPLED_10_DISPLAY, window_size=10, reduce=True, fraction_accepted=0,
                      majority_mode='ascending', dtype=np.float64)

M_p.focal_mean(output_file=P_DOWNSAMPLED_10, window_size=10, reduce=True, fraction_accepted=0)
M_p_pet.focal_mean(output_file=P_PET_DOWNSAMPLED_10, window_size=10, reduce=True, fraction_accepted=0)
M_wtd.focal_mean(output_file=WTD_DOWNSAMPLED_10, window_size=10, reduce=True, fraction_accepted=0)
M_wtd_std.focal_mean(output_file=WTD_STD_DOWNSAMPLED_10, window_size=10, reduce=True, fraction_accepted=0)
M_th.focal_mean(output_file=TH_DOWNSAMPLED_10, window_size=10, reduce=True, fraction_accepted=0)
M_fapar.focal_mean(output_file=FAPAR_DOWNSAMPLED_10, window_size=10, reduce=True, fraction_accepted=0)

mp.Map.close()

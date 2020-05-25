#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create world map classified in:
    1: Open water / wetland / polder
    2: Lowland
    3: Undulating terrain
    4: Hilly
    5: Low mountainous
    6: Mountainous
    7: High mountainous
"""

import thesis
import numpy as np
import geomappy as mp
import warnings
warnings.filterwarnings("once")


def update_class_map(ind, val):
    global classified_map, accessed_map
    ind = np.logical_and(~accessed_map, ind)
    classified_map[ind] = val
    accessed_map[ind] = True


wtd = "data/wtd/wtd_original.tif"
wtd_mean_5 = "data/wtd/wtd_mean_5.tif"
wtd_std_5 = "data/wtd/wtd_std_5.tif"
output_map = "data/landscape_classes/landscape_classes.tif"

M_wtd = mp.Map(wtd, tiles=(20, 20))
# M_wtd.focal_mean(output_file=wtd_mean_5, window_size=5)
# M_wtd.focal_std(output_file=wtd_std_5, window_size=5)

M_wtd_mean_5 = mp.Map(wtd_mean_5, tiles=(20, 20))
M_wtd_std_5 = mp.Map(wtd_std_5, tiles=(20, 20))

M_output = mp.Map(output_map, ref_map=wtd, tiles=(20, 20), mode="w", dtype=np.int8, nodata=0, overwrite=True)

for i in M_wtd:
    mp.progress_bar((i+1)/M_wtd.c_tiles)
    wtd = M_wtd[i]

    if np.sum(~np.isnan(wtd)) == 0:
        M_output[i] = wtd.astype(np.int8)
        continue

    wtd_std_5 = M_wtd_std_5[i]
    wtd_mean_5 = M_wtd_mean_5[i]

    ###########################################################
    classified_map = np.full_like(wtd, 0, dtype=np.int8)
    accessed_map = np.zeros_like(wtd, dtype="bool")
    accessed_map[np.isnan(wtd)] = True

    # Open water / wetland
    ind = wtd_mean_5 < 0.25
    update_class_map(ind, 1)
    
    # Lowland
    ind = wtd_std_5 < 1
    update_class_map(ind, 2)
    
    # Undulating terrain including levees
    ind = wtd_std_5 < 5
    update_class_map(ind, 3)
    
    # Hilly
    ind = wtd_std_5 < 25
    update_class_map(ind, 4)
    
    # Low mountainous
    ind = wtd_std_5 < 50
    update_class_map(ind, 5)
    
    # Mountainous
    ind = wtd_std_5 < 150
    update_class_map(ind, 6)
    
    # Mountain peeks 
    # 150 meters sigma -> range of 6*sigma = 1000 meters
    ind = wtd_std_5 >= 150
    update_class_map(ind, 7)

    M_output[i] = classified_map

mp.Map.close()

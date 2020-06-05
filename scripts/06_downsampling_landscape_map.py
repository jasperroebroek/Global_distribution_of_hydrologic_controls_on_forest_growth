#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 11:42:35 2019
@author: jroebroek
"""
import numpy as np
import geomappy as mp

LANDSCAPE = "data/landscape_classes/landscape_classes.tif"
LANDSCAPE_DOWNSAMPLED_10 = "data/landscape_classes/landscape_downsampled_10.tif"
LANDSCAPE_DOWNSAMPLED_10_DISPLAY = "data/landscape_classes/landscape_downsampled_10_display.tif"

landscape = mp.Map(LANDSCAPE, tiles=(20, 20))

landscape.focal_majority(output_file=LANDSCAPE_DOWNSAMPLED_10, reduce=True, window_size=10)
landscape.focal_majority(output_file=LANDSCAPE_DOWNSAMPLED_10_DISPLAY, reduce=True, window_size=10,
                         majority_mode="ascending", dtype=np.float64)

mp.Map.close()

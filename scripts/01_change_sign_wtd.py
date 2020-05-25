"""
Convert water table depth dataset to be defined negatively
"""
import thesis
import geomappy as mp
import numpy as np

WTD_ORIGINAL = "data/wtd/wtd_original.tif"
WTD_CONVERTED = "data/wtd/wtd_2.tif"

with mp.Map(WTD_ORIGINAL) as src:
    with mp.Map(WTD_CONVERTED, ref_map=WTD_ORIGINAL, mode='w', nodata=np.nan, dtype=np.float32, overwrite=True) as dst:
        dst[0] = -src[0]

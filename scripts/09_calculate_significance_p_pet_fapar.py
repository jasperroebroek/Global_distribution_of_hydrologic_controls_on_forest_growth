import thesis
import geomappy as mp
from geomappy.rolling import rolling_sum
from geomappy.utils import progress_bar
from scipy.stats import t
import numpy as np
np.warnings.filterwarnings('ignore')  # comparison with nans that can be ignored

M_fapar = mp.Raster("data/fapar/mean_fpar_reprojected.tif")
M_p_pet = mp.Raster("data/pet/p_pet.tif")
M_corr = mp.Raster("data/correlation_p_pet_fapar/correlation_p_pet_fapar_ge3_15.tif")
M_th = mp.Raster("data/tree_height/tree_height_global.tif")

M_sig = mp.Raster("data/cmap_2d_significant/p_pet_fapar_sign.tif", mode="w", ref_map="data/wtd/wtd.tif")

window_size = 15
fringe = window_size // 2
n_max = window_size ** 2

sign_threshold = np.full(n_max + 1, np.nan)

for n in range(0, n_max + 1):
    # The 0.284 value was obtained from 06_global_threshold_table.py
    df = n - 2 - (0.284 * n)
    x = t.ppf(0.95, df=df)
    r = x / np.sqrt(df + x ** 2)
    sign_threshold[n] = x / np.sqrt(df + x ** 2)

mp.Raster.set_window_size(window_size)
M_corr.window_size = 1
mp.Raster.set_tiles((10, 10))

for i in M_th:
    progress_bar((i + 1) / M_th.c_tiles)
    th = M_th[i]
    p_pet = M_p_pet[i]
    fapar = M_fapar[i]
    fapar[th < 3] = np.nan
    corr = M_corr[i]

    count_values = rolling_sum(np.logical_and(~np.isnan(fapar), ~np.isnan(p_pet)), window_size=15)

    threshold = sign_threshold[count_values]

    significance = np.full_like(corr, 0, dtype=np.float64)
    significance[np.isnan(corr)] = np.nan
    significance[corr < -threshold] = -1
    significance[corr > threshold] = 1

    M_sig[i] = significance

mp.Raster.close()

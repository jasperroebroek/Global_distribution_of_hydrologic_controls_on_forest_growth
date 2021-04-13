import geomappy as mp
from geomappy.utils import progress_bar
from geomappy.rolling import rolling_sum, rolling_window
import numpy as np
from numba import njit, prange
from scipy.stats import t, mode
import matplotlib.pyplot as plt


@njit
def correlation_threshold_inference(map1, map2, num_obs=225):
    shape = map1.shape[0]
    bs = np.full(shape, np.nan)
    pe = np.full(shape, np.nan)

    for i in prange(0, shape):
        d1 = map1[i].ravel()
        d2 = map2[i].ravel()

        mask = np.logical_and(~np.isnan(d1), ~np.isnan(d2))
        d1 = d1[mask]
        d2 = d2[mask]

        if d1.size < num_obs:
            continue

        d1_mean = d1.mean()
        d2_mean = d2.mean()

        d1_dist = d1 - d1_mean
        d2_dist = d2 - d2_mean

        r_den = np.sqrt(np.sum(d1_dist ** 2) * np.sum(d2_dist ** 2))

        if r_den == 0:
            continue

        # bootstrapping
        samples = np.full(1000, np.nan)
        for ii in range(1000):
            while True:
                idx = np.random.choice(np.arange(d1.size), d1.size, replace=True)
                x = d1[idx]
                y = d2[idx]

                x_mean = x.mean()
                y_mean = y.mean()

                x_dist = x - x_mean
                y_dist = y - y_mean

                r_num_sample = np.sum(x_dist * y_dist)
                r_den_sample = np.sqrt(np.sum(x_dist ** 2) * np.sum(y_dist ** 2))

                if r_den_sample != 0:
                    samples[ii] = r_num_sample / r_den_sample
                    break

        bs[i] = 2 * samples.std()

        # permutation test
        samples = np.full(1000, np.nan)
        for ii in range(1000):
            while True:
                idx = np.random.choice(np.arange(d1.size), d1.size, replace=False)
                x = d1
                y = d2[idx]

                x_mean = x.mean()
                y_mean = y.mean()

                x_dist = x - x_mean
                y_dist = y - y_mean

                r_num_sample = np.sum(x_dist * y_dist)
                r_den_sample = np.sqrt(np.sum(x_dist ** 2) * np.sum(y_dist ** 2))

                if r_den_sample != 0:
                    samples[ii] = r_num_sample / r_den_sample
                    break

        pe[i] = 2 * samples.std()

    return bs, pe


def calc_df_offset(value, n=225):
    df_offset = 0
    while True:
        df = n - 2 - df_offset
        t_value = t.ppf(0.95, df=df)
        threshold = t_value / np.sqrt(df + t_value ** 2)
        if threshold > value:
            return df_offset
        df_offset = df_offset + 1


M_th = mp.Raster("data/tree_height/tree_height_global.tif")
M_fapar = mp.Raster("data/fapar/mean_fpar_reprojected.tif")
M_p_pet = mp.Raster("data/pet/p_pet.tif")
M_wtd = mp.Raster("data/wtd/wtd.tif")

mp.Raster.set_window_size(15)
mp.Raster.set_tiles((20, 20))

d_p_pet = {}
d_wtd = {}
for i in M_th:
    progress_bar((i+1) / M_th.c_tiles)
    th = M_th[i]
    fapar = M_fapar[i]
    fapar[th < 3] = np.nan
    p_pet = M_p_pet[i]
    wtd = M_wtd[i]

    # P_PET
    nans = np.logical_or(np.isnan(fapar), np.isnan(p_pet))
    mask = (rolling_sum(nans, window_size=15) == 0)

    a = rolling_window(fapar, window_size=15)[mask]
    b = rolling_window(p_pet, window_size=15)[mask]

    if a.size == 0:
        continue

    sample_size = min(100, a.shape[0])

    x = correlation_threshold_inference(a[np.random.choice(np.arange(a.shape[0]), sample_size, replace=False)],
                                        b[np.random.choice(np.arange(b.shape[0]), sample_size, replace=False)])
    d_p_pet[i] = x

    # WTD
    nans = np.logical_or(np.isnan(fapar), np.isnan(wtd))
    mask = (rolling_sum(nans, window_size=15) == 0)

    a = rolling_window(fapar, window_size=15)[mask]
    b = rolling_window(wtd, window_size=15)[mask]

    sample_size = min(100, a.shape[0])

    x = correlation_threshold_inference(a[np.random.choice(np.arange(a.shape[0]), sample_size, replace=False)],
                                        b[np.random.choice(np.arange(b.shape[0]), sample_size, replace=False)])
    d_wtd[i] = x

data_p_pet = np.hstack([d_p_pet[x] for x in d_p_pet])
data_wtd = np.hstack([d_wtd[x] for x in d_wtd])

mask = np.isnan(data_p_pet[0])
data_p_pet = data_p_pet[:, ~mask]

mask = np.isnan(data_wtd[0])
data_wtd = data_wtd[:, ~mask]

xs = np.arange(0, 0.21, 0.025)
xs_threshold = [calc_df_offset(x) for x in xs]

print(f"WTD degree of freedom offset: {calc_df_offset(np.mean((np.median(data_p_pet[0]), np.median(data_p_pet[1]))))}")
print(f"P/pET degree of freedom offset: {calc_df_offset(np.mean((np.median(data_wtd[0]), np.median(data_wtd[1]))))}")

f, ax = plt.subplots(ncols=2, figsize=(15, 5))

ax[0].hist(data_p_pet[0], bins=30, range=(0.00, 0.2), alpha=0.5, density=True)
ax[0].hist(data_p_pet[1], bins=30, range=(0.00, 0.2), alpha=0.5, density=True)
ax[0].axvline(np.median(data_p_pet[0]), color='Blue', label='Bootstrap')
ax[0].axvline(np.median(data_p_pet[1]), color='Red', label='Permutation test')
ax[0].axvline(np.mean((np.median(data_p_pet[0]), np.median(data_p_pet[1]))), color='Grey', label='Chosen threshold')
ax[0].set_xlabel("Correlation threshold")
ax[0].legend()

ax = np.append(ax, ax[0].twiny())
ax[2].xaxis.set_ticks_position('bottom') # set the position of the second x-axis to bottom
ax[2].xaxis.set_label_position('bottom') # set the position of the second x-axis to bottom
ax[2].spines['bottom'].set_position(('outward', 44))
ax[2].set_xlabel('Degree of freedom offset')
ax[2].set_xlim(ax[0].get_xlim())
ax[2].set_xticks(xs)
ax[2].set_xticklabels(xs_threshold)

ax[1].hist(data_wtd[0], bins=30, range=(0.00, 0.2), alpha=0.5, density=True)
ax[1].hist(data_wtd[1], bins=30, range=(0.00, 0.2), alpha=0.5, density=True)
ax[1].axvline(np.median(data_wtd[0]), color='Blue', label='Bootstrap')
ax[1].axvline(np.median(data_wtd[1]), color='Red', label='Permutation test')
ax[1].axvline(np.mean((np.median(data_wtd[0]), np.median(data_wtd[1]))), color='Grey', label='Chosen threshold')
ax[1].set_xlabel("Correlation threshold")
# ax[0].legend()

ax = np.append(ax, ax[1].twiny())
ax[3].xaxis.set_ticks_position('bottom') # set the position of the second x-axis to bottom
ax[3].xaxis.set_label_position('bottom') # set the position of the second x-axis to bottom
ax[3].spines['bottom'].set_position(('outward', 44))
ax[3].set_xlabel('Degree of freedom offset')
ax[3].set_xlim(ax[1].get_xlim())
ax[3].set_xticks(xs)
ax[3].set_xticklabels(xs_threshold)

ax[0].set_title("Correlation P/pET with fPAR")
ax[1].set_title("Correlation WTD with fPAR")
plt.savefig("figures/treshold.png", dpi=300, bbox_inches='tight')
plt.show()

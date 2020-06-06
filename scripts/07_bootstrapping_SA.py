from thesis import *
import numpy as np
from matplotlib.colors import ListedColormap
from numba import njit
from scipy.stats import t
import geomappy as mp
from geomappy.colors import cmap_2d
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


@njit
def correlation_significance_inference(map1, map2, window_size=5, fraction_accepted=0.7, t_test_thresholds=None):
    shape = (map1.shape[0] // window_size, map2.shape[1] // window_size)
    corr_t = np.full(shape, np.nan)
    corr_bs = np.full(shape, np.nan)
    corr_pe = np.full(shape, np.nan)

    for i in range(0, map1.shape[0], window_size):
        print(i)
        for j in range(0, map1.shape[1], window_size):
            ind = (slice(i, i + window_size), slice(j, j + window_size))
            ind_i = i // window_size
            ind_j = j // window_size

            d1 = map1[ind].ravel()
            d2 = map2[ind].ravel()

            mask = np.logical_and(~np.isnan(d1), ~np.isnan(d2))
            d1 = d1[mask]
            d2 = d2[mask]

            if d1.size < fraction_accepted * window_size ** 2:
                continue

            d1_mean = d1.mean()
            d2_mean = d2.mean()

            d1_dist = d1 - d1_mean
            d2_dist = d2 - d2_mean

            r_num = np.sum(d1_dist * d2_dist)
            r_den = np.sqrt(np.sum(d1_dist ** 2) * np.sum(d2_dist ** 2))

            if r_den == 0:
                continue

            correlation_value = r_num / r_den
            if correlation_value > t_test_thresholds[d1.size]:
                corr_t[ind_i, ind_j] = 2
            elif correlation_value < -t_test_thresholds[d1.size]:
                corr_t[ind_i, ind_j] = 0
            else:
                corr_t[ind_i, ind_j] = 1

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

            bootstrap_threshold = 2 * samples.std()
            if correlation_value > bootstrap_threshold:
                corr_bs[ind_i, ind_j] = 2
            elif correlation_value < -bootstrap_threshold:
                corr_bs[ind_i, ind_j] = 0
            else:
                corr_bs[ind_i, ind_j] = 1

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

            permutation_threshold = 2 * samples.std()
            if correlation_value > permutation_threshold:
                corr_pe[ind_i, ind_j] = 2
            elif correlation_value < -permutation_threshold:
                corr_pe[ind_i, ind_j] = 0
            else:
                corr_pe[ind_i, ind_j] = 1

    return corr_t, corr_bs, corr_pe


def t_test_threshold_table(offset_fraction=0, window_size=15):
    n_max = window_size ** 2
    sign_threshold = np.full(n_max + 1, np.nan)
    for n in range(0, n_max + 1):
        df = n - 2 - (offset_fraction * n)
        x = t.ppf(0.95, df=df)
        r = x / np.sqrt(df + x ** 2)
        sign_threshold[n] = x / np.sqrt(df + x ** 2)
    return sign_threshold


M_wtd = mp.Raster("/Volumes/Elements SE/Thesis/Data_raw/wtd/wtd_South_America_CF.tif", epsg=4326)
wtd = -M_wtd[0]
th = mp.Raster("data/tree_height/tree_height_global.tif")[M_wtd.bounds]
fapar = mp.Raster("data/fapar/mean_fpar_reprojected.tif")[M_wtd.bounds]
p = mp.Raster("data/pet/p_pet.tif")[M_wtd.bounds]
fapar[th < 3] = np.nan

# offset_fraction value from previous file (06_global_threshold_table.py)
WTD_corr_t, WTD_corr_bs, WTD_corr_pe = correlation_significance_inference(wtd, fapar, window_size=15,
    fraction_accepted=0.25, t_test_thresholds=t_test_threshold_table(offset_fraction=0.178))
P_corr_t, P_corr_bs, P_corr_pe = correlation_significance_inference(p, fapar, window_size=15,
    fraction_accepted=0.25, t_test_thresholds=t_test_threshold_table(offset_fraction=0.284))

WTD_mask = np.logical_or.reduce((np.isnan(WTD_corr_t), np.isnan(WTD_corr_bs), np.isnan(WTD_corr_pe)))
cm = confusion_matrix(WTD_corr_t[~WTD_mask], WTD_corr_bs[~WTD_mask])
print(f"WTD bootstrapping: {cm.diagonal().sum()/cm.sum()}")
cm = confusion_matrix(WTD_corr_t[~WTD_mask], WTD_corr_pe[~WTD_mask])
print(f"WTD permutation: {cm.diagonal().sum()/cm.sum()}")

P_mask = np.logical_or.reduce((np.isnan(P_corr_t), np.isnan(P_corr_bs), np.isnan(P_corr_pe)))
cm = confusion_matrix(P_corr_t[~P_mask], P_corr_bs[~P_mask])
print(f"P/pET bootstrapping: {cm.diagonal().sum()/cm.sum()}")
cm = confusion_matrix(P_corr_t[~P_mask], P_corr_pe[~P_mask])
print(f"P/pET permutation: {cm.diagonal().sum()/cm.sum()}")

mask = np.logical_or(WTD_mask, P_mask)
cmap = np.arange(9).reshape(3, 3)
classification_t = cmap[P_corr_t[~mask].astype(int), WTD_corr_t[~mask].astype(int)]
classification_bs = cmap[P_corr_bs[~mask].astype(int), WTD_corr_bs[~mask].astype(int)]
classification_pe = cmap[P_corr_pe[~mask].astype(int), WTD_corr_pe[~mask].astype(int)]

cm_t = confusion_matrix(classification_t, classification_t)
cm_bs = confusion_matrix(classification_t, classification_bs)
print(f"Classification bootstrapping: {cm_bs.diagonal().sum()/cm_bs.sum()}")
cm_pe = confusion_matrix(classification_t, classification_pe)
print(f"Classification permutation: {cm_pe.diagonal().sum()/cm_pe.sum()}")

cmap = cmap_2d((3, 3), alpha=0.5, diverging_alpha=0.25).astype(np.float32)
cmap[1, 1, :] = (0.9, 0.9, 0.9)
cmap = np.vstack((cmap, np.ones((1, 3, 3))))
cmap = np.hstack((cmap, np.ones((4, 1, 3))))

P_corr_t_plot = P_corr_t.copy()
P_corr_t_plot[mask] = 3
P_corr_t_plot = P_corr_t_plot.astype(int)
WTD_corr_t_plot = WTD_corr_t.copy()
WTD_corr_t_plot[mask] = 3
WTD_corr_t_plot = WTD_corr_t_plot.astype(int)
classification_t_plot = cmap[P_corr_t_plot, WTD_corr_t_plot]

P_corr_bs_plot = P_corr_bs.copy()
P_corr_bs_plot[mask] = 3
P_corr_bs_plot = P_corr_bs_plot.astype(int)
WTD_corr_bs_plot = WTD_corr_bs.copy()
WTD_corr_bs_plot[mask] = 3
WTD_corr_bs_plot = WTD_corr_bs_plot.astype(int)
classification_bs_plot = cmap[P_corr_bs_plot, WTD_corr_bs_plot]

P_corr_pe_plot = P_corr_pe.copy()
P_corr_pe_plot[mask] = 3
P_corr_pe_plot = P_corr_pe_plot.astype(int)
WTD_corr_pe_plot = WTD_corr_pe.copy()
WTD_corr_pe_plot[mask] = 3
WTD_corr_pe_plot = WTD_corr_pe_plot.astype(int)
classification_pe_plot = cmap[P_corr_pe_plot, WTD_corr_pe_plot]

# CONFUSION MATRIX LAYOUT
cmap_cm = cmap_2d((3, 3), alpha=0.5, diverging_alpha=0.25).astype(np.float32)
cmap_cm[1, 1, :] = (0.9, 0.9, 0.9)
cm_empty = np.full((10, 10), np.nan, dtype=float)
cm_empty[0, :] = np.arange(10) - 1
cm_empty[:, 0] = np.arange(10) - 1
cmap_cm = ListedColormap(np.vstack([[1, 1, 1], cmap_cm.reshape(9, 3)]))

l = [cm_t, cm_bs, cm_pe]
labels = ["T-test classification", "Bootstrap classification", "Permutation test classification"]
maps = [classification_t_plot, classification_bs_plot, classification_pe_plot]

f, ax = plt.subplots(ncols=3, nrows=2, figsize=(20, 20))
ax = ax.flatten()
plt.tight_layout(h_pad=-40, w_pad=2)
ax[:3] = [mp.basemap(*M_wtd.bounds, ax=cax, resolution='50m', xticks=10, yticks=10, linewidth=0.8) for cax in ax[:3]]

for num, cax in enumerate(ax[:3]):
    mp.plot_map(maps[num], ax=cax)
    cax.set_title(labels[num])

for num, cax in enumerate(ax[3:]):
    cm = l[num]
    cax.imshow(cm_empty, cmap=cmap_cm)
    cax.set_xticks([])
    cax.set_yticks([])
    cax.text(0, 0, f"{100*cm.diagonal().sum()/cm.sum():2.0f}%", ha='center', va='center', weight='bold')
    for i in range(9):
        cax.axhline(i+0.5, color='black', linestyle="--", alpha=0.2)
        cax.axvline(i+0.5, color='black', linestyle="--", alpha=0.2)
        for j in range(9):
            if i == j:
                weight = 'bold'
            else:
                weight = 'normal'
            cax.text(i+1, j+1, cm[i, j], ha='center', va='center', weight=weight)
    cax.set_ylabel(labels[num], labelpad=2)
    cax.xaxis.set_label_position('top')
    cax.set_xlabel("T-test classification", labelpad=2)

plt.savefig("figures/confusion_matrices.png", dpi=300, bbox_inches='tight')
plt.show()

mp.Raster.close()

"""
Create correlation maps: WTD & FAPAR, P & FAPAR, WTD & P
"""
import thesis
import geomappy as mp
import numpy as np

WTD = "data/wtd/wtd.tif"
P = "data/precipitation/precipitation.tif"
TH = "data/tree_height/tree_height_global.tif"
FAPAR = "data/fapar/mean_fpar_reprojected.tif"
T = "data/temperature/temperature.tif"

CORR_WTD_FAPAR_15 = "data/correlation_wtd_fapar/correlation_wtd_fapar_ge3_15.tif"
CORR_P_FAPAR_15 = "data/correlation_p_fapar/correlation_p_fapar_ge3_15.tif"
CORR_P_WTD_15 = "data/correlation_p_wtd/correlation_p_wtd_15.tif"
CORR_WTD_T_15 = "data/correlation_wtd_t/correlation_wtd_t_15.tif"
CORR_P_T_15 = "/Volumes/Elements SE/Thesis/Data/correlation_p_t/correlation_p_t_15.tif"
CORR_T_FAPAR_15 = "data/correlation_t_fapar/correlation_t_fapar_ge3_15.tif"

CORR_WTD_FAPAR_15_DOWNSAMPLED_10 = "data/correlation_wtd_fapar/correlation_wtd_fapar_ge3_15_downsampled_10.tif"
CORR_P_FAPAR_15_DOWNSAMPLED_10 = "data/correlation_p_fapar/correlation_p_fapar_ge3_15_downsampled_10.tif"
CORR_P_WTD_15_DOWNSAMPLED_10 = "data/correlation_p_wtd/correlation_p_wtd_15_downsampled_10.tif"
CORR_WTD_T_15_DOWNSAMPLED_10 = "data/correlation_wtd_t/correlation_wtd_t_15_downsampled_10.tif"
CORR_P_T_15_DOWNSAMPLED_10 = "/Volumes/Elements SE/Thesis/Data/correlation_p_t/correlation_p_t_15_downsampled_10.tif"
CORR_T_FAPAR_15_DOWNSAMPLED_10 = "data/correlation_t_fapar/correlation_t_fapar_ge3_15_downsampled_10.tif"

# ### Correlate WTD and fPAR
# M_wtd = mp.Map(WTD, tiles=(10, 10), window_size=15)
# M_th = mp.Map(TH, tiles=(10, 10), window_size=15)
# M_fapar = mp.Map(FAPAR, tiles=(10, 10), window_size=15)
#
# with mp.Map(CORR_WTD_FAPAR_15, mode='w', ref_map=WTD, tiles=(10, 10),  window_size=15) as M_corr_wtd_fapar_15:
#     for i in M_wtd:
#         mp.progress_bar((i+1)/(M_wtd.c_tiles))
#         wtd = M_wtd[i]
#         th = M_th[i]
#         fapar = M_fapar[i]
#         fapar[th < 3] = np.nan
#         M_corr_wtd_fapar_15[i] = mp.correlate_maps(wtd, fapar, window_size=15, fraction_accepted=0.25)
#
# M_corr_wtd_fapar_15 = mp.Map(CORR_WTD_FAPAR_15, tiles=(10, 10))
# M_corr_wtd_fapar_15.focal_mean(output_file=CORR_WTD_FAPAR_DOWNSAMPLED_10, window_size=10, reduce=True,
#                                fraction_accepted=0)
#
# mp.Map.close()
#
# ### Correlate P and fPAR
# M_p = mp.Map(P, tiles=(10, 10), window_size=15)
# M_th = mp.Map(TH, tiles=(10, 10), window_size=15)
# M_fapar = mp.Map(FAPAR, tiles=(10, 10), window_size=15)
#
# with mp.Map(CORR_P_FAPAR_15, mode='w', ref_map=P, tiles=(10, 10),  window_size=15) as M_corr_p_fapar_15:
#     for i in M_p:
#         mp.progress_bar((i+1)/(M_p.c_tiles))
#         c_p = M_p[i]
#         th = M_th[i]
#         fapar = M_fapar[i]
#         fapar[th < 3] = np.nan
#         M_corr_p_fapar_15[i] = mp.correlate_maps(c_p, fapar, window_size=15, verbose=False, fraction_accepted=0.25)
#
# M_corr_p_fapar_15 = mp.Map(CORR_P_FAPAR_15, tiles=(10, 10))
# M_corr_p_fapar_15.focal_mean(output_file=CORR_P_FAPAR_DOWNSAMPLED_10, window_size=10, reduce=True, fraction_accepted=0)
#
# mp.Map.close()
#
# ### Correlate P and WTD
# M_p = mp.Map(P)
# M_wtd = mp.Map(WTD)
#
# mp.Map.set_tiles((10, 10))
# mp.Map.set_window_size(15)
#
# M_p.correlate(M_wtd, output_file=CORR_P_WTD_15, verbose=False, fraction_accepted=0.25)
#
# M_corr_p_wtd = mp.Map(CORR_P_WTD_15, tiles=(10, 10))
# M_corr_p_wtd.focal_mean(output_file=CORR_P_WTD_15_DOWNSAMPLED_10, window_size=10, reduce=True, fraction_accepted=0)
#
# mp.Map.close()
#
# ### Correlate T and fPAR
# M_t = mp.Map(T, tiles=(10, 10), window_size=15)
# M_th = mp.Map(TH, tiles=(10, 10), window_size=15)
# M_fapar = mp.Map(FAPAR, tiles=(10, 10), window_size=15)
#
# with mp.Map(CORR_T_FAPAR_15, mode='w', ref_map=WTD, tiles=(10, 10),  window_size=15) as M_corr_t_fapar_15:
#     for i in M_t:
#         mp.progress_bar((i+1)/(M_t.c_tiles))
#         t = M_t[i]
#         th = M_th[i]
#         fapar = M_fapar[i]
#         fapar[th < 3] = np.nan
#         M_corr_t_fapar_15[i] = mp.correlate_maps(t, fapar, window_size=15, fraction_accepted=0.25)
#
#
# M_corr_t_fapar_15 = mp.Map(CORR_T_FAPAR_15, tiles=(10, 10))
# M_corr_t_fapar_15.focal_mean(output_file=CORR_T_FAPAR_15_DOWNSAMPLED_10, window_size=10, reduce=True,
#                              fraction_accepted=0)

# ### Correlate P and T
# M_p = mp.Map(P)
# M_t = mp.Map(T)
#
# mp.Map.set_tiles((10, 10))
# mp.Map.set_window_size(15)

# M_p.correlate(M_t, output_file=CORR_P_T, window_size=15, fraction_accepted=0.25)

M_corr_p_t = mp.Map(CORR_P_T_15, tiles=(10, 10))
M_corr_p_t.focal_mean(output_file=CORR_P_T_15_DOWNSAMPLED_10, window_size=10, reduce=True, fraction_accepted=0)

mp.Map.close()

# ### Correlate WTD and T
# M_wtd = mp.Map(WTD)
# M_t = mp.Map(T)
#
# mp.Map.set_tiles((10, 10))
# mp.Map.set_window_size(15)
#
# M_wtd.correlate(M_t, output_file=CORR_WTD_T_15, verbose=False, fraction_accepted=0.25)
#
# M_corr_wtd_t = mp.Map(CORR_WTD_T_15, tiles=(10, 10))
# M_corr_wtd_t.focal_mean(output_file=CORR_WTD_T_15_DOWNSAMPLED_10, window_size=10, reduce=True, fraction_accepted=0)
#
# mp.Map.close()

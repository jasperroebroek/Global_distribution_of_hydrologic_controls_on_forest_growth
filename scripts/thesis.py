#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

from matplotlib.colors import ListedColormap

import geomappy as mp
from geomappy.bounds import bounds_to_polygons
from geomappy.colors import cmap_2d, cmap_from_borders
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colorbar import ColorbarBase
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, InsetPosition
import pandas as pd
import cartopy.crs as ccrs
os.chdir('../')


def mode(x, fill=np.nan):
    """
    Returns the mode of a list with a default value

    Parameters
    ----------
    x : pd.Series
        List of data on which the mode will be calculated. Must be convertable to pd.Series
    fill : numeric
        Default value

    Returns
    -------
    mode
    """
    mode = pd.Series.mode(x)
    if len(mode) != 1:
        return fill
    return mode


def draw_thesis_location_zoom(ind):
    """
    Drawing array of nine maps, displaying the input data and results. Not meant for custom use.

    Parameters
    ----------
    ind : .
        look at mp.RasterBase.get_pointer()
    """
    classes = mp.Raster("data/cmap_2d_significant/classes_serial_downsampled_10.tif")
    climate = mp.Raster("data/climate/climate_downsampled_10_display.tif")
    landscape = mp.Raster("data/landscape_classes/landscape_downsampled_10_display.tif")
    p_pet = mp.Raster("data/pet/p_pet_downsampled_10.tif")
    fapar = mp.Raster("data/fapar/mean_fpar_downsampled_10.tif")
    corr_wtd_fapar = mp.Raster("data/correlation_wtd_fapar/correlation_wtd_fapar_ge3_15_downsampled_10.tif")
    corr_p_pet_fapar = mp.Raster("data/correlation_p_pet_fapar/correlation_p_pet_fapar_ge3_15_downsampled_10.tif")
    corr_p_pet_wtd = mp.Raster("data/correlation_p_pet_wtd/correlation_p_pet_wtd_15_downsampled_10.tif")
    wtd = mp.Raster("data/wtd/wtd_downsampled_10.tif")

    f, ax = plt.subplots(3, 3, figsize=(20, 20))
    plt.tight_layout(h_pad=-15, w_pad=7)
    ax = [mp.basemap(ax=cax, coastlines=False) for cax in ax.flatten()]

    basemap_kwargs = {'coastline_linewidth': 0.3, 'xticks': 5, 'yticks': 5, 'resolution': '10m'}
    fontsize = 14

    # Classes
    cmap = cmap_2d((3, 3), alpha=0.5, diverging_alpha=0.25)
    cmap[1, 1, :] = (0.9, 0.9, 0.9)
    cmap = cmap.reshape((9, 3))
    classes.plot_classified_map(ind, basemap=True, basemap_kwargs=basemap_kwargs, legend=None, colors=cmap,
                                force_equal_figsize=True, ax=ax[0], fontsize=fontsize)
    ax[0].set_title("Ecohydrological classes", fontsize=fontsize)

    # P/PET
    bins = np.arange(0, 3.1, 0.2)
    p_pet.plot_map(ind, bins=bins, basemap=True, basemap_kwargs=basemap_kwargs, cmap="Blues", legend='colorbar',
                   ax=ax[1], fontsize=fontsize, legend_kwargs={'title': "[-]", 'title_font': {'pad': 10}})
    ax[1].set_title("P/PET", fontsize=fontsize)

    # WTD
    bins = [0, 1, 2, 5, 10, 15, 20, 25, 35, 50, 100, 300]
    bins = bins[::-1]
    bins = [-i for i in bins]
    wtd.plot_map(ind, bins=bins, basemap=True, basemap_kwargs=basemap_kwargs, cmap="Blues", legend='colorbar', ax=ax[2],
                 fontsize=fontsize, legend_kwargs={'title': "[m]", 'title_font': {'pad': 10}})
    ax[2].set_title("WTD", fontsize=fontsize)

    # FAPAR
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 100]
    fapar.plot_map(ind, bins=bins, basemap=True, basemap_kwargs=basemap_kwargs, cmap="Greens", legend='colorbar',
                   ax=ax[3], fontsize=fontsize, legend_kwargs={'title': "[%]", 'title_font': {'pad': 10}})
    ax[3].set_title("fAPAR", fontsize=fontsize)

    # Climate
    cmap = [(1, 1, 1)]
    bins = [0]
    labels = ["Water"]
    with open("data/climate/koppen_legend.txt") as f:
        for line in f:
            line = line.strip()
            try:
                int(line[0])
                rgb = [int(c) / 255 for c in line[line.find('[') + 1:-1].split()]
                cmap.append(rgb)
                labels.append(line.split()[1])
                bins.append(line[:line.find(':')])
            except:
                pass

    bins = np.array(bins, dtype=np.int)
    climate.plot_classified_map(ind, bins=bins, basemap=True, basemap_kwargs=basemap_kwargs, colors=cmap,
                                labels=labels, legend='colorbar', ax=ax[4], fontsize=fontsize, suppress_warnings=True,
                                clip_legend=True)

    ax[4].set_title("Climate", fontsize=fontsize)

    # Landscape
    cmap = ['#004dac', '#729116', '#b1bc1d', '#e7de23', '#af9a15', '#785707', '#fff9f2']
    labels = ['Wetland and open water', 'Lowland', 'Undulating', 'Hilly', 'Low mountainous', 'Mountainous',
              'High mountainous']
    bins = [1, 2, 3, 4, 5, 6, 7]
    landscape.plot_classified_map(ind, bins=bins, basemap=True, basemap_kwargs=basemap_kwargs, colors=cmap,
                                  labels=labels, legend='colorbar', ax=ax[5], fontsize=fontsize, suppress_warnings=True)
    ax[5].set_title("Landscape", fontsize=fontsize)

    # Correlation WTD and fAPAR
    bins = [-1, -0.5, -0.25, -0.15, -0.05, 0.05, 0.15, 0.25, 0.5, 1]
    corr_wtd_fapar.plot_map(ind, bins=bins, basemap=True, basemap_kwargs=basemap_kwargs, legend='colorbar',
                            ax=ax[6], fontsize=fontsize, cmap="RdYlBu_r")

    ax[6].set_title("Correlation WTD & fAPAR", fontsize=fontsize)

    # Correlation P and fAPAR
    corr_p_pet_fapar.plot_map(ind, bins=bins, basemap=True, basemap_kwargs=basemap_kwargs, legend='colorbar',
                              ax=ax[7], fontsize=fontsize, cmap="RdYlBu_r")

    ax[7].set_title("Correlation P/PET & fAPAR", fontsize=fontsize)

    # Correlation P/PET and fAPAR
    corr_p_pet_wtd.plot_map(ind, bins=bins, basemap=True, basemap_kwargs=basemap_kwargs, legend='colorbar',
                            ax=ax[8], fontsize=fontsize, cmap="RdYlBu_r")
    ax[8].set_title("Correlation P/PET & WTD", fontsize=fontsize)
    mp.Raster.close(verbose=False)


def draw_thesis_location_zoom_full(ind, ticks=1):
    """
    Drawing array of nine maps, displaying the input data and results. Not meant for custom use.

    Parameters
    ----------
    ind : .
        look at mp.RasterBase.get_pointer()
    """
    classes = mp.Raster("data/cmap_2d_significant/classes_serial.tif")
    climate = mp.Raster("data/climate/climate.tif")
    landscape = mp.Raster("data/wtd/wtd_std_5.tif")
    p_pet = mp.Raster("data/pet/p_pet.tif")
    fapar = mp.Raster("data/fapar/mean_fpar_reprojected.tif")
    corr_wtd_fapar = mp.Raster("data/correlation_wtd_fapar/correlation_wtd_fapar_ge3_15.tif")
    corr_p_pet_fapar = mp.Raster("data/correlation_p_pet_fapar/correlation_p_pet_fapar_ge3_15.tif")
    corr_p_pet_wtd = mp.Raster("data/correlation_p_pet_wtd/correlation_p_pet_wtd_15.tif")
    wtd = mp.Raster("data/wtd/wtd.tif")

    f, ax = plt.subplots(3, 3, figsize=(20, 20))
    plt.tight_layout(h_pad=-15, w_pad=7)
    ax = [mp.basemap(ax=cax, coastlines=False) for cax in ax.flatten()]

    basemap_kwargs = {'coastline_linewidth': 0.3, 'xticks': ticks, 'yticks': ticks, 'resolution': '10m'}
    fontsize = 14

    # Classes
    cmap = cmap_2d((3, 3), alpha=0.5, diverging_alpha=0.25)
    cmap[1, 1, :] = (0.9, 0.9, 0.9)
    cmap = cmap.reshape((9, 3))
    cmap = np.vstack(((1,1,1), cmap))
    classes.plot_classified_map(ind, basemap=True, basemap_kwargs=basemap_kwargs, legend=None, colors=cmap,
                                force_equal_figsize=True, ax=ax[0], fontsize=fontsize, suppress_warnings=True)
    ax[0].set_title("Ecohydrological classes", fontsize=fontsize)

    # P/PET
    bins = np.arange(0, 3.1, 0.2)
    p_pet.plot_map(ind, bins=bins, basemap=True, basemap_kwargs=basemap_kwargs, cmap="Blues", legend='colorbar',
                   ax=ax[1], fontsize=fontsize, legend_kwargs={'title': "[-]", 'title_font': {'pad': 10}})
    ax[1].set_title("P/PET", fontsize=fontsize)

    # WTD
    bins = [0, 1, 2, 5, 10, 15, 20, 25, 35, 50, 100, 200, 300, 1000]
    bins = bins[::-1]
    bins = [-i for i in bins]
    wtd.plot_map(ind, bins=bins, basemap=True, basemap_kwargs=basemap_kwargs, cmap="Blues", legend='colorbar', ax=ax[2],
                 fontsize=fontsize, legend_kwargs={'title': "[m]", 'title_font': {'pad': 10}})
    ax[2].set_title("WTD", fontsize=fontsize)

    # FAPAR
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 100]
    fapar.plot_map(ind, bins=bins, basemap=True, basemap_kwargs=basemap_kwargs, cmap="Greens", legend='colorbar',
                   ax=ax[3], fontsize=fontsize, legend_kwargs={'title': "[%]", 'title_font': {'pad': 10}})
    ax[3].set_title("fAPAR", fontsize=fontsize)

    # Climate
    cmap = [(1, 1, 1)]
    bins = [0]
    labels = ["Water"]
    with open("data/climate/koppen_legend.txt") as f:
        for line in f:
            line = line.strip()
            try:
                int(line[0])
                rgb = [int(c) / 255 for c in line[line.find('[') + 1:-1].split()]
                cmap.append(rgb)
                labels.append(line.split()[1])
                bins.append(line[:line.find(':')])
            except:
                pass

    bins = np.array(bins, dtype=np.int)
    climate.plot_classified_map(ind, bins=bins, basemap=True, basemap_kwargs=basemap_kwargs, colors=cmap,
                                labels=labels, legend='colorbar', ax=ax[4], fontsize=fontsize, suppress_warnings=True,
                                clip_legend=True)

    ax[4].set_title("Climate", fontsize=fontsize)

    # Landscape
    dem_cmap = np.vstack((cmap_from_borders(("#0000b8", "#006310"), n=2, return_type="list"),
                          cmap_from_borders(("#006310", "#faea00"), n=7, return_type="list"),
                          cmap_from_borders(("#faea00", "#504100"), n=8, return_type="list"),
                          cmap_from_borders(("#504100", "#ffffff"), n=2, return_type="list")))
    bins = np.round(np.hstack(((0,), np.logspace(0, np.log10(400), num=18))), 1)
    landscape.plot_map(ind, bins=bins, basemap=True, basemap_kwargs=basemap_kwargs, legend='colorbar',
                       cmap=ListedColormap(dem_cmap), ax=ax[5], fontsize=fontsize)
    ax[5].set_title("Landscape", fontsize=fontsize)

    # Correlation WTD and fAPAR
    bins = [-1, -0.5, -0.25, -0.15, -0.05, 0.05, 0.15, 0.25, 0.5, 1]
    corr_wtd_fapar.plot_map(ind, bins=bins, basemap=True, basemap_kwargs=basemap_kwargs, legend='colorbar',
                            ax=ax[6], fontsize=fontsize, cmap="RdYlBu_r")

    ax[6].set_title("Correlation WTD & fAPAR", fontsize=fontsize)

    # Correlation P and fAPAR
    corr_p_pet_fapar.plot_map(ind, bins=bins, basemap=True, basemap_kwargs=basemap_kwargs, legend='colorbar',
                              ax=ax[7], fontsize=fontsize, cmap="RdYlBu_r")

    ax[7].set_title("Correlation P/PET & fAPAR", fontsize=fontsize)

    # Correlation P/PET and fAPAR
    corr_p_pet_wtd.plot_map(ind, bins=bins, basemap=True, basemap_kwargs=basemap_kwargs, legend='colorbar',
                            ax=ax[8], fontsize=fontsize, cmap="RdYlBu_r")
    ax[8].set_title("Correlation P/PET & WTD", fontsize=fontsize)
    mp.Raster.close(verbose=False)


def draw_thesis_map(loc, loc_small=None, classified=False, legend_2d=False, legend=None, fontsize=28, **kwargs):
    """
    Plot map for thesis. Not intended for reuse.
    """
    fontdict = {'fontsize': fontsize, 'va': 'center', 'ha': 'center'}

    M_thesis = mp.Raster(loc)
    if isinstance(loc_small, type(None)):
        loc_small = loc
    M_thesis_small = mp.Raster(loc_small)

    inset_coords = ((-93, 29, -78, 44),  # Mississippi
                    (7, 35, 22, 50),  # Italy
                    (15, -7, 30, 8),  # Congo
                    (142, -35, 157, -20))  # Australia

    bounds = [-180, -89, 180, 90]
    basemap_kwargs = {'coastline_linewidth': 0.25}
    ax = mp.basemap(*bounds, figsize=(40, 40), **basemap_kwargs)

    bounds = (-180, -60, 180, 90)
    extent = (bounds[0], bounds[2], bounds[1], bounds[3])
    if classified:
        ax, _legend = M_thesis.plot_classified_map(bounds, ax=ax, suppress_warnings=True, legend=legend, extent=extent,
                                                   transform=ax.projection, **kwargs)
    else:
        ax, _legend = M_thesis.plot_map(bounds, ax=ax, legend=legend, extent=extent, transform=ax.projection, **kwargs)

    inset_locations = [(-55, -88, 44, 44),
                       (-5, -88, 44, 44),
                       (45, -88, 44, 44),
                       (95, -88, 44, 44)]

    inset_locations = [((loc[0] + 180) / 360, (loc[1] + 90) / 180, loc[2] / 360, loc[3] / 180) for loc in
                       inset_locations]

    labels = [chr(65 + i) for i in range(len(inset_locations))]

    for i in range(len(inset_locations)):
        left, bottom, right, top = inset_coords[i]

        ax_inset = plt.gcf().add_subplot(projection=ccrs.PlateCarree(), label=f"{np.random.rand()}")

        basemap_kwargs = {'grid': False, 'coastline_linewidth': 0.5}
        if classified:
            ax_inset, _ = M_thesis_small.plot_classified_map(inset_coords[i], basemap=True, fontsize=fontsize,
                                                             resolution='10m', ax=ax_inset, xticks=[], yticks=[],
                                                             basemap_kwargs=basemap_kwargs, suppress_warnings=True,
                                                             legend=None, **kwargs)
        else:
            ax_inset, _ = M_thesis_small.plot_map(inset_coords[i], basemap=True, fontsize=fontsize, resolution='10m',
                                                  ax=ax_inset, xticks=[], yticks=[], basemap_kwargs=basemap_kwargs,
                                                  legend=None, **kwargs)

        ip = InsetPosition(ax, inset_locations[i])
        ax_inset.set_axes_locator(ip)

        bounds_to_polygons([inset_coords[i]]).plot_shapes(ax=ax, facecolor=(1, 1, 1, 0.5))

        text_location_x = (left + right) / 2
        text_location_y = (top + bottom) / 2
        t = ax.text(text_location_x, text_location_y, labels[i], fontdict=fontdict)
        t.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='grey'))

        text_location_x = inset_locations[i][0] + inset_locations[i][2] / 2
        text_location_y = inset_locations[i][1] + inset_locations[i][3] + 0.015
        t = ax.text(text_location_x, text_location_y, labels[i], fontdict=fontdict, transform=ax.transAxes, zorder=5)
        t.set_bbox(dict(facecolor='white', alpha=0.6, edgecolor='grey'))

    if legend_2d:
        data = mp.Raster("data/cmap_2d_significant/classes_serial_downsampled_10.tif")[0]
        data[np.isnan(data)] = -1
        bins, vals = np.unique(data, return_counts=True)
        bins = bins[1:]
        vals = vals[1:]
        vals = vals / vals.sum() * 100

        axins = inset_axes(ax, width="100%", height="100%",
                           bbox_to_anchor=(-0.01, 0.25, 0.26, 0.26),
                           bbox_transform=ax.transAxes)

        cmap = cmap_2d((3, 3), alpha=0.5, diverging_alpha=0.25)
        cmap[1, 1, :] = (0.9, 0.9, 0.9)

        axins.imshow(cmap, origin='lower')
        axins.set_xticks([0, 1, 2], minor=False)
        axins.set_yticks([0, 1, 2], minor=False)
        axins.set_xticklabels([u'\u2014', "0", "+"], minor=False, fontdict=fontdict)
        axins.set_yticklabels([u'\u2014', "0", "+"], minor=False, fontsize=fontsize)
        axins.set_xlabel("Correlation WTD and fAPAR", fontdict=fontdict, labelpad=20)
        axins.set_ylabel("Correlation P/PET and fAPAR", fontdict=fontdict, labelpad=20)
        axins.tick_params(axis='both', which='both', length=0, pad=20)

        for i in range(3):
            for j in range(3):
                t = axins.text(i, j, f'{vals[i + 3 * j]:1.0f}', fontdict=fontdict)
                t.set_bbox(dict(facecolor='white', alpha=0.6, edgecolor='grey'))

    ax.tick_params(axis='both', which='both', length=0, labelsize=fontsize, pad=12)
    if isinstance(_legend, ColorbarBase):
        _legend.ax.tick_params(labelsize=fontsize, pad=12)

    mp.Raster.close(verbose=False)

    return ax, _legend

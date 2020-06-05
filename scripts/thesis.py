#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import geomappy as mp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import ListedColormap
from mpl_toolkits import axes_grid1
import pandas as pd


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


def draw_thesis_location_zoom(ind, titles=True, figsize=(20, 20), save=None, legend_title=True, legend='colorbar',
                              **kwargs):
    """
    Drawing array of nine maps, displaying the input data and results. Not meant for custom use.

    Parameters
    ----------
    ind : .
        look at mp.MapBase.get_pointer()
    titles : bool, optional
        insert the titles above the plots
    figsize : tuple, optional
        matplotlib equivalent figsize. The default is (20,20)
    save : str, optional
        export location
    legend_title : bool, optional
        print the title of the legend
    legend : ['colorbar', 'legend'], optional
        print a hovering legend or a colorbar, which is the default.
    kwargs : dict, optional,
        keyword arguments passed into the mp.MapRead.plot_classified() command
    """
    if isinstance(kwargs, type(None)):
        kwargs = {}

    if 'legend_title_pad' not in kwargs:
        kwargs.update({'legend_title_pad': 5})

    pad_fraction = kwargs.get('pad_fraction', 0.35)
    aspect = kwargs.get('aspect', 25)

    classes = mp.Map("data/cmap_2d_significant/classes.tif")
    orography = mp.Map("data/correlation_p_wtd/correlation_p_wtd_15_downsampled_mean_10.tif")
    climate = mp.Map("data/climate/climate_downsampled_10_display.tif")
    landscape = mp.Map("data/landscape_classes/landscape_downsampled_10_display.tif")
    precipitation = mp.Map("data/precipitation/precipitation_downsampled_10.tif")
    tree_height = mp.Map("data/tree_height/tree_height_global_downsampled_10.tif")
    fapar = mp.Map("data/fapar/mean_fpar_downsampled_10.tif")
    corr_wtd_fapar = mp.Map("data/correlation_wtd_fapar/correlation_wtd_fapar_ge3_15_downsampled_mean_10.tif")
    corr_p_fapar = mp.Map("data/correlation_p_fapar/correlation_p_fapar_ge3_15_downsampled_mean_10.tif")
    wtd = mp.Map("data/wtd/wtd_downsampled_10.tif")

    f, ax = mp.subplots((3, 3), figsize=figsize)

    # Classes
    im = classes.plot(ind, basemap=True, coastline_kwargs={'linewidth': 0}, y_labels=37, x_labels=73, ax=ax[0])
    if titles:
        ax[0].set_title("Ecohydrological classes")
    if legend == 'colorbar':
        divider = axes_grid1.make_axes_locatable(ax[0])
        width = axes_grid1.axes_size.AxesY(ax[0], aspect=1. / aspect)
        pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
        current_ax = plt.gca()
        cax = divider.append_axes("right", size=width, pad=pad)
        cax.remove()

    # Precipitation
    bins = [0, 250, 500, 750, 1000, 1500, 2000, 3000, 10000]
    precipitation.plot_classified(ind, bins=bins, basemap=True, coastline_kwargs={'linewidth': 0}, y_labels=37,
                                  x_labels=73, ax=ax[1], title="$[mm]$" if legend_title else None,
                                  cmap="Blues", legend=legend, **kwargs)
    if titles:
        ax[1].set_title("Precipitation")

    # WTD
    bins = [0, 1, 2, 5, 10, 15, 20, 25, 35, 50, 100, 300]
    bins = bins[::-1]
    bins = [-i for i in bins]
    wtd.plot_classified(ind, bins=bins, basemap=True, coastline_kwargs={'linewidth': 0}, y_labels=37,
                        x_labels=73, ax=ax[2], title=f"$[{'-' * (legend == 'colorbar')}m]$" if legend_title else None,
                        cmap="Blues", legend=legend, **kwargs, return_image=True)

    if legend == 'colorbar':
        ax[2].images[0].colorbar.ax.set_yticklabels([-i for i in bins])

    if titles:
        ax[2].set_title("WTD")

    # FAPAR
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 100]
    fapar.plot_classified(ind, bins=bins, basemap=True, coastline_kwargs={'linewidth': 0}, y_labels=37,
                          x_labels=73, ax=ax[3], title="$[\%]}$" if legend_title else None,
                          cmap="Greens", legend=legend, **kwargs)
    if titles:
        ax[3].set_title("fAPAR")

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
    climate.plot_classified(ind, bins=bins, labels=labels, colors=cmap, suppress_warnings=True, mode='classes',
                            basemap=True, coastline_kwargs={'linewidth': 0}, y_labels=37, x_labels=73, ax=ax[4],
                            title="" if legend_title else None, legend=legend, clip_legend=True,
                            **kwargs)
    if titles:
        ax[4].set_title("Climate")

        # Landscape
    cmap = ['#004dac', '#729116', '#b1bc1d', '#e7de23', '#af9a15', '#785707', '#fff9f2']
    labels = ['Wetland and open water', 'Lowland', 'Undulating', 'Hilly', 'Low mountainous', 'Mountainous',
              'High mountainous']
    bins = [1, 2, 3, 4, 5, 6, 7]
    landscape.plot_classified(ind, bins=bins, labels=labels, basemap=True, colors=cmap,
                              coastline_kwargs={'linewidth': 0}, y_labels=37, x_labels=73, ax=ax[5],
                              mode='classes', legend=legend, **kwargs)
    if titles:
        ax[5].set_title("Landscape")

    # Correlation WTD and FAPAR
    bins = [-1, -0.5, -0.25, -0.11, 0.11, 0.25, 0.5, 1]
    corr_wtd_fapar.plot_classified(ind, bins=bins, labels=labels, basemap=True, coastline_kwargs={'linewidth': 0},
                                   y_labels=37, x_labels=73, ax=ax[6], legend=legend, cmap="RdYlBu", **kwargs)
    if titles:
        ax[6].set_title("Correlation WTD and fAPAR")

    # Correlation P and FAPAR
    corr_p_fapar.plot_classified(ind, bins=bins, labels=labels, basemap=True, coastline_kwargs={'linewidth': 0},
                                 y_labels=37, x_labels=73, ax=ax[7], legend=legend, cmap="RdYlBu", **kwargs)
    if titles:
        ax[7].set_title("Correlation P and fAPAR")

    # Orography
    orography.plot_classified(ind, bins=bins, labels=labels, basemap=True, coastline_kwargs={'linewidth': 0},
                              y_labels=37, x_labels=73, ax=ax[8], legend=legend, cmap="RdYlBu", **kwargs)
    if titles:
        ax[8].set_title("Correlation WTD and P")

    plt.tight_layout()

    if legend == 'colorbar':
        plt.subplots_adjust(wspace=0.17, hspace=-0.2)

    if not isinstance(save, type(None)):
        plt.savefig(save)

    plt.show()


def draw_thesis_location_zoom_full(ind, titles=True, figsize=(20, 20), save=None, legend_title=True, legend='colorbar',
                                   y_labels=181, x_labels=361, **kwargs):
    """
    Drawing array of nine maps, displaying the input data and results. Not meant for custom use.

    Parameters
    ----------
    ind : .
        look at mp.MapBase.get_pointer()
    titles : bool, optional
        insert the titles above the plots
    figsize : tuple, optional
        matplotlib equivalent figsize. The default is (20,20)
    save : str, optional
        export location
    legend_title : bool, optional
        print the title of the legend
    legend : ['colorbar', 'legend'], optional
        print a hovering legend or a colorbar, which is the default.
    y_labels : int, optional
        number of labels on the y axis. This number represents the y axis covering the whole world. To print a number
        and grid line every degree
    x_labels : int, optional
        number of labels on the x axis. This number represents the x axis covering the whole world. To print a number
        and grid line every degree
    kwargs : dict, optional,
        keyword arguments passed into the mp.MapRead.plot_classified() command
    """
    if isinstance(kwargs, type(None)):
        kwargs = {}

    if 'legend_title_pad' not in kwargs:
        kwargs.update({'legend_title_pad': 5})

    pad_fraction = kwargs.get('pad_fraction', 0.35)
    aspect = kwargs.get('aspect', 25)

    classes = mp.Map(
        "data/cmap_2d_significant/classes_full.tif")
    orography = mp.Map("data/correlation_p_wtd/correlation_p_wtd_15.tif")
    climate = mp.Map("data/climate/climate.tif")
    wtd_std = mp.Map("data/wtd/wtd_std_5.tif")
    landscape_map = mp.Map("data/landscape_classes/landscape_classes.tif")
    precipitation = mp.Map("data/precipitation/precipitation.tif")
    tree_height = mp.Map("data/tree_height/tree_height_global.tif")
    fapar = mp.Map("data/fapar/mean_fpar_reprojected.tif")
    corr_wtd_fapar = mp.Map("data/correlation_wtd_fapar/correlation_wtd_fapar_ge3_15.tif")
    corr_p_fapar = mp.Map("data/correlation_p_fapar/correlation_p_fapar_ge3_15.tif")
    wtd = mp.Map("data/wtd/wtd.tif")

    f, ax = mp.subplots((3, 3), figsize=figsize)

    # Classes
    im = classes.plot(ind, basemap=True, coastline_kwargs={'linewidth': 0}, y_labels=y_labels, x_labels=x_labels,
                      ax=ax[0])
    if titles:
        ax[0].set_title("Ecohydrological classes")
    if legend == 'colorbar':
        divider = axes_grid1.make_axes_locatable(ax[0])
        width = axes_grid1.axes_size.AxesY(ax[0], aspect=1. / aspect)
        pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
        current_ax = plt.gca()
        cax = divider.append_axes("right", size=width, pad=pad)
        cax.remove()

    # Precipitation
    # bins = [0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2500, 3000, 4000, 10000]
    bins = np.hstack([np.arange(0, 1000, 100), np.arange(1000, 3000, 250), 3000, 4000, 10000])
    precipitation.plot_classified(ind, bins=bins, basemap=True, coastline_kwargs={'linewidth': 0}, y_labels=y_labels,
                                  x_labels=x_labels, ax=ax[1], title="$[mm]$" if legend_title else None,
                                  cmap="Blues", legend=legend, suppress_warnings=True, **kwargs)
    if titles:
        ax[1].set_title("Precipitation")

    # WTD
    bins = [0, 1, 2, 5, 10, 15, 20, 25, 35, 50, 100, 200, 300, 1000]
    bins = bins[::-1]
    bins = [-i for i in bins]
    wtd.plot_classified(ind, bins=bins, basemap=True, coastline_kwargs={'linewidth': 0}, y_labels=y_labels,
                        x_labels=x_labels, ax=ax[2],
                        title=f"$[{'-' * (legend == 'colorbar')}m]$" if legend_title else None,
                        cmap="Blues", legend=legend, **kwargs, return_image=True)

    if legend == 'colorbar':
        ax[2].images[0].colorbar.ax.set_yticklabels([-i for i in bins])

    if titles:
        ax[2].set_title("WTD")

    # FAPAR
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 100]
    fapar.plot_classified(ind, bins=bins, basemap=True, coastline_kwargs={'linewidth': 0}, y_labels=y_labels,
                          x_labels=x_labels,
                          ax=ax[3], title="$[\%]}$" if legend_title else None, cmap="Greens", legend=legend, **kwargs)
    if titles:
        ax[3].set_title("fAPAR")

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
    climate.plot_classified(ind, bins=bins, labels=labels, colors=cmap, suppress_warnings=True, mode='classes',
                            basemap=True, coastline_kwargs={'linewidth': 0}, y_labels=y_labels, x_labels=x_labels,
                            ax=ax[4], title="" if legend_title else None, legend=legend, clip_legend=True, **kwargs)
    if titles:
        ax[4].set_title("Climate")

    # Landscape
    dem_cmap = np.vstack((mp.cmap_from_borders(("#0000b8", "#006310"), n=2, return_type="list"),
                          mp.cmap_from_borders(("#006310", "#faea00"), n=7, return_type="list"),
                          mp.cmap_from_borders(("#faea00", "#504100"), n=8, return_type="list"),
                          mp.cmap_from_borders(("#504100", "#ffffff"), n=2, return_type="list")))
    bins = np.round(np.hstack(((0,), np.logspace(0, np.log10(400), num=18))), 1)
    wtd_std.plot_classified(ind, bins=bins, basemap=True, cmap=ListedColormap(dem_cmap),
                            coastline_kwargs={'linewidth': 0}, y_labels=y_labels, x_labels=x_labels, ax=ax[5],
                            mode='continues', legend=legend, suppress_warnings=True, **kwargs)
    if titles:
        ax[5].set_title("Landscape")

    # Correlation WTD and FAPAR
    bins = [-1, -0.5, -0.25, -0.11, 0.11, 0.25, 0.5, 1]
    corr_wtd_fapar.plot_classified(ind, bins=bins, labels=labels, basemap=True, coastline_kwargs={'linewidth': 0},
                                   y_labels=y_labels, x_labels=x_labels, ax=ax[6], legend=legend, cmap="RdYlBu",
                                   **kwargs)
    if titles:
        ax[6].set_title("Correlation WTD and fAPAR")

    # Correlation P and FAPAR
    corr_p_fapar.plot_classified(ind, bins=bins, labels=labels, basemap=True, coastline_kwargs={'linewidth': 0},
                                 y_labels=y_labels, x_labels=x_labels, ax=ax[7], legend=legend, cmap="RdYlBu", **kwargs)
    if titles:
        ax[7].set_title("Correlation P and fAPAR")

    # Orography
    orography.plot_classified(ind, bins=bins, labels=labels, basemap=True, coastline_kwargs={'linewidth': 0},
                              y_labels=y_labels, x_labels=x_labels, ax=ax[8], legend=legend, cmap="RdYlBu", **kwargs)
    if titles:
        ax[8].set_title("Correlation WTD and P")

    plt.tight_layout()

    if legend == 'colorbar':
        plt.subplots_adjust(wspace=0.17, hspace=-0.2)

    if not isinstance(save, type(None)):
        plt.savefig(save)

    plt.show()


def draw_thesis_map(loc, colorbar=False, legend_2d=False, imshow_kwargs=None, classified=False,
                    classified_colors=None, classified_labels=None, classified_bins=None, classified_legend_bbox=None,
                    classified_legend_title=None, classified_to_classify=True, font=None, x_labels=13, y_labels=7,
                    dashes=[1, 5], insets=False, inset_coords=None, inset_size=[0.24, 0.24], inset_bottom_margin=0.015,
                    inset_locations=None, labels=None, ncol=1):
    """
    Plot map for thesis. Not intended for reuse.

    Parameters
    ----------
    loc : str
        Location of the map that will be drawn
    colorbar : bool, optional
        Add colorbar to the map. At the moment this will be to the right of the map, with a font that is too small.
        This needs to be fixed.
    legend_2d : bool, optional
        2D cmap plotted with the percentages of the classes written inside.
    imshow_kwargs : dict, optional
        Kwargs to be passed to plt.imshow() command
    classified : bool, optional
        Is the data classified data or does it need to be classified? The default is False.
    classified_colors : list, optional
        List of colors used when `classified` is True. The number of colors should be equal to the number of bins.
        If not provided colors are generated with the discrete_cmap function.
    classified_labels : list, optional
        List of labels that will be inserted in the legend when `classified` is True. The length of this list should
        be equal to the number of bins.
    classified_bins : list, optional
        List of bins that will be used to classify the data. If not provided it will use all unique values given in
        the data.
    classified_legend_bbox : tuple, optional
        Tuple containing the bbox location of the legend. If not provided it will be placed at the left bottom of the
        map.
    classified_legend_title : str, optional
        Title of the legend. Default is None.
    classified_to_classify : bool, optional
        parameter `to_classify` of `plot_classified_map_old`
    font : dict, optional
        Dictionary containing font information in a matplotlib format
    x_labels : int, optional
        Number of labels on the x-axis of the map. Default is 13 which corresponds to a line every 30 degrees.
    y_labels : int, optional
        Number of labels on the y-axis of the map. Default is 7 which corresponds to a line every 30 degrees.
    dashes : tuple, optional
        Dashes used for the parallels and meridians. It is a basemap parameter.
    insets : bool, optional
        Print insets in the map. Default is False
    inset_coords : list, optional
        List of lists containing the four coordinates of every inset. A default of four different locations is
        baked in.
    inset_size : tuple, optional
        Size of the insets in axes coordinates
    inset_bottom_margin : float, optional
        The distance of the insets from the bottom of the map. A default is given.
    inset_locations : tuple, optional
        Locations of the insets. This parameter gives complete control over the location and size of each inset.
        Default locations for the four `inset_coords` is baked in.
    labels : list, optional
        A list of names for the insets.
    ncol : int, optional
        columns of the legend, default is 1
    """
    M = mp.Map(loc)
    map_data = M[0]

    if isinstance(inset_coords, type(None)):
        inset_coords = ((-93, 29, -78, 44),  # Mississippi
                        (7, 35, 22, 50),  # Italy
                        # (70, 8, 85, 23),  # India
                        (15, -7, 30, 8),  # Congo
                        # (47, 50, 62, 65), # Russia
                        (142, -35, 158, -20))  # Australia

    if map_data.ndim == 2:
        if insets:
            inset_data = [M[inset_coords[i]] for i in range(len(inset_coords))]
    else:
        if insets:
            inset_data = [M[inset_coords[i]] for i in range(len(inset_coords))]

    bounds = [-180, -90, 180, 90]
    f, ax = plt.subplots(1, figsize=(40, 40))

    if isinstance(font, type(None)):
        font = {'fontsize': 20, 'horizontalalignment': 'center', 'verticalalignment': 'center'}
    else:
        font_temp = font
        font = {'fontsize': 20, 'horizontalalignment': 'center', 'verticalalignment': 'center'}
        font.update(font_temp)

    if isinstance(imshow_kwargs, type(None)):
        imshow_kwargs = {}

    main_map = mp.basemap(llcrnrlon=bounds[0], llcrnrlat=bounds[1], urcrnrlon=bounds[2], urcrnrlat=bounds[3], ax=ax)

    main_map.drawcoastlines(linewidth=0.14)
    main_map.drawparallels(np.linspace(-90, 90, num=y_labels), labels=[1, 0, 0, 0], dashes=dashes, fontdict=font)
    main_map.drawmeridians(np.linspace(-180, 180, num=x_labels), labels=[0, 0, 0, 1], dashes=dashes, fontdict=font)

    if classified:
        if isinstance(classified_bins, type(None)):
            classified_bins = np.unique(map_data[~np.isnan(map_data)])

        if not isinstance(classified_colors, type(None)):
            assert len(classified_bins) + classified_to_classify == len(
                classified_colors), f"length of bins and colors don't match\n" \
                                    f"bins: {len(classified_bins)}\n" \
                                    f"colors {len(classified_colors)}"
        else:
            classified_colors = mp.cmap_discrete(len(classified_bins), return_type="list")

        if not isinstance(classified_labels, type(None)):
            assert len(classified_bins) + classified_to_classify == len(
                classified_labels), f"length of bins and labels don't match, " \
                                    f"{len(classified_bins)}, {len(classified_labels)}"
        else:
            classified_labels = list(classified_bins)

        legend_patches = [Patch(facecolor=icolor, label=label, edgecolor="lightgrey")
                          for icolor, label in zip(classified_colors, classified_labels)]

        mp.plot_classified_map(map_data, bins=classified_bins, colors=classified_colors, labels=classified_labels,
                               legend=False, suppress_warnings=True, ax=main_map, to_classify=classified_to_classify)

        ax.legend(handles=legend_patches,
                  facecolor="white",
                  edgecolor="lightgrey",
                  loc="lower left",
                  bbox_to_anchor=classified_legend_bbox,
                  fontsize=font['fontsize'],
                  title=classified_legend_title,
                  title_fontsize=font['fontsize'],
                  ncol=ncol)

        colorbar = False
    else:
        im = main_map.imshow(map_data, origin="upper", **imshow_kwargs)

    if isinstance(inset_locations, type(None)):
        inset_locations = [(0.30, inset_bottom_margin, *inset_size),
                           (0.45, inset_bottom_margin, *inset_size),
                           (0.60, inset_bottom_margin, *inset_size),
                           (0.75, inset_bottom_margin, *inset_size)]

    if isinstance(labels, type(None)):
        labels = [chr(65 + i) for i in range(len(inset_locations))]

    if insets:
        for i in range(len(inset_coords)):
            c = inset_coords[i]
            left, bottom, right, top = c

            axins = inset_axes(ax, width="100%", height="100%", bbox_to_anchor=inset_locations[i],
                               bbox_transform=ax.transAxes)
            inset_map = mp.basemap(llcrnrlon=left, llcrnrlat=bottom, urcrnrlon=right, urcrnrlat=top, ax=axins,
                                   resolution='c')  # !! change to f
            inset_map.drawcoastlines(linewidth=0.14)
            ax.indicate_inset_zoom(axins, linewidth=3, edgecolor="black")

            if classified:
                map_data = inset_data[i]
                mp.plot_classified_map(map_data, bins=classified_bins, colors=classified_colors,
                                       labels=classified_labels,
                                       legend=False, ax=inset_map, suppress_warnings=True,
                                       to_classify=classified_to_classify)

            else:
                inset_map.imshow(inset_data[i], origin='upper', **imshow_kwargs)

            text_location_x = (left + right) / 2
            text_location_y = (top + bottom) / 2
            t = ax.text(text_location_x, text_location_y, labels[i], fontdict=font)
            t.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='grey'))

            text_location_x = inset_locations[i][0] + inset_locations[i][2] / 2
            text_location_y = inset_locations[i][1] + inset_locations[i][3] / 2 + inset_size[0] / 2 + 0.01
            t = ax.text(text_location_x, text_location_y, labels[i], fontdict=font, transform=ax.transAxes)
            t.set_bbox(dict(facecolor='white', alpha=0.6, edgecolor='grey'))

    if legend_2d:
        map_classes = mp.Map("data/cmap_2d_significant/classes_serial.tif")
        data = map_classes[0]
        data[np.isnan(data)] = -1
        bins, vals = np.unique(data, return_counts=True)
        bins = bins[1:]
        vals = vals[1:]
        vals = vals / vals.sum() * 100

        axins = inset_axes(ax, width="100%", height="100%",
                           bbox_to_anchor=(-0.01, 0.25, 0.26, 0.26),
                           bbox_transform=ax.transAxes)
        cmap = mp.cmap_2d((3, 3), diverging=False, alpha=0.5, diverging_alpha=0.25, flip=False, rotate=0)
        cmap[1, 1, :] = (0.9, 0.9, 0.9)

        axins.imshow(cmap, origin='lower')
        axins.set_xticks([0, 1, 2], minor=False)
        axins.set_yticks([0, 1, 2], minor=False)
        axins.set_xticklabels([u'\u2014', "0", "+"], minor=False, fontdict=font)
        axins.set_yticklabels([u'\u2014', "0", "+"], minor=False, fontdict=font)
        axins.set_xlabel("Correlation WTD and fAPAR", fontdict=font, labelpad=20)
        axins.set_ylabel("Correlation P and fAPAR", fontdict=font, labelpad=20)
        axins.tick_params(axis='both', which='both', length=0, pad=20)

        for i in range(3):
            for j in range(3):
                t = axins.text(i, j, f'{vals[i + 3 * j]:1.0f}', fontdict=font)
                t.set_bbox(dict(facecolor='white', alpha=0.6, edgecolor='grey'))

    if colorbar:
        mp.add_colorbar(im, pad_fraction=2)

    return ax

import thesis
import geomappy as mp
import numpy as np

CLASSES = "data/cmap_2d_significant/classes_serial.tif"
M_classes = mp.Raster(CLASSES)
classes_data = M_classes[0].astype(np.int8) + 1
profile = M_classes.profile
profile['dtype'] = np.int8
profile['nodata'] = 0
profile['compress'] = 'LZW'
mp.Raster("data/export/classes.tif", mode='w', profile=profile)[0] = classes_data

LANDSCAPE = "data/landscape_classes/landscape_classes.tif"
M_landscape = mp.Raster(LANDSCAPE)
landscape_data = M_landscape[0].astype(np.int8)
profile = M_landscape.profile
profile['dtype'] = np.int8
profile['nodata'] = 0
profile['compress'] = 'LZW'
mp.Raster("data/export/landscape.tif", mode='w', profile=profile)[0] = landscape_data

mp.Raster.close()
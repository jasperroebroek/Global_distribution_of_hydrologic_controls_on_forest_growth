import geomappy as mp

M_p = p = mp.Map("data/precipitation/precipitation.tif")

pet = mp.Map("data/pet/et0_yr.tif", fill_value=0)[M_p.bounds]
p = M_p[0]

M_p_pet = mp.Map("data/pet/p_pet.tif", mode='w', ref_map="data/precipitation/precipitation.tif", overwrite=True)
M_p_pet[0] = p/pet

mp.Map.close()

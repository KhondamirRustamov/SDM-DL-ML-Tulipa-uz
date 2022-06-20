import numpy as np
import os

# download worldclim data
list_of_vectors = os.listdir('climate')
list_of_maps = [np.load('climate/'+i) for i in list_of_vectors]


def get_new_matrix(worldclim, extent, min_old):
    # function to crop worldclim map
    x_new = 21600 + round(min_old[0] / 180 * 21600)
    y_new = 10800 - round(min_old[1] / 90 * 10800)

    x_new_min = (21600 + round(extent[0] / 180 * 21600)) - x_new
    x_new_max = (21600 + round(extent[1] / 180 * 21600)) - x_new
    y_new_min = (10800 - round(extent[2] / 90 * 10800)) - y_new
    y_new_max = (10800 - round(extent[3] / 90 * 10800)) - y_new
    return worldclim[y_new_min:y_new_max, x_new_min:x_new_max]


extent = [64, 75, 44, 36]  # define the cropping box
for i, z in zip(list_of_maps, range(len(list_of_maps))):
    new_matrix = np.array(get_new_matrix(i, extent, [55, 47]))
    np.save('new_worldclim/BIO'+str(z+1)+'.npy', new_matrix)

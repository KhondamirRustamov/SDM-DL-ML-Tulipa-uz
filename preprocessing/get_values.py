import pandas as pd
import numpy as np
import os

# download present points
present = pd.read_csv('data/present_points.csv')
present = pd.DataFrame(present).sample(frac=1).reset_index(drop=True)
present_array = np.array([present['latitude'], present['longitude']])


# download pseudo-absence points
absence = pd.read_csv('data/pseudo_absence_rsep.csv')
absence = pd.DataFrame(absence).sample(frac=1).reset_index(drop=True)
absence_array = np.array([absence['latitude'], absence['longitude']])


# download worldclim data
list_of_vectors = os.listdir('worldclim')
list_of_maps = [np.load('worldclim/'+i) for i in list_of_vectors]
worldclim_concatanate = np.concatenate((list_of_maps)).reshape((20, 960, 1320))


def get_values_from_matrix(worldclim, pa, min_x, min_y, presence=True):  # get worldclim data for all points
    new_matrix = []
    y_hat = []
    min_x_new = 21600 + round(min_x / 180 * 21600)
    min_y_new = 10800 - round(min_y / 90 * 10800)
    print(min_x_new, min_y_new)
    for i in range(len(pa[0])):
        x_new = (21600 + round(pa[1, i] / 180 * 21600)) - min_x_new
        y_new = (10800 - round(pa[0, i] / 90 * 10800)) - min_y_new
        new_matrix.append(worldclim[:, y_new, x_new])
        if presence is True:
            y_hat.append(1)
        else:
            y_hat.append(0)
    return new_matrix, y_hat


def k_fold(array, percentage1, percentage2):  # create 70%, 10%, 20% training, validation and testing sets
    return array[:(int(len(array)*percentage1/100))], array[(int(len(array)*percentage1/100)):(int(len(array)*percentage2/100))], array[(int(len(array)*percentage2/100)):]


x_pres, y_pres = get_values_from_matrix(worldclim_concatanate, present_array, 64, 44)
x_abs, y_abs = get_values_from_matrix(worldclim_concatanate, absence_array, 64, 44, presence=False)

x_pres_train, x_pres_val, x_pres_test = k_fold(x_pres, 70, 80)
y_pres_train, y_pres_val, y_pres_test = k_fold(y_pres, 70, 80)
x_abs_train, x_abs_val, x_abs_test = k_fold(x_abs, 70, 80)
y_abs_train, y_abs_val, y_abs_test = k_fold(y_abs, 70, 80)


# concatenate present and pseudo-absence records in training, validation and test sets respectevily
x_train = np.concatenate((x_pres_train, x_abs_train))
y_train = np.concatenate((y_pres_train, y_abs_train))
x_val = np.concatenate((x_pres_val, x_abs_val))
y_val = np.concatenate((y_pres_val, y_abs_val))
x_test = np.concatenate((x_pres_test, x_abs_test))
y_test = np.concatenate((y_pres_test, y_abs_test))


# save results as numpy matrices
np.save('data/species/rsep/x_train.npy', x_train)
np.save('data/species/rsep/y_train.npy', y_train)
np.save('data/species/rsep/x_val.npy', x_val)
np.save('data/species/rsep/y_val.npy', y_val)
np.save('data/species/rsep/x_test.npy', x_test)
np.save('data/species/rsep/y_test.npy', y_test)

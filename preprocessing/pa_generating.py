import pandas as pd
import numpy as np
import random
from sklearn import svm
from PIL import Image


# download present points of species
present = pd.read_csv('data/present_points.csv')
present = pd.DataFrame(present)
present_array = np.array([present['latitude'],present['longitude']])

y_array = np.array(present['latitude'].to_list())
x_array = np.array(present['longitude'].to_list())

# download worldclim data
worldclim = np.load('data/worldclim_new.npy')


def get_values_from_matrix(worldclim, pa, min_x, min_y):
    # get worldclim data to selected points
    new_matrix = []
    min_x_new = 21600 + round(min_x / 180 * 21600)
    min_y_new = 10800 - round(min_y / 90 * 10800)
    for i in range(len(pa[0])):
        x_new = (21600 + round(pa[1, i] / 180 * 21600)) - min_x_new
        y_new = (10800 - round(pa[0, i] / 90 * 10800)) - min_y_new
        new_matrix.append(worldclim[:, y_new, x_new])
    return new_matrix


def get_values(worldclim, pa, min_x, min_y):
    # get worldclim data to selected point
    new_matrix = []
    min_x_new = 21600 + round(min_x / 180 * 21600)
    min_y_new = 10800 - round(min_y / 90 * 10800)
    x_new = (21600 + round(pa[0] / 180 * 21600)) - min_x_new
    y_new = (10800 - round(pa[1] / 90 * 10800)) - min_y_new
    new_matrix.append(worldclim[:, y_new, x_new])
    return new_matrix


# get wordclim data for all presented points
x_pres = get_values_from_matrix(worldclim, present_array, 55, 47)


# define and train OCSVM on present points and worldclim data
clf = svm.OneClassSVM(nu=0.3, kernel="rbf", gamma=0.05)
clf.fit(x_pres)

# create a map of OCSVM prediction
wordclim_new = worldclim.T.reshape((3960000, 19))
prediction = clf.predict(wordclim_new)
prediction = np.array([1 if x == 1 else 0.5 for x in prediction]).reshape((3000, 1320)).T

new_matrix = (prediction * 255).astype(np.uint8)
img = Image.fromarray(new_matrix, mode='L')
img.save('pseudo_absence_rsep.jpg')


def find_random(x1, x2, y1, y2):  # RSEP pseudo_absence selecting strategy using OCSVM prediction
    random_x = random.uniform(x1, x2)
    random_y = random.uniform(y1, y2)
    random_array = np.array([random_x, random_y])
    random_data = get_values(worldclim, random_array, 55, 47)
    detector = clf.predict(random_data)
    if detector != -1:
        while detector != -1:
            random_x = random.uniform(x1, x2)
            random_y = random.uniform(y1, y2)
            random_array = np.array([random_x, random_y])
            random_data = get_values(worldclim, random_array, 55, 47)
            detector = clf.predict(random_data)
    else:
        random_x, random_y = random_x, random_y
    return random_x, random_y


def random_float_number(x1, x2, y1, y2, array_x, array_y):  # RSEB pseudo_absence selecting strategy
    random_x = random.uniform(x1, x2)
    random_y = random.uniform(y1, y2)
    if min(abs(array_x-random_x)) <= 0.0833 and min(abs(array_y-random_y)) <= 0.0833:  # set 10 km buffer zone
        while min(abs(array_x-random_x)) <= 0.0833 and min(abs(array_y-random_y)) <= 0.0833:
            random_x = random.uniform(x1, x2)
            random_y = random.uniform(y1, y2)
    else:
        random_x = random_x
        random_y = random_y
    return random_x, random_y


def random_coordinates(extent):  # RS pseudo_absence selecting strategy
    random_x = random.uniform(extent[0], extent[1])
    random_y = random.uniform(extent[2], extent[3])
    return random_x, random_y


extent = [65, 74, 43, 37]
x_pseudo_rs=[]
y_pseudo_rs=[]
x_pseudo_rseb=[]
y_pseudo_rseb=[]
x_pseudo_rsep=[]
y_pseudo_rsep=[]

# RSEP pseudo_absence sampling
for i in range(len(x_pres)*10):
    x_new, y_new = find_random(65.000, 74.000, 37.000, 43.000)
    x_pseudo_rsep.append(x_new)
    y_pseudo_rsep.append(y_new)

# save RSEP sampling in .csv file
species = ['pseudo_absence' for i in range(len(x_pseudo_rsep))]
dict_rsep = {
    'species': species,
    'longitude': x_pseudo_rsep,
    'latitude': y_pseudo_rsep
}
df_rsep = pd.DataFrame(dict_rsep)
df_rsep.to_csv('unused/pseudo_absence_rsep.csv', index=False)


# RSEB pseudo_absence sampling
for i in range(len(x_pres)*10):
    x_new, y_new = random_float_number(65.000, 74.000, 37.000, 43.000, x_array, y_array)
    x_pseudo_rseb.append(x_new)
    y_pseudo_rseb.append(y_new)

# save RSEB sampling in .csv file
dict_rseb = {
    'species': species,
    'longitude': x_pseudo_rseb,
    'latitude': y_pseudo_rseb
}
df_rseb = pd.DataFrame(dict_rseb)
df_rseb.to_csv('unused/pseudo_absence_rseb.csv', index=False)


# RS pseudo_absence sampling
for i in range(len(x_pres)*10):
    x_new, y_new = random_coordinates(extent)
    x_pseudo_rs.append(x_new)
    y_pseudo_rs.append(y_new)

# save RS sampling in .csv file
dict_rs = {
    'species': species,
    'longitude': x_pseudo_rs,
    'latitude': y_pseudo_rs
}
df_rs = pd.DataFrame(dict_rs)
df_rs.to_csv('unused/pseudo_absence_rs.csv', index=False)
print('saved')
import numpy as np
import tensorflow as tf
from DNN_model import create_model
from estimator import estimator
from sklearn.preprocessing import StandardScaler
from PIL import Image
import os


Image.MAX_IMAGE_PIXELS = 933120000

# download wordclim data
list_of_vectors = os.listdir('worldclim')
list_of_maps = [np.load('worldclim/'+i) for i in list_of_vectors]
worldclim = np.concatenate((list_of_maps)).reshape((20, 960, 1320)).T.reshape((1267200, 20))


# define training set
x_train = np.load('data/species/rsep/x_train.npy')
y_train1 = np.load('data/species/rsep/y_train.npy')
y_train = np.array([[0, 1] if x == 1 else [1, 0] for x in y_train1])

# validation set
x_val = np.load('data/species/rsep/x_val.npy')
y_val1 = np.load('data/species/rsep/y_val.npy')
y_val = np.array([[0, 1] if x == 1 else [1, 0] for x in y_val1])

# test set
x_test = np.load('data/species/rsep/x_test.npy')
y_test1 = np.load('data/species/rsep/y_test.npy')
y_test = np.array([[0, 1] if x == 1 else [1, 0] for x in y_test1])


# define the number of positive and negative records to set class weights
pos = sum(y_train1)+sum(y_test1)+sum(y_val1)
neg = len(y_train1)+len(y_test1)+len(y_val1)-pos
total = pos+neg

# define a batch size
BATCH_SIZE = int(total/10)


# transform data using Sklearn
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
worldclim = scaler.transform(worldclim)
x_train = np.clip(x_train, -10, 10)
x_test = np.clip(x_test, -10, 10)
worldclim =np.clip(worldclim, -10, 10)


# set special class weights for unbalanced data
weight_for_0 = (1 / neg) * (total / 2.0)
weight_for_1 = (1 / pos) * (total / 2.0)
class_weight = {0: weight_for_0, 1: weight_for_1}


# define early stopping strategy
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_prc',
    verbose=1,
    patience=10,
    mode='max',
    restore_best_weights=True)


# model creating and training
model = create_model()
model.fit(x_train, y_train,
          epochs=1000,
          batch_size=BATCH_SIZE,
          class_weight=class_weight,
          callbacks=[early_stopping],
          validation_data=(x_val, y_val),)
prediction = model.predict(x_test)


# biased test data to estimation of model performance
biased = np.array([np.argmax(x) for x in y_test])

# calculate the performance of the model
AUC, TSS, Kappa, thre = estimator(biased, prediction)
print(AUC, TSS, Kappa, thre)


# create the prediction map
new_map = model.predict(worldclim)
new = np.array([0 if i[1] > thre else 255 for i in new_map]).reshape((1320, 960)).T
new_matrix = new.astype(np.uint8)
img = Image.fromarray(new_matrix, mode='L')
img.save('species_prediction_map.tif')



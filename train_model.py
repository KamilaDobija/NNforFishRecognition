import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten
# for dealing with randomness
from tensorflow import set_random_seed, ConfigProto, Session, get_default_graph
import os
import random
from keras import backend as K

# deal with randomness

# Exemplary seed value
seed_value = 0
# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
os.environ['PYTHONHASHSEED'] = str(seed_value)
# 2. Set the `python` built-in pseudo-random generator at a fixed value
random.seed(seed_value)
# 3. Set the `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)
# 4. Set the `tensorflow` pseudo-random generator at a fixed value
set_random_seed(seed_value)
# 5. Configure a new global `tensorflow` session
session_conf = ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = Session(graph=get_default_graph(), config=session_conf)
K.set_session(sess)

# load dataset
dataset = np.load('dataset_fish.npy', allow_pickle=True)
img_size = 200

# prepare train set and test set (x - img, y - label)
# 730 images in train set, 180 images in test set, 3 images left out for prediction purposes
x_train = np.array([x[0] for x in dataset[:730]]).reshape(-1, img_size, img_size, 1)
y_train = np.array([x[1] for x in dataset[:730]])
x_test = np.array([x[0] for x in dataset[730:910]]).reshape(-1, img_size, img_size, 1)
y_test = np.array([x[1] for x in dataset[730:910]])


# training model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(img_size, img_size, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
# model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=64, epochs=4)

# performance evaluation
loss, acc = model.evaluate(x_test[:180], y_test[:180], verbose=0)
print(acc * 100)

# save model
# model.save("model_for_fish.h5")

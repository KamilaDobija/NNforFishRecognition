import numpy as np
from keras.models import load_model

img_size = 200

# load images to predict output
predict_dataset = np.load('dataset_fish.npy', allow_pickle=True)
img_to_predict = np.array([x[0] for x in predict_dataset[910:]]).reshape(-1, img_size, img_size, 1)
img_label = np.array([x[1] for x in predict_dataset[910:]])

# load saved training model
model = load_model("model_for_fish.h5")
# model.summary()

# predict 3 fish
prediction = model.predict(img_to_predict, verbose=1)
print(prediction)
print(img_label)

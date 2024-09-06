import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Input, Dense
from keras.utils import to_categorical

(img_train, lab_train), (img_test, lab_test) = mnist.load_data()

train_img = img_train / 255
test_img = img_test / 255

train_lab = to_categorical(lab_train, num_classes=10)
test_lab = to_categorical(lab_test, num_classes=10)

train_samples, rows_img, cols_imgs = train_img.shape
train_img = train_img.reshape(train_samples, rows_img * cols_imgs)

model = Sequential([
    Input(shape = (rows_img * cols_imgs,)), # Input Layer
    Dense(256, activation='relu'), # Hidden Layer
    Dense(10, activation='softmax') # Output Layer
])

model.compile(optimizer='adam', loss='categorical_crossentropy')

model.fit(train_img, train_lab, epochs=10, batch_size=32)

test_samples, _, _= test_img.shape
test_img = test_img.reshape(test_samples, rows_img * cols_imgs)

eval = model.evaluate(test_img, test_lab)

pred = model.predict(test_img[:10, :])
pred_i = pred.argmax(1)

print(pred_i)
print(lab_test[:10])


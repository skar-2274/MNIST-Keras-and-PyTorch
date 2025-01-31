import numpy as np
import os
from tensorflow import keras
from keras.datasets import fashion_mnist
from keras.models import Sequential, load_model
from keras.layers import Input, Dense, Dropout
from keras.utils import to_categorical
import time

# Runtime
start_time = time.time()

# Load training and test data
(img_train, lab_train), (img_test, lab_test) = fashion_mnist.load_data()

train_img = img_train.astype('float32') / 255
test_img = img_test.astype('float32') / 255

train_lab = to_categorical(lab_train, num_classes = 10)
test_lab = to_categorical(lab_test, num_classes = 10)

train_sample, rows_img, cols_imgs = train_img.shape
train_img = train_img.reshape(train_sample, rows_img * cols_imgs)

# Create model with conditions
model_path = "fashion_mnist_model.h5"

if os.path.exists(model_path):
    print("Loading pre-trained model...")
    model = load_model(model_path)

else:
    print("Training new model")
    model = Sequential([
        Input(shape = (rows_img * cols_imgs)), # Input Layer
        Dense(512, activation='relu'), # Hidden layer
        Dropout(0.5), # Half the neurons dropout for regularisation
        Dense(256, activation='relu'), # Second Hidden Layer
        Dense(10, activation='softmax') # Output Layer
    ])

    # Compile, fit and save model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_img, train_lab, epochs=10, batch_size=32, validation_split=0.2)

    model.save(model_path)
    print(f"Model saved to {model_path}")

# Shape test data
test_samples, _, _ = test_img.shape
test_img = test_img.reshape(test_samples, rows_img * cols_imgs)

# Evaluate test data
eval = model.evaluate(test_img, test_lab)
print(f"Test Loss: {eval[0]}")
print(f"Test Accuracy: {eval[1]}")

# Class labels
class_labels = {
    0: "T-shirt/Top",
    1: "Trousers",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneakers",
    8: "Bag",
    9: "Ankle Boot"
}

# Prediction
pred = model.predict(test_img[:10, :])
pred_i = pred.argmax(1)

for i, pred_class in enumerate(pred_i):
    print(f"\nImage {i+1}: Predicted -> {class_labels[pred_class]}, Actual -> {class_labels[test_lab[i].argmax()]}")

# Measure runtime
end_time = time.time()
total_time = end_time - start_time

print(f"\nRuntime: {total_time:.2f} seconds")

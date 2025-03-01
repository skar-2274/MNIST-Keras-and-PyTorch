import numpy as np
import os
from tensorflow import keras
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Input, Dense, Dropout
from keras.utils import to_categorical
import time

# Runtime
start_time = time.time()

# Load training and test data
def data():
    (img_train, lab_train), (img_test, lab_test) = mnist.load_data()

    # Normalises values to 0's and 1's
    train_img = img_train.astype('float32')  / 255
    test_img = img_test.astype('float32')  / 255

    train_lab = to_categorical(lab_train, num_classes=10)
    test_lab = to_categorical(lab_test, num_classes=10)

    train_samples, rows_img, cols_imgs = train_img.shape
    train_img = train_img.reshape(train_samples, rows_img * cols_imgs)

    test_samples, _, _= test_img.shape
    test_img = test_img.reshape(test_samples, rows_img * cols_imgs)

    return train_img, train_lab, test_img, test_lab

# Create model with conditions
def build_model(input_shape):
    model = Sequential([
        Input(shape = (input_shape)), # Input Layer
        Dense(512, activation='relu'), # First Hidden Layer
        Dropout(0.5), # Half the neurons dropout for regularisation
        Dense(256, activation='relu'), # Second Hidden Layer
        Dense(10, activation='softmax') # Output Layer
    ])

    # Compile, fit and save model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Train Model
def train_model(model, train_img, train_lab, model_path):
    print("Training new model")
    model.fit(train_img, train_lab, epochs=10, batch_size=32, validation_split=0.2)
    model.save(model_path)
    print(f"Model saved to {model_path}")

# Evaluate model
def evaluate_model(model, test_img, test_lab):
    eval = model.evaluate(test_img, test_lab)
    print(f"Test Loss: {eval[0]}")
    print(f"Test Accuracy: {eval[1]}")

# Prediction
def prediction(model, test_img, test_lab, num_samples=10, N=20):
    pred = model.predict(test_img[:N, :])
    pred_i = pred.argmax(1)

    print("\nPredictions:", pred_i)
    print("Actual Values:", test_lab[:N].argmax(axis=1))

# Main function
def main():
    model_path = "mnist_model.h5"

    train_img, train_lab, test_img, test_lab = data()

    if os.path.exists(model_path):
        print("Loading pre-trained model...")
        model = load_model(model_path)

    else:
        model = build_model(train_img.shape[1])
        train_model(model, train_img, train_lab, model_path)

    evaluate_model(model,test_img, test_lab)
    prediction(model, test_img, test_lab)

if __name__ == "__main__":
    main()

# Measure runtime
end_time = time.time()
total_time = end_time - start_time

print(f"\nRuntime: {total_time:.2f} seconds")

## this won't work here  look 

## to run it on colab visit https://colab.research.google.com/drive/1N5ueLF2phDZW-PvrCWFY1jq__9zO5r3X#scrollTo=7wwA3eapWxKD
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

def main():
    # -------------------------
    # 1. Load and preprocess data
    # -------------------------
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Reshape to (samples, height, width, channels) and normalize
    X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

    # One-hot encode labels
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # -------------------------
    # 2. Build the CNN model
    # -------------------------
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        MaxPooling2D((2,2)),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])

    # -------------------------
    # 3. Compile the model
    # -------------------------
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # -------------------------
    # 4. Train the model
    # -------------------------
    history = model.fit(
        X_train, y_train,
        epochs=5,            # Increase for better accuracy
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )

    # -------------------------
    # 5. Evaluate the model
    # -------------------------
    loss, accuracy = model.evaluate(X_test, y_test)
    print("Test Accuracy:", accuracy)

    # -------------------------
    # 6. Plot training history
    # -------------------------
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='val')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training vs Validation Accuracy')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()

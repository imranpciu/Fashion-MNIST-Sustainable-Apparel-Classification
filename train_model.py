import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.utils import to_categorical

try:
    # Load Fashion MNIST training data from CSV file
    train_data = pd.read_csv("archive/fashion-mnist_train.csv")
    train_images = train_data.drop(columns=['label']).values.reshape((-1, 28, 28, 1)).astype('float32') / 255
    train_labels = to_categorical(train_data['label'].values, num_classes=10)

    # Explore dataset - EDA
    class_counts = pd.Series(train_data['label']).value_counts()
    print("Class distribution:")
    print(class_counts)

    # Visualize sample images
    plt.imshow(train_images[0].reshape(28, 28))
    plt.show()

    # Build and train the model
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28, 1)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=10, batch_size=64, validation_split=0.2)

    # Save the trained model
    model.save('fashion_mnist_model.h5')

    print("Training completed successfully.")

except Exception as e:
    print(f'Error: {e}')




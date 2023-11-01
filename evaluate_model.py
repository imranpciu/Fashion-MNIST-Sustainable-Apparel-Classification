import sys
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.utils import to_categorical
from io import StringIO

# Specify the path to the evaluation CSV file
evaluation_data_path = "archive/fashion-mnist_test.csv"

try:
    # Load the trained model
    model = load_model('fashion_mnist_model.h5')

    # Load evaluation data from CSV file
    evaluation_data = pd.read_csv(evaluation_data_path)

    # Extract class labels and pixel values from evaluation data
    evaluation_labels = evaluation_data['label'].values  # Assuming the column name for labels is 'label'
    evaluation_images = evaluation_data.drop(columns=['label']).values  # Pixel values (excluding the 'label' column)

    # Reshape and normalize pixel values
    evaluation_images = evaluation_images.reshape((-1, 28, 28, 1)).astype('float32') / 255

    # Convert labels to categorical format
    evaluation_labels_categorical = to_categorical(evaluation_labels, num_classes=10)

    # Evaluate the model on the evaluation data
    evaluation_results = model.evaluate(evaluation_images, evaluation_labels_categorical)

    # Save results to output.txt
    with open('output.txt', 'w') as f:
        f.write(f'Model Architecture:\n')
        model.summary(print_fn=lambda x: f.write(x + '\n'))  # Write model summary to output.txt
        f.write(f'\nEvaluation Accuracy: {evaluation_results[1]:.4f}\n')  # Format accuracy value with 4 decimal places
        f.write('Additional insights or observations: EDA and analysis were performed during training.\n')

except Exception as e:
    # Handle errors gracefully and provide appropriate error message
    with open('output.txt', 'w') as f:
        f.write(f'Error: {e}\n')


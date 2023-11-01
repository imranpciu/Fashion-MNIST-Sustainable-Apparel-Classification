# Fashion MNIST Sustainable Apparel Classification

## Overview

This project aims to develop an AI solution for classifying sustainable apparel products using the Fashion MNIST dataset. The objective is to align with our company's vision by accurately categorizing sustainable clothing items. The project involves data analysis, model development, and exploring ways to enhance human-in-the-loop efficiency.

## Dataset

The dataset utilized for this project is the Fashion MNIST dataset, comprising grayscale images of various apparel products. You can access the dataset [here](https://www.kaggle.com/datasets/zalando-research/fashionmnist/data).

## Task Description

### 1. Data Analysis

- Explored the dataset to comprehend class distribution and image characteristics.
- Documented insights and patterns observed during exploration.

### 2. Model Development

- Designed and trained a neural network model to classify apparel products based on the dataset.
- Utilized a neural network architecture with layers including Flatten, Dense (128 units, ReLU activation), and Dense (10 units, softmax activation) to categorize items into 10 classes.

### 3. Human-in-the-Loop

- Proposed a method to enhance human-in-the-loop efficiency by combining human expertise with the model's predictions to improve accuracy. The detailed implementation is not provided in these scripts.

## Scripts Description

### `train_model.py`

- Loaded Fashion MNIST training data from a CSV file.
- Explored the dataset, visualized sample images, and performed exploratory data analysis (EDA).
- Preprocessed the data, normalized pixel values, and converted labels to categorical format.
- Built and trained a neural network model.
- Saved the trained model as `fashion_mnist_model.h5`.

### `evaluate_model.py`

- Loaded the trained model.
- Loaded evaluation data from a CSV file.
- Reshaped and normalized pixel values for evaluation.
- Evaluated the model on the evaluation data.
- Generated `output.txt` containing model architecture, evaluation accuracy, and additional insights (EDA and analysis performed during training).

## How to Run

1. Ensure Python is installed on your system.
2. Clone this repository to your local machine.
3. Navigate to the project directory using the command line.
4. Install virtual enviroment using **(Any Python version)**:
     ```bash
     pipenv --python 3.9.12
     ```
5. Start virtual environment using:
     ```bash
     pipenv shell
     ```
6. Install the required Python packages using pip:
     ```bash
      pip install -r requirements.txt
     ```
7. Run the script to train the model:
   ```bash
     train_model.py
   ```
> This script will perform data analysis, train the neural network, and save the trained model as **fashion_mnist_model.h5**.

9. After training, run the script to evaluate the model:
     ```bash
    evaluate_model.py
    ```
> This script will load the trained model, evaluate it on the evaluation data, and generate `output.txt` containing model architecture, evaluation accuracy, and additional insights.

## Output

- `fashion_mnist_model.h5`: Trained neural network model.
- `output.txt`: Evaluation results, including model architecture summary, evaluation accuracy, and any additional insights.

## Notes

- Ensure the evaluation data file (`fashion-mnist_test.csv`) is placed in the appropriate location before running the evaluation script.





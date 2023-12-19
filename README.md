# Convolutional Neural Network (CNN) for Image Classification

## Overview

This repository contains code for training a Convolutional Neural Network (CNN) for image classification. The CNN is designed to classify images into predefined classes such as buildings, forest, glacier, mountain, sea, and street.

## Contents

1. **Data**

   - The dataset is organized into three sets: training set, test set, and prediction set.
   - Directories:
     - `seg_train/`: Contains the training images organized into subdirectories for each class.
     - `seg_test/`: Contains the test images organized similarly to the training set.
     - `seg_pred/`: Contains images for prediction without class labels.
     
2. **Code Files**

   - `data_preprocessing.py`: Contains functions for reading image names, checking class imbalance, and plotting sampled images.
   - `model_training.py`: Defines and trains the CNN model.
   - `evaluation.py`: Includes functions for evaluating the trained model, plotting confusion matrix, and making predictions on new images.

3. **Instructions**

   - Run `data_preprocessing.py` to prepare the data, check class imbalance, and visualize sampled images.
   - If training the model (`train_model` set to `True`), run `model_training.py`.
   - If using a pre-trained model, update the paths in `model_training.py` to load the existing model weights.

4. **Requirements**

   - Python 3.x
   - TensorFlow (install with `pip install tensorflow`)
   - NumPy, Pandas, Matplotlib, Seaborn (install with `pip install numpy pandas matplotlib seaborn`)

5. **Model Architecture**

   - The CNN architecture is designed for image classification with layers for rescaling, data augmentation, convolution, batch normalization, dropout, and dense layers.

6. **Results**

   - After training the model, check the summary for the total and trainable parameters.
   - Evaluate the model on the test set and visualize the results.

7. **Notes**

   - Modify the code and parameters based on specific requirements.
   - Adjust hyperparameters, such as learning rate, batch size, and model architecture, for optimal performance.

8. **Contributing**

   - Contributions are welcome. Fork the repository, make changes, and submit a pull request.

9. **License**

   - This project is licensed under the [MIT License](LICENSE).

Feel free to update this README file with additional information specific to your project and organization.

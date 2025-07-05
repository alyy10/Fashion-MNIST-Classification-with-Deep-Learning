# Fashion MNIST Classification with Deep Learning

## Project Theme
This project focuses on building and evaluating various deep learning models for image classification using the **Fashion MNIST** dataset. The goal is to classify 28x28 grayscale images into 10 categories of fashion items such as T-shirts, trousers, dresses, and more. The project explores the performance of different neural network architectures, from a shallow fully connected network to simple and deep convolutional neural networks (CNNs), to predict clothing types with high accuracy.

## End-to-End Approach

### 1. Data Loading and Preprocessing:
- **Dataset**: The **Fashion MNIST** dataset is loaded using `tensorflow.keras.datasets.fashion_mnist`, which provides 60,000 training and 10,000 test images of 10 clothing categories.
- **Preprocessing**:
  - **Reshaping**: The images are reshaped into 28x28 arrays if they are not in the correct format.
  - **Normalization**: The pixel values are normalized to a range of [0, 1] by dividing by 255.0 to improve convergence in neural networks.
  
### 2. Model Development:
- **Shallow Neural Network**: A baseline model with a fully connected (dense) network consisting of an input layer, one hidden layer, and an output layer.
- **Simple CNN**: A convolutional neural network to capture spatial patterns in images. This model uses convolutional layers followed by dense layers for classification.
- **Deep CNN**: A deeper version of the CNN, incorporating more layers to learn complex patterns and features from the images.
- **Model with Additional Layers**: Adding more complexity to the deep CNN to observe if additional layers improve performance.

### 3. Model Training:
- The models are trained for 10 epochs, with a batch size of 32, using the **Adam optimizer** and **sparse categorical cross-entropy** loss function for multi-class classification.
- The models are trained on the training data and evaluated on the test set. Training and validation accuracy are tracked to assess performance.

### 4. Evaluation:
- **Test Accuracy**: The accuracy on the unseen test data is reported after training.
- **Confusion Matrix**: A confusion matrix is generated to visualize the model's performance in classifying each category.
- **Classification Report**: Precision, recall, and F1-scores are calculated for each class, showing how well the model performs for each category.

### 5. Visualizations:
- **Training and Validation Curves**: Accuracy and loss curves are plotted for both training and validation sets to visualize the model’s performance across epochs.
- **Sample Predictions**: A set of 10 test images is displayed with predicted and actual labels to visually inspect model performance.

## Tech Stack Used
- **TensorFlow / Keras**: Deep learning framework used for building and training the models.
- **NumPy**: For handling numerical operations and data preprocessing.
- **Matplotlib / Seaborn**: For plotting training histories, confusion matrices, and displaying sample predictions.
- **Scikit-learn**: For generating the classification report and confusion matrix.
- **Fashion MNIST Dataset**: A dataset of 60,000 training and 10,000 test images for fashion item classification.

## How We Achieved the Solution
1. **Data Loading & Preprocessing**:
   - The **Fashion MNIST dataset** was loaded using TensorFlow's `fashion_mnist` function. The data was preprocessed by reshaping and normalizing the images to make them suitable for training.
   
2. **Model Creation**:
   - **Shallow Neural Network**: We first created a simple dense network to set a baseline for performance.
   - **Simple CNN**: Convolutional layers were added to the network to allow it to capture spatial features from the images.
   - **Deep CNN**: Further layers were added to the CNN, making it deeper to learn more complex patterns.
   - **Model with Additional Layers**: We experimented by adding extra layers to the deep CNN to see if performance could be improved.
   
3. **Model Training**:
   - The models were trained using **10 epochs** and evaluated using the test set.
   - During training, we monitored the training and validation accuracy to avoid overfitting and ensure the models were generalizing well to unseen data.

4. **Evaluation and Visualization**:
   - We evaluated the models using **classification reports** (precision, recall, F1-score), **confusion matrices**, and **sample predictions** to assess performance and identify potential improvements.
   - The **training and validation curves** were plotted to check for signs of overfitting or underfitting during the training process.

## Results and Findings

### 1. Shallow Neural Network (Baseline Model)
   - **Test Accuracy**: 88.66%
   - **Training Accuracy**: 91.03%
   - **Validation Accuracy**: 88.66%
   - **Test Loss**: 0.3321
   
   **Analysis**:
   - The shallow neural network performs reasonably well with an accuracy of around 88.66% but leaves room for improvement.
   - The training accuracy is slightly higher than the test accuracy, suggesting some overfitting, but the gap is small.

### 2. Simple CNN
   - **Test Accuracy**: 91.05%
   - **Training Accuracy**: 91.97%
   - **Validation Accuracy**: 91.05%
   - **Test Loss**: 0.2508
   
   **Analysis**:
   - The simple CNN performs significantly better than the shallow network, with an accuracy of 91.05% on the test set.
   - The model's ability to generalize is shown by the similar training and validation accuracies.
   - The test loss is lower compared to the shallow network’s, indicating a better fit for the data.

### 3. Deep CNN
   - **Test Accuracy**: 91.05%
   - **Training Accuracy**: 91.97%
   - **Validation Accuracy**: 91.05%
   - **Test Loss**: 0.2508
   
   **Analysis**:
   - The performance of the deep CNN is identical to the simple CNN, with no improvement observed in terms of accuracy or loss.
   - This suggests that the increased depth did not provide additional benefits in this specific case, possibly due to insufficient data or overcomplication of the model.

### 4. Shallow Network with Incorrect Results
   - **Test Accuracy**: 10%
   - **Classification Report**: Precision, recall, and F1-scores are extremely low across all classes.
   
   **Analysis**:
   - The shallow network has failed dramatically, with only 10% accuracy.
   - The model is not learning effectively, as evidenced by the undefined metrics and very low precision and recall for all classes.

### 5. Simple CNN with Proper Results
   - **Test Accuracy**: 91%
   - **Classification Report**: Precision, recall, and F1-scores are good, with values around 0.85-0.98 for most classes.
   
   **Analysis**:
   - The simple CNN achieves an accuracy of 91% and performs well across various metrics.
   - Precision and recall are strong for most classes, showing that the model is good at identifying the correct categories of fashion items, especially "Trouser," "Sandal," and "Sneaker."

---

## Conclusion
- **Best Performing Model**: The **Simple CNN** model achieved the best results, with 91% accuracy, good precision, recall, and F1-scores across all classes.
- **Shallow Network**: The shallow network performed decently but was outperformed by the CNN models.
- **Deep CNN**: Despite its increased complexity, the deep CNN did not improve upon the simple CNN, indicating that deeper models may not always provide significant gains.
- **Problematic Shallow Network**: The shallow network with incorrect results highlights the importance of correct data preprocessing and model configuration.

The project demonstrates how CNNs can significantly improve classification accuracy in image-based tasks compared to shallow neural networks. The best results were achieved with the **Simple CNN**, making it the most suitable choice for this fashion item classification task.


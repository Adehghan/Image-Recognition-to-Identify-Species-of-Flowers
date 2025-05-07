
# Flower Species Image Classification with CNN

This deep learning project uses a Convolutional Neural Network (CNN) to classify images of flowers into five categories: daisy, dandelion, rose, sunflower, and tulip. The model is built using TensorFlow and Keras and explores data augmentation, dropout regularization, and hyperparameter tuning to enhance accuracy.

---

## Dataset

- Source: [TensorFlow Flower Photos Dataset](http://download.tensorflow.org/example_images/flower_photos.tgz)
- Classes: Daisy, Dandelion, Rose, Sunflower, Tulip
- Total Images: ~3,670

---

## Model Architecture

- 3 Convolutional Layers with ReLU Activation
- MaxPooling after each convolution
- Flattening and Dense Layers
- Dropout Regularization
- Output Layer with Softmax Activation for 5-class classification

---

## Techniques Used

- Convolutional Neural Networks (CNN)
- Data Augmentation with ImageDataGenerator
- Dropout and Regularization
- Hyperparameter Tuning (learning rate, kernel size, layers)
- Evaluation with Confusion Matrix and Accuracy Metrics

---

## Technologies

- Python
- TensorFlow / Keras
- NumPy, Pandas
- Matplotlib

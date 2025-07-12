# MNIST Digit Recognition Project

A machine learning project that implements a neural network to recognize handwritten digits using the MNIST dataset. This project demonstrates the complete pipeline from data preprocessing to model training and prediction on custom images.

## 🎯 Project Overview

This project uses TensorFlow/Keras to build a neural network that can recognize handwritten digits (0-9) from the MNIST dataset. The model achieves high accuracy and can be used to predict digits from custom images.

## ✨ Features

- **Neural Network Architecture**: Multi-layer perceptron with ReLU activation
- **Data Preprocessing**: Automatic normalization of input data
- **Model Training**: Configurable training epochs with accuracy monitoring
- **Custom Image Prediction**: Ability to predict digits from custom PNG images
- **Model Persistence**: Save and load trained models
- **Visualization**: Display prediction results with matplotlib

## 🏗️ Architecture

The neural network consists of:
- **Input Layer**: Flattened 28x28 pixel images
- **Hidden Layer 1**: 128 neurons with ReLU activation
- **Hidden Layer 2**: 128 neurons with ReLU activation  
- **Output Layer**: 10 neurons with softmax activation (one for each digit 0-9)

## 📋 Requirements

- Python 3.7+
- TensorFlow
- OpenCV (cv2)
- NumPy
- Matplotlib

## 🚀 Installation

1. Clone this repository:
```bash
git clone <your-repository-url>
cd ML
```

2. Install the required dependencies:
```bash
pip install tensorflow opencv-python numpy matplotlib
```

## 📖 Usage

### Training the Model

Run the main script to train the model:

```bash
python main.py
```

This will:
1. Load and preprocess the MNIST dataset
2. Train the neural network for 3 epochs
3. Evaluate the model on test data
4. Save the trained model as `digits.keras`
5. Make predictions on custom images (1.png through 5.png)

### Custom Predictions

To use your own images:
1. Place PNG images in the project directory
2. Update the file names in the prediction loop in `main.py`
3. Run the script to get predictions

## 📊 Model Performance

The model typically achieves:
- **Accuracy**: ~95-98% on the MNIST test set
- **Loss**: Low categorical crossentropy loss
- **Training Time**: ~30-60 seconds on CPU

## 📁 Project Structure

```
ML/
├── main.py              # Main training and prediction script
├── digits.keras         # Trained model file
├── 1.png               # Custom test image
├── 2.png               # Custom test image
├── 3.png               # Custom test image
├── 4.png               # Custom test image
├── 5.png               # Custom test image
└── README.md           # This file
```

## 🔧 Customization

### Adjusting Model Parameters

You can modify the following parameters in `main.py`:

- **Training epochs**: Change the `epochs` parameter in `model.fit()`
- **Hidden layer size**: Modify the number of neurons in Dense layers
- **Learning rate**: Change the optimizer parameters
- **Image preprocessing**: Adjust normalization and resizing parameters

### Adding More Layers

To add more hidden layers:
```python
model.add(tf.keras.layers.Dense(64, activation="relu"))
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- MNIST dataset creators
- TensorFlow/Keras development team
- OpenCV community

## 📞 Contact

For questions or suggestions, please open an issue on GitHub.

---

**Note**: This project is designed for educational purposes and demonstrates fundamental machine learning concepts using the MNIST dataset. 
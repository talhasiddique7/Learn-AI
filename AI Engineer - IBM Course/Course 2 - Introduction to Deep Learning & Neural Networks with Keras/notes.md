## **Module 1 - Introduction to Deep Learning**

### **1.1 Introduction to Deep Learning**
Deep learning is a subset of machine learning where algorithms are inspired by the structure and function of the human brain (neural networks). It allows machines to automatically learn hierarchical representations of data for tasks like classification, regression, and more.

#### **Key Characteristics:**
- Handles unstructured data (e.g., images, audio, text).
- Requires large datasets for training.
- Uses deep neural networks with multiple layers.

**Example Applications:**
- Image recognition (e.g., identifying cats vs. dogs).
- Speech-to-text systems.
- Language translation (e.g., Google Translate).

---

### **1.2 Biological Neural Networks**
Biological neural networks are the natural neurons in the human brain. Each neuron receives input signals, processes them, and transmits output signals.

#### **Key Features:**
- **Dendrites**: Receive signals from other neurons.
- **Cell Body**: Processes the input signals.
- **Axon**: Sends signals to other neurons.

#### **Relation to Artificial Neural Networks:**
- Biological neurons inspired the creation of artificial neurons.
- Signals in biological neurons are analogous to inputs and outputs in artificial neurons.

---

### **1.3 Artificial Neural Networks - Forward Propagation**
Artificial Neural Networks (ANNs) are computational models that mimic the way neurons work. **Forward propagation** is the process of passing input data through the network to compute predictions.

#### **Steps:**
1. **Input Layer**: Receives input features (\(x_1, x_2, \ldots, x_n\)).
2. **Hidden Layers**: Perform weighted sums and apply activation functions.
3. **Output Layer**: Produces the final prediction.

#### **Mathematical Representation**:
For a single neuron:
\[
z = W \cdot X + b
\]
\[
y = f(z)
\]
Where:
- \(W\): Weights
- \(X\): Inputs
- \(b\): Bias
- \(f\): Activation function

**Example**:
- Input: \(x_1 = 0.5\), \(x_2 = 0.6\)
- Weights: \(w_1 = 0.8\), \(w_2 = 0.4\)
- Bias: \(b = 0.2\)
\[
z = (0.5 \cdot 0.8) + (0.6 \cdot 0.4) + 0.2 = 0.82
\]

---

## **Module 2 - Artificial Neural Networks**

### **2.1 Gradient Descent**
Gradient descent is an optimization algorithm used to minimize the cost function by updating weights iteratively.

#### **Steps**:
1. Initialize weights randomly.
2. Compute the gradient of the cost function.
3. Update weights using:
\[
W = W - \eta \cdot \frac{\partial L}{\partial W}
\]
Where:
- \(W\): Weight
- \(\eta\): Learning rate
- \(\frac{\partial L}{\partial W}\): Gradient of loss with respect to weight.

---

### **2.2 Backpropagation**
Backpropagation is the process of propagating the error backward to update weights. It uses the chain rule to compute gradients for each layer.

#### **Steps**:
1. Perform forward propagation to compute predictions.
2. Compute the error (loss).
3. Propagate the error backward to calculate gradients.
4. Update weights using gradient descent.

---

### **2.3 Vanishing Gradient**
The vanishing gradient problem occurs when gradients become very small, making it difficult for the network to update weights in earlier layers. It often happens with deep networks using activation functions like Sigmoid or Tanh.

#### **Solution**:
- Use activation functions like ReLU.
- Use architectures like LSTM or GRU for recurrent networks.

---

### **2.4 Activation Functions**
Activation functions introduce non-linearity into the network, enabling it to learn complex patterns.

#### **Common Functions**:
1. **ReLU**: \(\max(0, z)\)
2. **Sigmoid**: \(\frac{1}{1 + e^{-z}}\)
3. **Tanh**: \(\frac{e^z - e^{-z}}{e^z + e^{-z}}\)
4. **Softmax**: Outputs probabilities for multi-class classification.

---

## **Module 3 - Deep Learning Libraries**

### **3.1 Introduction to Deep Learning Libraries**
Popular libraries include:
- **TensorFlow**: Open-source framework for building machine learning models.
- **Keras**: High-level API for TensorFlow.
- **PyTorch**: Dynamic computation graph-based framework.

---

### **3.2 Regression Models with Keras**
**Steps**:
1. Define the model:
```python
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(1, input_dim=1))
```
2. Compile:
```python
model.compile(optimizer='adam', loss='mse')
```
3. Train:
```python
model.fit(x_train, y_train, epochs=100)
```

---

### **3.3 Classification Models with Keras**
For binary classification:
1. Define the model:
```python
model.add(Dense(1, activation='sigmoid'))
```
2. Compile:
```python
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```
3. Train:
```python
model.fit(x_train, y_train, epochs=100, batch_size=10)
```

---

## **Module 4 - Deep Learning Models**

### **4.1 Shallow and Deep Neural Networks**
- **Shallow Networks**: Few layers, simple patterns.
- **Deep Networks**: Many layers, capable of learning hierarchical features.

---

### **4.2 Convolutional Neural Networks (CNNs)**
Used for image-related tasks.

#### **Key Components**:
- **Convolutional Layer**: Extracts features using filters.
- **Pooling Layer**: Reduces spatial dimensions.
- **Fully Connected Layer**: Performs classification.

**Example Applications**: Image recognition, object detection.

---

### **4.3 Recurrent Neural Networks (RNNs)**
Used for sequential data (e.g., time-series, text).

#### **Key Components**:
- Maintains a hidden state to capture sequence dependencies.
- Suffers from vanishing gradients; solved using LSTM or GRU.

---

### **4.4 Autoencoders**
Unsupervised learning models used for:
- Dimensionality reduction.
- Anomaly detection.

#### **Structure**:
- **Encoder**: Compresses input into a latent space.
- **Decoder**: Reconstructs the input from the latent space.

---

## **Module 5 - Course Assignment**
The course assignment involves:
1. Building and training models (e.g., Regression, Classification, CNNs).
2. Evaluating metrics like MAE, MSE, R², Accuracy, etc.
3. Implementing practical examples using TensorFlow/Keras.

---

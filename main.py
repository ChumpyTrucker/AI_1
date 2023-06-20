# Import numpy for numerical computations
import numpy as np

# Define the number of inputs, hidden units, and outputs
n_inputs = 2
n_hidden = 3
n_outputs = 1


# Define the activation function for the neurons
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Define the derivative of the activation function
def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))


# Define the loss function (mean squared error)
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


# Define the learning rate
alpha = 0.1

# Initialize the weights randomly
W1 = np.random.randn(n_inputs, n_hidden)
W2 = np.random.randn(n_hidden, n_outputs)

# Define some training data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Inputs
Y = np.array([[0], [1], [1], [0]])  # Outputs

# Train the network for 1000 epochs
for epoch in range(1000):

    # Forward pass: compute the outputs and the loss
    Z1 = X.dot(W1)  # Linear combination of inputs and weights
    A1 = sigmoid(Z1)  # Activation of hidden layer
    Z2 = A1.dot(W2)  # Linear combination of hidden layer and weights
    A2 = sigmoid(Z2)  # Activation of output layer
    L = mse(Y, A2)  # Loss

    # Backward pass: compute the gradients and update the weights
    dL_dA2 = -2 * (Y - A2)  # Derivative of loss with respect to output activation
    dA2_dZ2 = sigmoid_prime(Z2)  # Derivative of output activation with respect to output linear combination
    dZ2_dW2 = A1.T  # Derivative of output linear combination with respect to output weights
    dL_dW2 = dZ2_dW2.dot(dL_dA2 * dA2_dZ2)  # Derivative of loss with respect to output weights

    dL_dA1 = (dL_dA2 * dA2_dZ2).dot(W2.T)  # Derivative of loss with respect to hidden activation
    dA1_dZ1 = sigmoid_prime(Z1)  # Derivative of hidden activation with respect to hidden linear combination
    dZ1_dW1 = X.T  # Derivative of hidden linear combination with respect to hidden weights
    dL_dW1 = dZ1_dW1.dot(dL_dA1 * dA1_dZ1)  # Derivative of loss with respect to hidden weights

    W2 -= alpha * dL_dW2  # Update output weights
    W1 -= alpha * dL_dW1  # Update hidden weights

    # Print the loss every 100 epochs
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {L}")

# Test the network on the training data
for x, y in zip(X, Y):
    z1 = x.dot(W1)
    a1 = sigmoid(z1)
    z2 = a1.dot(W2)
    a2 = sigmoid(z2)
    print(f"Input: {x}, Output: {a2}, True: {y}")

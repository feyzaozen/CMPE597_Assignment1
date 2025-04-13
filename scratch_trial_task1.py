import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, average_precision_score

#Data loader. This part uses the os module to streamline the data loading process. This way, the data loader can work in any machine.
base_dir = os.path.dirname(os.path.dirname(__file__))
dataset_path = os.path.join(base_dir, 'dataset')

train_images = np.load(os.path.join(dataset_path, 'train_images.npy'))
train_labels = np.load(os.path.join(dataset_path, 'train_labels.npy'))
test_images = np.load(os.path.join(dataset_path, 'test_images.npy'))
test_labels = np.load(os.path.join(dataset_path, 'test_labels.npy'))

#Here we flatten and normalize the test and train images and apply one-hot encoding conversion to the categories.
#Numpy's flatten() and eye() functions are used to achieve it.
#To normalize the images, we simply divide them by 255 because each grayscale pixel's value ranges from 0 to 255.
X_train = train_images.reshape(len(train_images), -1) / 255.0
y_train = np.eye(5)[train_labels]

X_test  = test_images.reshape(len(test_images), -1)  / 255.0
y_test = np.eye(5)[test_labels] 

#Here, hyper-parameters are set.
input_size = 784 
hidden_sizes = [128, 64] #This variable represents the layers and number of neurons in each layer. If the array has one value, then
#there is only one hidden layer. If it has two, the network has two hidden layers etc.
output_size = 5
lr = 0.01
momentum = 0.9
epochs = 30
batch_size = 32

#We wanted to try different architectures with different amounts of layers. These two variables ensure that the code handles the
#calculations no matter how many layers the user enters.
layer_sizes = [input_size] + hidden_sizes + [output_size]
num_layers = len(layer_sizes) - 1

#We initialize the weights and biases here. Biases are initialized to zero, weights are initialized randomly following standard normal
#distribution.
weights = [np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2 / (layer_sizes[i] + layer_sizes[i + 1]))
           for i in range(num_layers)]
biases = [np.zeros((1, layer_sizes[i + 1])) for i in range(num_layers)]

#Velocities for weights and biases that will be used to implement Nesterov momentum are initialized here as zeros.
velocities_W = [np.zeros_like(W) for W in weights]
velocities_b = [np.zeros_like(b) for b in biases]

#Activation functions are defined here. Softmax is a necessity for the final layer, for the hidden layers we tried
#relu, sigmoid and tanh.
def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def tanh(Z):
    return np.tanh(Z)

#This function defines the forward propagation process.
def forward(X):
    activations = [X]
    for i in range(num_layers - 1):
        Z = activations[-1] @ weights[i] + biases[i]
        A = relu(Z)
        activations.append(A)
    

    Z_out = activations[-1] @ weights[-1] + biases[-1]
    A_out = softmax(Z_out)
    activations.append(A_out)
    
    return activations

#This function defines the backward propagation process. It also generates "scratch_first_layer_grad", which will be used to compare
#the gradients of our implementations from scratch and our implementation using PyTorch.
def backward(X, Y, activations, epoch=None, batch_index=None):
    global weights, biases, velocities_W, velocities_b
    global scratch_first_layer_grad

    batch_size = X.shape[0]
    dZ = activations[-1] - Y

    for i in range(num_layers - 1, -1, -1):
        dW = (activations[i].T @ dZ) / batch_size
        db = np.sum(dZ, axis=0, keepdims=True) / batch_size
        #We print the first layer gradients to compare them with PyTorch.
        if epoch == 0 and batch_index == 0 and i == 0:
            scratch_first_layer_grad = dW.copy()

        velocities_W[i] = momentum * velocities_W[i] - lr * dW
        velocities_b[i] = momentum * velocities_b[i] - lr * db

        weights[i] += velocities_W[i]
        biases[i] += velocities_b[i]

        if i > 0:
            dZ = (dZ @ weights[i].T) * (activations[i] > 0)


#This function computes the cross entropy loss.
def compute_loss(Y_true, Y_pred):
    return -np.mean(np.sum(Y_true * np.log(Y_pred + 1e-8), axis=1))

#These variables hold the values loss and accuracy values that will be plotted.
train_losses_scratch, train_accuracies_scratch, val_accuracies_scratch, test_accuracies_scratch = [], [], [], []

#This section seperates 10% of the training set for validation.
val_size = int(0.1 * len(X_train))
X_val, y_val = X_train[:val_size], y_train[:val_size]
X_train, y_train = X_train[val_size:], y_train[val_size:]

#This section includes the training loop. We fisrt randomly shuffle the data to prevent overfitting to a specific batch size, then
#initiate the variables. Then, we batch the data, define forward and backward propagations and loss. Lastly, we compute the loss
#and accuracy values for each epoch.
for epoch in range(epochs):
    indices = np.random.permutation(len(X_train))
    X_train, y_train = X_train[indices], y_train[indices]

    epoch_loss = 0
    correct_predictions = 0

    for i in range(0, len(X_train), batch_size):
        X_batch = X_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]

        activations = forward(X_batch)
        backward(X_batch, y_batch, activations, epoch=epoch, batch_index=i)

        loss = compute_loss(y_batch, activations[-1])

        epoch_loss += loss
        correct_predictions += np.sum(np.argmax(activations[-1], axis=1) == np.argmax(y_batch, axis=1))

    train_losses_scratch.append(epoch_loss / len(X_train))
    train_accuracy = correct_predictions / len(X_train)
    train_accuracies_scratch.append(train_accuracy)

    val_activations = forward(X_val)
    val_accuracy = np.mean(np.argmax(val_activations[-1], axis=1) == np.argmax(y_val, axis=1))
    val_accuracies_scratch.append(val_accuracy)

    test_activations = forward(X_test)
    test_accuracy = np.mean(np.argmax(test_activations[-1], axis=1) == np.argmax(y_test, axis=1))
    test_accuracies_scratch.append(test_accuracy)


    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}: Loss={epoch_loss:.4f}, Train Acc={train_accuracy:.4f}, Val Acc={val_accuracy:.4f}, Test Acc={test_accuracy:.4f}")

#We evaluate the model performance using sci-kit learn's built-in functions and print them.
y_pred_probs = forward(X_test)[-1]
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

accuracy = accuracy_score(y_true, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
auc_roc = roc_auc_score(y_test, y_pred_probs, multi_class="ovr")
auc_pr = average_precision_score(y_test, y_pred_probs)

print(f"Test Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
print(f"AUC-ROC Score: {auc_roc:.4f}")
print(f"AUC-PR Score: {auc_pr:.4f}")

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses_scratch, label="Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies_scratch, label="Training Accuracy")
plt.plot(val_accuracies_scratch, label="Validation Accuracy")
plt.plot(test_accuracies_scratch, label="Test Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.show()

#We save the results to compare them in the second part of the question.
np.save("train_losses_scratch.npy", train_losses_scratch)
np.save("train_accuracies_scratch.npy", train_accuracies_scratch)
np.save("val_accuracies_scratch.npy", val_accuracies_scratch)
np.save("scratch_first_layer_grad.npy", scratch_first_layer_grad)
np.save("test_accuracies_scratch.npy", test_accuracies_scratch)


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# The seed value for random operations
SEED = 42
np.random.seed(SEED)

# In QuickDraw dataset, each image is 28x28 pixels size, and gray scale. 
# We first convert it to a 784 (28x28) dimensional vector and then divide by 255 to normalize the pixel values ​​to the range [0,1].
glove_file = "glove.6B.50d.txt"
train_img = np.load("dataset/train_images.npy").reshape(-1, 784) / 255.
train_lbl = np.load("dataset/train_labels.npy")
test_img  = np.load("dataset/test_images.npy").reshape(-1, 784) / 255.
test_lbl  = np.load("dataset/test_labels.npy")

# Category names in our subset 
cat_names = ["rabbit", "yoga", "hand", "snowman", "motorbike"]

# We get the 50 dimensional embeddings corresponding to the category names from the GloVe file
vecs = {}
with open(glove_file, encoding="utf8") as f:
    for line in f:
        word, *values = line.strip().split()
        if word in cat_names:
            vecs[word] = np.asarray(values, dtype=np.float32)
        if len(vecs) == len(cat_names):
            break

# Glove embedding matrix from vecs selected
glove_matrix = np.stack([vecs[w] for w in cat_names])  

#Split train set into train and validation sets
val_ratio = 0.1
perm = np.random.permutation(len(train_img))
val_size = int(len(train_img) * val_ratio)
val_idx, tr_idx = perm[:val_size], perm[val_size:]

x_tr, y_tr = train_img[tr_idx], train_lbl[tr_idx]
x_val, y_val = train_img[val_idx], train_lbl[val_idx]
x_te, y_te = test_img, test_lbl

# RELU actiivation function
def relu(x): 
    return np.maximum(0, x)

# Derivative of RELU activation function
def drelu(x): 
    return (x > 0).astype(x.dtype)

# Batch iteration function for mini batch training
def batch_iter(x, y, batch_size, shuffle=True):
    idx = np.arange(len(x))
    if shuffle:
        np.random.shuffle(idx)
    for i in range(0, len(x), batch_size):
        j = idx[i:i+batch_size]
        yield x[j], y[j]

# L2 normalization function
def l2norm(v, eps=1e-8):
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + eps)


class MLP:
    def __init__(self, sizes):
        # Weights (W), biases (b), weight momentum (vW) and bias momentum (vb)
        self.W, self.b, self.vW, self.vb = [], [], [], []
        for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
            std = np.sqrt(2.0 / fan_in)  # He initialization used here
            self.W.append(np.random.randn(fan_in, fan_out).astype(np.float32) * std)
            # Biases are initialized as zero
            self.b.append(np.zeros(fan_out, dtype=np.float32))
            self.vW.append(np.zeros((fan_in, fan_out), dtype=np.float32))
            self.vb.append(np.zeros(fan_out, dtype=np.float32))
        self.z, self.a, self.dW, self.db = [], [], [], []

    def forward(self, x, training=False):
        # Forward pass 
        self.a = [x]
        self.z = []
        for W, b in zip(self.W, self.b):
            # Multiply the weight matrix with the output of the previous layer and add bias -> z = a_prev @ W + b.
            z = self.a[-1] @ W + b
            self.z.append(z)
            # Apply RELU activation on z -> a = relu(z)
            self.a.append(relu(z))
            # L2 normaliziation
        return l2norm(self.a[-1])

    def backward(self, grad_output):
        # Backward pass
        self.dW = [np.zeros_like(w) for w in self.W]
        self.db = [np.zeros_like(b) for b in self.b]
        grad = grad_output
        # Loop layers from end to start
        for i in reversed(range(len(self.W))):
            # derivative of relu
            grad = grad * drelu(self.z[i])
            # Weight gradient
            self.dW[i] += self.a[i].T @ grad / len(self.a[0])
            # Bias gradient
            self.db[i] += grad.mean(0)
            grad = grad @ self.W[i].T

    def step(self, lr, mu):
        # Momentum update
        for i in range(len(self.W)):
            self.vW[i] = mu * self.vW[i] - lr * self.dW[i]
            self.vb[i] = mu * self.vb[i] - lr * self.db[i]
            # Update
            self.W[i] += self.vW[i]
            self.b[i] += self.vb[i]

    def zero_grad(self):
        self.dW = [np.zeros_like(w) for w in self.W]
        self.db = [np.zeros_like(b) for b in self.b]

# Triplet Loss

# Calculates the Euclidean distances between anchor, 
# positive and negative embeddings and produces a loss vector according to the triplet margin loss formula.
def triplet_forward(a, p, n, margin=0.5, eps=1e-8):
    dap = np.linalg.norm(a - p, axis=1)
    dan = np.linalg.norm(a - n, axis=1)
    loss_vec = np.maximum(0.0, dap - dan + margin)
    return loss_vec.mean(), dap, dan, (loss_vec > 0)[:, None]


# Calculates gradients for embeddings (derivatives of the differences used in the triplet loss calculation)
def triplet_backward(a, p, n, dap, dan, mask, eps=1e-8):
    ga = ((a - p)/(dap[:,None]+eps) - (a - n)/(dan[:,None]+eps))
    gp = -(a - p)/(dap[:,None]+eps)
    gn =  (a - n)/(dan[:,None]+eps)
    scale = 1.0 / len(a)
    return mask*ga*scale, mask*gp*scale, mask*gn*scale

# Prediction:
# This function calculates the similarities with dot product between 
# image embeddings and word embeddings and returns class of the highest score
def predict(img_embeddings, word_net):
    cat_embeddings = word_net.forward(glove_matrix, training=False)
    sims = img_embeddings @ cat_embeddings.T
    return sims.argmax(1)


# This function processes input data (x_data) and labels (y_data) in mini-batches
def run_split(x_data, y_data, training, img_net, word_net, bs, lr, mu):
    loss_sum, correct, tot = 0.0, 0, 0

    for xb, yb in batch_iter(x_data, y_data, bs, shuffle=training):
        # In training mode the gradients are reset before each mini-batch starts
        if training:
            img_net.zero_grad()
            word_net.zero_grad()

        anchor = img_net.forward(xb, training)
        # Random negative label generation
        neg_y = (yb + np.random.randint(1, 5, len(yb))) % 5 

        cat_idx = np.concatenate([yb, neg_y])
        positive_negative_embeddings = word_net.forward(glove_matrix[cat_idx], training)
        positive, negative = np.split(positive_negative_embeddings, 2)

        # Triplet loss calculation
        loss, dap, dan, mask = triplet_forward(anchor, positive, negative)
        loss_sum += loss * len(xb)


        if not training:
            # Guess the class 
            preds = predict(anchor, word_net)
            correct += (preds == yb).sum()
            tot += len(xb)

        # Calculate gradients and update weights
        if training:
            ga, gp, gn = triplet_backward(anchor, positive, negative, dap, dan, mask)
            img_net.backward(ga)
            word_net.backward(np.concatenate([gp, gn]))
            img_net.step(lr, mu)
            word_net.step(lr, mu)

    # Accuracy calculation
    acc = correct / tot if tot else None
    return loss_sum / len(x_data), acc

# Model initialization
img_net  = MLP([784, 256, 128])
word_net = MLP([50, 64, 128])


lr = 0.03
mu = 0.9
epochs = 30
bs = 256

hist = {"tr_loss": [], "val_loss": [], "tr_acc": [], "val_acc": []}

# Training loop
for ep in range(1, epochs + 1):
    tr_loss, _ = run_split(x_tr, y_tr, True, img_net, word_net, bs, lr, mu)
    _, tr_acc  = run_split(x_tr, y_tr, False, img_net, word_net, bs, lr, mu)
    vl_loss, vl_acc = run_split(x_val, y_val, False, img_net, word_net, bs, lr, mu)

    hist["tr_loss"].append(tr_loss)
    hist["val_loss"].append(vl_loss)
    hist["tr_acc"].append(tr_acc)
    hist["val_acc"].append(vl_acc)

    print(f"[{ep:02}] Train L {tr_loss:.4f} | Acc {tr_acc*100:5.1f}%   "
          f"Val L {vl_loss:.4f} | Acc {vl_acc*100:5.1f}%")

# Test
test_loss, test_acc = run_split(x_te, y_te, False, img_net, word_net, bs, lr, mu)
print(f"\nTest  Loss {test_loss:.4f} | Acc {test_acc*100:.1f}%")

preds = []
for xb, _ in batch_iter(x_te, y_te, 512, shuffle=False):
    xb_emb = img_net.forward(xb, False)
    preds.extend(predict(xb_emb, word_net))

# Classification report
print("\nClassification report (test):")
print(classification_report(y_te, preds, target_names=cat_names, digits=4))

# Charts
plt.figure(figsize=(11,4))

plt.subplot(1,2,1)
plt.plot(hist["tr_loss"], label="train")
plt.plot(hist["val_loss"], label="val")
plt.title("Triplet Loss")
plt.legend()

plt.subplot(1,2,2)
plt.plot(hist["tr_acc"], label="train")
plt.plot(hist["val_acc"], label="val")
plt.title("Accuracy")
plt.legend()

plt.tight_layout()
plt.show()

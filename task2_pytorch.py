import numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# The seed value for random operations
SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Category names in our subset 
cat_names = ["rabbit","yoga","hand","snowman","motorbike"]



vecs={}
with open("glove.6B.50d.txt",encoding="utf8") as f:
    for line in f:
        w,*vals = line.strip().split()
        if w in cat_names:
            vecs[w] = np.asarray(vals,dtype=np.float32)
        if len(vecs)==len(cat_names): break

# Combining the category embeddings, 
# and convert them into a single tensor and assign them to the device that the model will use.     
glove_mat = torch.tensor(np.stack([vecs[w] for w in cat_names]),
                         dtype=torch.float32,device=DEVICE)

# In QuickDraw dataset, each image is 28x28 pixels size, and gray scale. 
# We first convert it to a 784 (28x28) dimensional vector and then divide by 255 to normalize the pixel values ​​to the range [0,1].

tr_img = np.load("quickdraw_subset_np/train_images.npy").reshape(-1,784)/255.
tr_lbl = np.load("quickdraw_subset_np/train_labels.npy")
te_img = np.load("quickdraw_subset_np/test_images.npy").reshape(-1,784)/255.
te_lbl = np.load("quickdraw_subset_np/test_labels.npy")

perm = np.random.permutation(len(tr_img))
val_sz = int(0.1*len(tr_img))
val_idx, tr_idx = perm[:val_sz], perm[val_sz:]

X_tr,y_tr = tr_img[tr_idx],tr_lbl[tr_idx]
X_val,y_val = tr_img[val_idx],tr_lbl[val_idx]
X_te,y_te = te_img,te_lbl


# Load QuickDraw data with PyTorch DataLoader.
class QD(Dataset):
    def __init__(s,X,y): 
        s.x=torch.tensor(X,dtype=torch.float32); s.y=torch.tensor(y)
    def __len__(s): 
        return len(s.x)
    def __getitem__(s,i): 
        return s.x[i],s.y[i]
    
# Then create DataLoader objects for training, validation, and test sets.
dl_tr  = DataLoader(QD(X_tr,y_tr), 256, shuffle=True)
dl_val = DataLoader(QD(X_val,y_val),512)
dl_te  = DataLoader(QD(X_te,y_te), 512)

# Common embedding size for image and words
EMB = 128


# 784 Input → 256 → 128 → L2 normalize
class ImgNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, EMB)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return nn.functional.normalize(x, dim=1)

# 50 (GloVe Embedding Size) → 64 → 128 → L2 normalize
class WordNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(50, 64)
        self.fc2 = nn.Linear(64, EMB)
    def forward(self, w):
        w = torch.relu(self.fc1(w))
        w = self.fc2(w)
        return nn.functional.normalize(w, dim=1)
    
img_net  = ImgNet().to(DEVICE)
word_net = WordNet().to(DEVICE)


# Triplet loss function
criterion = nn.TripletMarginLoss(margin=0.5)
# Stochastic gradient descent optimizer with momentum 0.9
opt = torch.optim.SGD(list(img_net.parameters())+list(word_net.parameters()),
                      lr=0.03, momentum=0.9)


# Prediction:
# This function calculates the similarities with dot product between 
# image embeddings and word embeddings and returns class of the highest score
def predict(img_embedding):
    cat_embedding = word_net(glove_mat)
    return (img_embedding @ cat_embedding.T).argmax(1)

# This function processes input data (x_data) and labels (y_data) in mini-batches
def run(loader, train):
    img_net.train(train)
    word_net.train(train)
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
        anchor_embeddings = img_net(x_batch)
        # Glove embeddings of correct classes
        positive_embeddings = word_net(glove_mat[y_batch])
        # Random negative labels
        negative_indices = (y_batch + torch.randint(1, 5, (len(y_batch),), device=DEVICE)) % 5
        negative_embeddings = word_net(glove_mat[negative_indices])
        # Triplet loss calculation
        loss = criterion(anchor_embeddings, positive_embeddings, negative_embeddings)

        if train:
            opt.zero_grad()
            loss.backward()
            opt.step()

        total_loss += loss.item() * len(x_batch)

        # Prediction
        predictions = predict(anchor_embeddings)
        total_correct += (predictions == y_batch).sum().item()
        total_samples += len(x_batch)

    average_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return average_loss, accuracy



# Training loop
history = {"tr_loss": [], "val_loss": [], "tr_acc": [], "val_acc": []}
EPOCHS = 30       

for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = run(dl_tr,  True)
    val_loss, val_acc = run(dl_val, False)

    history["tr_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["tr_acc"].append(train_acc)
    history["val_acc"].append(val_acc)

    print(f"[{epoch:02}] Train L {train_loss:.4f} | Acc {train_acc*100:5.1f}%   "
          f"Val L {val_loss:.4f} | Acc {val_acc*100:5.1f}%")



# Test the model on the test set
test_loss, test_accuracy = run(dl_te,False)
print(f"\nTest Loss {test_loss:.4f} | Acc {test_accuracy*100:.1f}%")
# Classification report
all_p,all_t=[],[]
for x,y in dl_te:
    x=x.to(DEVICE)
    all_p+=predict(img_net(x)).cpu().tolist()
    all_t+=y.tolist()
print(classification_report(all_t,all_p,target_names=cat_names,digits=4))

# Charts
plt.figure(figsize=(10,4))
plt.subplot(1,2,1); plt.plot(history["tr_loss"],label="train"); plt.plot(history["val_loss"],label="val"); plt.title("Loss"); plt.legend()
plt.subplot(1,2,2); plt.plot(history["tr_acc"],label="train"); plt.plot(history["val_acc"],label="val"); plt.title("Acc"); plt.legend()
plt.tight_layout(); plt.show()

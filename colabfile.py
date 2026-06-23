import torch
from pennylane import numpy as np
import torch.nn as nn
import pennylane as qml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import os

class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim
        )
    def forward(self, x):
        x = self.embedding(x)
        return x

    
class BiGRULayer(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super() .__init__()
        self.bigru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        ) 
    def forward(self, x):
        output, hidden = self.bigru(x)
        return output
    
class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(
            hidden_dim * 2,
            1
        )

    def forward(self, x):
        scores = self.attention(x)
        weights = torch.softmax(
            scores,
            dim=1
        )
        context = torch.sum(
            weights * x,
            dim=1
        )
        return context, weights

## Quantum Lyaer
n_qubits = 4
dev = qml.device(
    "default.qubit",
    wires = n_qubits
)

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(
        inputs,
        wires=range(n_qubits)
    )
    qml.StronglyEntanglingLayers(
        weights,
        wires=range(n_qubits)
    )
    return[
        qml.expval(qml.PauliZ(i))
        for i in range(n_qubits)
    ]

# Creating Torch Layer
weight_shapes = {
    "weights": (2, n_qubits, 3)
}
qlayer = qml.qnn.TorchLayer(
    quantum_circuit,
    weight_shapes
)

class QuantumLayer(nn.Module):
    def __init__(self,
                 input_dim=256,
                 n_qubits=4):
        super().__init__()
        self.fc_reduce = nn.Linear(
            input_dim,
            n_qubits
        )
        weight_shapes = {
            "weights": (2, n_qubits, 3)
        }
        self.quantum = qml.qnn.TorchLayer(
            quantum_circuit,
            weight_shapes
        )

    def forward(self, x):
        x = self.fc_reduce(x)
        x = self.quantum(x)
        return x
    
class Classifier(nn.Module):
    def __init__(self,
                 input_dim,
                 num_classes):
        super().__init__()
        self.fc = nn.Linear(
            input_dim,
            num_classes
        )
    def forward(self, x):
        x = self.fc(x)
        return x
        
class QeMMA(nn.Module):
    def __init__(
            self,
            vocab_size,
            embedding_dim,
            hidden_dim,
            num_classes
    ):
        super().__init__()
        self.embedding = EmbeddingLayer(
            vocab_size,
            embedding_dim
        )
        self.bigru = BiGRULayer(
            embedding_dim,
            hidden_dim
        )
        self.attention = AttentionLayer(
            hidden_dim
        )
        self.quantum= QuantumLayer(
            input_dim=hidden_dim*2,
            n_qubits=4
        )
        self.classifier = Classifier(
            input_dim = 4,
            num_classes=num_classes
        )
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.bigru(x)
        context, weights = self.attention(x)
        x = self.quantum(context)
        x = self.classifier(x)
        return x


################################################################################################
############# Training File ####################################################################
################################################################################################



# Read CSV
df = pd.read_csv("Twitter_Data.csv")
print(df.head)

df = df.dropna()

label_map = {
    -1: 0,   # negative
     0: 1,   # neutral
     1: 2    # positive
}

df["category"] = df["category"].map(label_map)

texts = df["clean_text"].astype(str).tolist()
labels = df["category"].astype(int).tolist()

# Build Vocabulary
vocab = {
    "<PAD>":0,
    "<UNK>":1
}

for text in texts:
    for word in text.split():
        if word not in vocab:
            vocab[word] = len(vocab)

# Encoded Text
encoded_texts = []

for text in texts:
    encoded = [
        vocab.get(word,1)
        for word in text.split()

    ]

    encoded_texts.append(encoded)

# print(len(encoded_texts))
# print(len(labels))
print(len(texts))
print(len(encoded_texts))
print(len(labels))

# Padding
MAX_LEN = 50
padded_texts = []
for seq in encoded_texts:
    if len(seq) > MAX_LEN:
        seq = seq[:MAX_LEN]

    else:

        seq = seq + [0]*(MAX_LEN-len(seq))

    padded_texts.append(seq)

print(type(texts))
# print(type(labels))

# # print(len(texts))
# # print(len(labels))

# # print(type(padded_texts))
# # print(type(labels))

# # print(len(padded_texts))
# # print(len(labels))

# train Validation Split
X_train, X_val, y_train, y_val = train_test_split(
    padded_texts,
    labels,
    test_size=0.2,
    random_state=42
)

# TensorDataset
X_train = torch.tensor(X_train)
y_train = torch.tensor(y_train)

X_val = torch.tensor(X_val)
y_val = torch.tensor(y_val)

train_dataset = TensorDataset(
    X_train,
    y_train
)
val_dataset = TensorDataset(
    X_val,
    y_val
)

# DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)
val_loader = DataLoader(
    val_dataset,
    batch_size=32
)

# Model
model = QeMMA(
    vocab_size=len(vocab),
    embedding_dim=300,
    hidden_dim=128,
    num_classes=3
)

# Loss
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

print("Texts:", len(texts))
print("Encoded:", len(encoded_texts))
print("Labels:", len(labels))

# CheckPoints
start_epoch = 0
if os.path.exists("latest_checkpoint.pth"):
    checkpoint = torch.load(
        "latest_checkpoint.pth"
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )
    
    start_epoch = checkpoint["epoch"] + 1

    print(
        f"Resuming from Epoch {start_epoch}"
    )
    
# Training Loop
for epoch in range(start_epoch, 2):
    model.train()
    
    # Batch Loop
    for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
    # for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(
            outputs,
            y_batch
        )
        # print(
        #     f"Epoch {epoch+1}, Loss = {loss.item():.4f}"
        # )
        if batch_idx % 100 == 0:
            print(
                f"Epoch {epoch+1} | "
                f"Batch {batch_idx}/{len(train_loader)} | "
                f"Loss = {loss.item():.4f}"
            )

        loss.backward()
        optimizer.step()

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
        },
        "latest_checkpoint.pth"
    )
    print(
        f"Checkpoint saved for Epoch {epoch+1}"
    )
        

# Validation
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for X_batch, y_batch in val_loader:        
        outputs = model(X_batch)
        
        preds = torch.argmax(
            outputs,
            dim=1
        )

        all_preds.extend(
            preds.tolist()
        )

        all_labels.extend(
            y_batch.tolist()
        )

# F1 Score
f1 = f1_score(
    all_labels,
    all_preds,
    average="macro"
)
print(f"F1 Score: {f1}")

# torch.save(
#     model.state_dict(),
#     "qemma.pth"
# )

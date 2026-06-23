import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from model import QeMMA

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


# Training Loop
for epoch in range(2):
    model.train()
    
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(
            outputs,
            y_batch
        )
        print(
            f"Epoch {epoch+1}, Loss = {loss.item():.4f}"
        )

        loss.backward()
        optimizer.step()
        

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

torch.save(
    model.state_dict(),
    "qemma.pth"
)

import torch
from pennylane import numpy as np
import torch.nn as nn
import pennylane as qml

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
#     print("Starting")
#     def forward(self, x):
#         print("inside forward")
#         x = self.embedding(x)
#         print("embedding:", x.shape)

#         x = self.bigru(x)
#         print("bigru:", x.shape)

#         context, weights = self.attention(x)
#         print("context:", context.shape)

#         x = self.quantum(context)
#         print("quantum:", x.shape)

#         x = self.classifier(x)
#         print("classifier:", x.shape)

        # return x
    

# print("Before model")

# model = QeMMA(
#     vocab_size=1000,
#     embedding_dim=128,
#     hidden_dim=128,
#     num_classes=2
# )

# print("After model")

# x = torch.randint(0, 1000, (2, 10))

# print("Before forward")

# output = model(x)

# print("After forward")

# criterion = nn.CrossEntropyLoss()

# y = torch.tensor([0, 1])

# loss = criterion(output, y)

# print("Loss:", loss)

# loss.backward()

# print(
#     "Gradient:",
#     model.quantum.fc_reduce.weight.grad
# )
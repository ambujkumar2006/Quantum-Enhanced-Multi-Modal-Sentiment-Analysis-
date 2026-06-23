import torch
from model1 import EmbeddingLayer
from model1 import BiGRULayer
from model1 import AttentionLayer
from model1 import QuantumLayer
from model1 import Classifier
from model1 import QeMMA
import torch.nn as nn

# # # # # # model = EmbeddingLayer(vocab_size=100, embedding_dim=16)

# # # # # # x = torch.tensor([[1,2,5,6]])

# # # # # # output = model(x)

# # # # # # print(output.shape)

# # # # # # model = BiGRULayer(
# # # # # #     embedding_dim=300,
# # # # # #     hidden_dim=128
# # # # # # )

# # # # # # x = torch.randn(32, 100, 300)

# # # # # # output = model(x)
# # # # # # print(output.shape)
# # # # # # loss = output.mean()
# # # # # # loss.backward()

# # # # # # for name, param in model.named_parameters():
# # # # # #     print(
# # # # # #         name,
# # # # # #         param.grad is not None
# # # # # #     )

# # # # # # attention = AttentionLayer(
# # # # # #     hidden_dim=128
# # # # # # )
# # # # # # x = torch.randn(
# # # # # #     32,
# # # # # #     100,
# # # # # #     256
# # # # # # )
# # # # # # context, weights = attention(x)
# # # # # # print(context.shape)
# # # # # # print(weights.shape)

# # # # # embedding = EmbeddingLayer(
# # # # #     vocab_size=100,
# # # # #     embedding_dim=300
# # # # # )

# # # # # bigru = BiGRULayer(
# # # # #     embedding_dim=300,
# # # # #     hidden_dim=128
# # # # # )

# # # # # attention = AttentionLayer(
# # # # #     hidden_dim=128
# # # # # )
# # # # # x = torch.randint(
# # # # #     0,
# # # # #     100,
# # # # #     (32,100)
# # # # # )
# # # # # x=embedding(x)
# # # # # print(x.shape)
# # # # # x = bigru(x)
# # # # # print(x.shape)
# # # # # context, weights = attention(x)
# # # # # print(context.shape)
# # # # # print(weights.shape)

# # # # quantum = QuantumLayer(
# # # #     input_dim =256,
# # # #     n_qubits = 4
# # # # )

# # # # x = torch.randn(
# # # #     32,
# # # #     256
# # # # )

# # # # out = quantum(x)
# # # # print(out.shape)

# # # # loss = out.mean()
# # # # loss.backward()

# # # # for name, param in quantum.named_parameters():
# # # #     print(
# # # #         name,
# # # #         param.grad is not None
# # # #     )

# # # classifier = Classifier(
# # #     input_dim=4,
# # #     num_classes=3
# # # )
# # # x = torch.randn(
# # #     32,
# # #     4
# # # )
# # # output = classifier(x)
# # # print(output.shape)

# # # loss = output.mean()
# # # loss.backward()

# # # for name, param in classifier.named_parameters():
# # #     print(
# # #         name,
# # #         param.grad is not None
# # #     )

# # #################################
# # ## Testing The Entire Pipeline ##
# # #################################

# # embedding = EmbeddingLayer(
# #     vocab_size=10000,
# #     embedding_dim=300
# # )
# # bigru = BiGRULayer(
# #     embedding_dim=300,
# #     hidden_dim=128
# # )
# # attention = AttentionLayer(
# #     hidden_dim=128
# # )
# # quantum = QuantumLayer(
# #     input_dim=256,
# #     n_qubits=4
# # )
# # classifier = Classifier(
# #     input_dim=4,
# #     num_classes=3
# # )

# # x = torch.randint(
# #     0,
# #     10000,
# #     (32,100)
# # )
# # x = embedding(x)
# # print(x.shape
# #       )
# # x = bigru(x)
# # print(x.shape)

# # x, weights = attention(x)
# # print(x.shape)

# # x = quantum(x)
# # print(x.shape)

# # x = classifier(x)
# # print(x.shape)

# # model = QeMMA(
# #     vocab_size=10000,
# #     embedding_dim=300,
# #     hidden_dim=128,
# #     num_classes=3
# # )

# # # x = torch.randint(
# # #     low=0,
# # #     high=10000,
# # #     size=(4,50)
# # # )

# # # output = model(x)

# # # y = torch.tensor([5
# # #     0,
# # #     1,
# # #     2,
# # #     1
# # # ])

# # # criterion = nn.CrossEntropyLoss()

# # # loss = criterion(
# # #     output,
# # #     y
# # # )

# # print(model)
# # # print(x.shape)
# # # print(output)
# # # print(output.shape)
# # # print(y.shape)
# # # print(loss)
# # # loss.backward()
# # # print(
# # #     model.quantum.fc_reduce.weight.grad
# # # )

# def forward(self, x):
#     x = self.embedding(x)
#     print("embedding:", x.shape)

#     x = self.bigru(x)
#     print("bigru:", x.shape)

#     context, weights = self.attention(x)
#     print("context:", context.shape)

#     x = self.quantum(context)
#     print("quantum:", x.shape)

#     x = self.classifier(x)
#     print("classifier:", x.shape)

#     return x

model = QeMMA(
    vocab_size=1000,
    embedding_dim=128,
    hidden_dim=128,
    num_classes=2
)

print("After model")

x = torch.randint(0, 1000, (2, 10))

print("Before forward")

output = model(x)

print("After forward")

criterion = nn.CrossEntropyLoss()

y = torch.tensor([0, 1])

loss = criterion(output, y)

print("Loss:", loss)

loss.backward()

print(
    "Gradient:",
    model.quantum.fc_reduce.weight.grad
)

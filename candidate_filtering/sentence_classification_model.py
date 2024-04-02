import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer


class SentenceClassificationModel(nn.Module):
    def __init__(self, embedding_dim):
        super(SentenceClassificationModel, self).__init__()
        # Linear layer that takes the sentence embeddings and outputs a score for each sentence
        self.classifier = nn.Linear(embedding_dim, 1)

    def forward(self, x):
        x = x.cuda()
        # x is the input embeddings with shape (batch_size, embedding_dim)
        # Apply the linear layer
        # Return the logits
        return self.classifier(x)
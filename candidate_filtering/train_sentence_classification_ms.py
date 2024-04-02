from argparse import ArgumentParser

from utils import load_json
from sentence_dataset import SentenceDataset
from sentence_classification_model import SentenceClassificationModel

from torch.utils.data import random_split
import torch
from torch.utils.data import ConcatDataset
torch.manual_seed(42)
from torch import nn
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer


def parse_sentences(dataset):
    sentences = []
    error_indices = []
    for sample in dataset:
        if sample["has_error"]:
            sentences.append([" ".join(sent.split()[1: ]) for sent in sample["sentences"]])
            error_indices.append(int(sample["error_index"]))

    return sentences, error_indices


def validate(model, data_loader):
    model.eval()  # Set the model to evaluation mode
    correct_predictions = 0
    total_predictions = 0
    val_loss = 0
    
    with torch.no_grad():
        for embeddings, labels in data_loader:
            logits = model(embeddings)
            labels = labels[:, None].cuda()
            loss = criterion(logits, labels)
            val_loss += loss.item()
            _, predicted_indices = torch.max(logits, dim=1)
            correct_predictions += (predicted_indices == labels.cuda()).sum().item()
            total_predictions += labels.size(0)
    avg_val_loss = val_loss / total_predictions
    accuracy = correct_predictions / total_predictions
    return avg_val_loss, accuracy

def train(model, train_loader, val_loader, optimizer, criterion, epochs, checkpoint_path):
    best_val_accuracy = 0.0
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        total_acc = 0
        for embeddings, labels in train_loader:
            optimizer.zero_grad()
            logits = model(embeddings)
            labels = labels[:, None].cuda()
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_loss, val_accuracy = validate(model, val_loader)
        # Check if the current validation accuracy is the best
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            print(f'New best validation accuracy: {best_val_accuracy}, saving model...')
            torch.save(model.state_dict(), checkpoint_path)
        print(f'Epoch {epoch+1}, Loss: {total_loss/len(train_loader)}, Validation Loss: {val_loss} Validation Accuracy: {val_accuracy}')

def pad_collate(batch):
    # Find the longest list of embeddings in the batch
    max_len = max(len(item[0]) for item in batch)
    batch_embeddings = []
    batch_labels = []
    for embeddings, label in batch:
        # Calculate how much padding is needed
        padding_length = max_len - len(embeddings)
        # Pad the embeddings for sentences and add to the batch
        padded_embeddings = torch.cat([embeddings, torch.zeros(padding_length, embeddings.size(1))], dim=0)
        batch_embeddings.append(padded_embeddings)
        batch_labels.append(label)
    # Stack all the embeddings and labels to create a batch
    batch_embeddings = torch.stack(batch_embeddings)
    batch_labels = torch.tensor(batch_labels)
    return batch_embeddings, batch_labels

parser = ArgumentParser()
parser.add_argument("--train_dataset")
parser.add_argument("--val_dataset")
args = parser.parse_args()

train_data = load_json(args.train_dataset)
sentences, error_indices = parse_sentences(train_data)
embedding_model = SentenceTransformer('pritamdeka/S-PubMedBert-MS-MARCO').cuda()
train_dataset = SentenceDataset(embedding_model, sentences, error_indices, cache_file="candidate_filtering/cache/train_embeddings_cache.pkl")

test_data = load_json(args.val_dataset)
sentences, error_indices = parse_sentences(test_data)
eval_dataset = SentenceDataset(embedding_model, sentences, error_indices, cache_file="candidate_filtering/cache/test_embeddings_cache.pkl")

dataset = ConcatDataset([train_dataset, eval_dataset])

train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])


train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=pad_collate)
val_loader = DataLoader(val_dataset, batch_size=16, collate_fn=pad_collate)

embedding_dim = embedding_model[1].pooling_output_dimension
classifier = SentenceClassificationModel(embedding_dim).cuda()
optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

train(classifier, train_loader, val_loader, optimizer, criterion, epochs=20, checkpoint_path="candidate_filtering/checkpoint/sentence_classification.pt")

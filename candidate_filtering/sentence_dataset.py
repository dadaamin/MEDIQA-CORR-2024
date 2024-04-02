import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from pathlib import Path
import pickle
import os

class SentenceDataset(Dataset):
    def __init__(self, embedding_model, sentences_lists, correct_indices, cache_file='candidate_filtering/cache/embeddings_cache.pkl'):
        self.sentences_lists = sentences_lists
        self.correct_indices = correct_indices
        self.cache_file = cache_file
        self.embeddings = self.load_or_compute_embeddings(embedding_model)

    def load_or_compute_embeddings(self, embedding_model):
        # Check if cache file exists
        if os.path.exists(self.cache_file):
            print("Loading embeddings from cache.")
            with open(self.cache_file, 'rb') as cache:
                embeddings = pickle.load(cache)
        else:
            embeddings = [embedding_model.encode(sentences) for sentences in tqdm(self.sentences_lists, desc="Computing embeddings and caching")]
            with open(self.cache_file, 'wb') as cache:
                pickle.dump(embeddings, cache)
        return embeddings

    def __len__(self):
        return len(self.sentences_lists)

    def __getitem__(self, idx):
        # Fetch the embeddings and the correct index for the given list of sentences
        embeddings = self.embeddings[idx]
        label = self.correct_indices[idx]
        return torch.tensor(embeddings, dtype=torch.float32), label
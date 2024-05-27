"""
Loading in data and defining necessary datasets.
"""
import torch
from torch.utils.data import Dataset
import numpy as np
from util import flatten


class TrainDataset(Dataset):
    def __init__(self, queries, num_entities, num_relations, negative_sample_size, answer):
        # note that queries is a list of (query, query_structure) tuples
        super(TrainDataset, self).__init__()
        self.queries = queries  # list of tuples: [(), (), ...]
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.negative_sample_size = negative_sample_size
        self.answer = answer

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        query = self.queries[idx][0]
        query_structure = self.queries[idx][1]
        tail = np.random.choice(list(self.answer[query]))

        negative_sample = self.corrupt_sample(query)
        positive_sample = torch.LongTensor([tail])

        return positive_sample, negative_sample, flatten(query), query_structure

    @staticmethod
    def collate_fn(batch):
        positives, negatives, flattened_queries, query_structures = zip(*batch)

        batched_positives = torch.cat(positives)
        batched_negatives = torch.stack(negatives)
        # flattened_queries = torch.cat(flattened_queries)
        # query_structures = torch.cat(query_structures)

        return batched_positives, batched_negatives, flattened_queries, query_structures

    def corrupt_sample(self, query):
        samples = []
        size = 0

        while size < self.negative_sample_size:
            negative_sample = np.random.randint(self.num_entities, size=self.negative_sample_size * 2)
            mask = np.isin(negative_sample, self.answer[query], assume_unique=True, invert=True)
            negative_sample = negative_sample[mask]
            samples.append(negative_sample)
            size += negative_sample.size

        negative_sample = torch.from_numpy(np.concatenate(samples)[:self.negative_sample_size])
        return negative_sample


class TestDataset(Dataset):
    def __init__(self, queries, num_entities, num_relations):
        # note that queries is a list of (query, query_structure) tuples
        super(TestDataset, self).__init__()
        self.queries = queries
        self.num_entities = num_entities
        self.num_relations = num_relations

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        query = self.queries[idx][0]
        query_structure = self.queries[idx][1]
        negative_sample = torch.LongTensor(range(self.num_entities))

        return negative_sample, flatten(query), query, query_structure

    @staticmethod
    def collate_fn(batch):
        negatives, flattened_queries, queries, query_structure = zip(*batch)

        batched_negatives = torch.stack(negatives)

        return batched_negatives, flattened_queries, queries, query_structure

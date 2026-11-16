import math

import numpy as np
import torch
from torch.utils.data import Sampler

RARE_LABELS = ["Edema", "Fibrosis", "Pneumonia", "Hernia"]


class RarityAwareSampler(Sampler):
    def __init__(self, df, batch_size, rare_per_batch=2,
                 num_replicas=1, rank=0, seed=42):
        self.batch_size = batch_size
        self.rare_per_batch = rare_per_batch
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0

        is_rare = df[RARE_LABELS].to_numpy().any(axis=1)
        self.rare_idx = np.where(is_rare)[0]
        self.common_idx = np.where(~is_rare)[0]

        self.num_samples = math.ceil(len(df) / num_replicas)
        self.num_steps = math.ceil(self.num_samples / batch_size)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        rare = self.rare_idx[
            torch.randperm(len(self.rare_idx), generator=g).numpy()
        ]
        common = self.common_idx[
            torch.randperm(len(self.common_idx), generator=g).numpy()
        ]

        n_rare = self.num_steps * self.rare_per_batch
        n_common = self.num_steps * (self.batch_size - self.rare_per_batch)

        rare = np.resize(rare, n_rare * self.num_replicas)
        common = np.resize(common, n_common * self.num_replicas)

        rare = rare[self.rank::self.num_replicas]
        common = common[self.rank::self.num_replicas]

        indices = []
        for step in range(self.num_steps):
            r = rare[step * self.rare_per_batch:(step + 1) * self.rare_per_batch]
            c = common[
                step * (self.batch_size - self.rare_per_batch):
                (step + 1) * (self.batch_size - self.rare_per_batch)
            ]
            batch = np.concatenate([r, c])
            batch = batch[torch.randperm(len(batch), generator=g).numpy()]
            indices.extend(batch.tolist())

        return iter(indices[:self.num_samples])

    def __len__(self):
        return self.num_samples

import os
import pandas as pd
from torch.utils.data import Dataset

class cVAEDataset(Dataset):
    def __init__(self, x, cond, u, topic, article):
        self.x = x
        self.cond = cond
        self.u = u
        self.topic = topic
        self.article = article


    def __len__(self):
        return len(self.article)

    def __getitem__(self, idx):
        return self.x[idx],self.cond[idx],self.u[idx],self.topic[idx], self.article[idx]
import os
import pandas as pd
from utils import data_dir

class loader:
    def __init__(self, config, eval_dir, meta_path):
        self.config = config
        self.eval_dir = eval_dir
        self.meta_path = meta_path
        self.data_dir = data_dir
        self.meta_df = pd.read_csv(self.meta_path, sep='\t', index_col=0)

    def read_articles(self):
        return os.listdir(self.data_dir + self.eval_dir)

    def load_data(self):
        pass


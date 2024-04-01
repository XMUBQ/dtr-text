class skeleton:
    def __init__(self, config):
        pass

    def train(self,config, fold_data, print_table=True):
        raise NotImplementedError

    def update(self, model, config, data_iter):
        raise NotImplementedError
import torch
from torch.utils.data import TensorDataset, DataLoader
from collections import OrderedDict


class sk_plus:
    def __init__(self, outside_name=''):
        self.outside_name = outside_name

    def eval(self, data_loader, **kwargs):
        raise NotImplementedError

    def fit(self, x, y=None, **kwargs):
        raise NotImplementedError

    def predict(self, x, **kwargs):
        raise NotImplementedError

    def transform(self, x, **kwargs):
        raise NotImplementedError

    @staticmethod
    def numpy2tensor(x, y=None, **kwargs):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        x = x.float()

        if not isinstance(y, torch.Tensor) and y is not None:
            y = torch.tensor(y).long()

        tensor_para_dict = OrderedDict()
        for key in kwargs:
            if not isinstance(kwargs[key], torch.Tensor):
                kwargs[key]=torch.tensor(kwargs[key]).float()
            tensor_para_dict[key] = kwargs[key]

        dataset = TensorDataset(x, y, *tensor_para_dict.values()) if y is not None else TensorDataset(x, *tensor_para_dict.values())
        dataloader = DataLoader(dataset, batch_size=128, shuffle=False)
        return dataloader

    def fit_transform(self, x, y=None, **kwargs):
        #print(kwargs)
        self.fit(x, y, **kwargs)
        return self.transform(x, **kwargs)

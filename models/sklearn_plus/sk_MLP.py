import torch
from models.sklearn_plus.sk_plus import sk_plus
from models.backbone.MLP import MLP
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from utils import cuda_available, ckpt_dir, plot_dir, initialize_weights
import matplotlib.pyplot as plt


class sk_MLP(sk_plus):
    def transform(self, x, **kwargs):
        print("Not applicable here")
        raise NotImplementedError

    def __init__(self, input_size, output_size, hidden=128, num_of_layer=2, outside_name=''):
        super(sk_MLP, self).__init__(outside_name=outside_name)
        self.mlp = MLP(input_size, output_size, hidden, num_of_layer)
        self.mlp.apply(initialize_weights)
        if cuda_available:
            self.mlp = self.mlp.cuda()
        self.epoch, self.test_gap = 1000, 1
        self.optimizer = optim.Adam(params=self.mlp.parameters(), lr=1e-5, weight_decay=1e-2)
        self.loss_function = nn.CrossEntropyLoss()

    def eval(self, data_loader, **kwargs):
        self.mlp.eval()
        accumulate_y_pred = []
        for batch_X in data_loader:
            if cuda_available:
                batch_X[0] = batch_X[0].cuda()
            output = self.mlp(batch_X[0])
            _, y_pred = torch.max(output, 1)
            accumulate_y_pred.extend(y_pred.clone().detach().cpu().tolist())

        return accumulate_y_pred

    def fit(self, x, y=None, **kwargs):
        assert y is not None
        dataloader = self.numpy2tensor(x, y)
        self.mlp.train()
        best_f1 = 100000
        best_epoch = 0
        loss_list = []
        for i in tqdm(range(self.epoch)):
            batch_loss = 0
            for batch_X, batch_y in dataloader:
                if cuda_available:
                    batch_X = batch_X.cuda()
                    batch_y = batch_y.cuda()

                output = self.mlp(batch_X)
                loss = self.loss_function(output, batch_y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                batch_loss += loss.item()
                torch.cuda.empty_cache()
            loss_list.append(batch_loss / len(dataloader))
            if i % self.test_gap == 0:
                if batch_loss < best_f1:
                    best_f1 = batch_loss
                    best_epoch = i
                    torch.save(self.mlp.state_dict(), ckpt_dir + 'temp_MLP_epoch_' + str(i) + '.pt')
        plt.plot(loss_list)
        plt.savefig(plot_dir + 'MLP_loss_list.png')
        plt.close()
        self.mlp.load_state_dict(torch.load(ckpt_dir + 'temp_MLP_epoch_' + str(best_epoch) + '.pt'))

    def predict(self, x, **kwargs):
        dataloader = self.numpy2tensor(x)
        return torch.tensor(self.eval(dataloader))
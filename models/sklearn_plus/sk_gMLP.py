import torch
from models.sklearn_plus.sk_plus import sk_plus
from models.sklearn_plus.backbone.MLP import MLP
import torch.optim as optim
import torch.nn as nn
import torch.distributions as dist
from tqdm import tqdm
from utils import cuda_available, ckpt_dir, plot_dir, bert_dim
from func_help import initialize_weights
import matplotlib.pyplot as plt


class sk_gMLP(sk_plus):
    def transform(self, x, **kwargs):
        print("Not applicable here")
        raise NotImplementedError

    def __init__(self, input_size, output_size, hidden=128, num_of_layer=1, outside_name=''):
        super(sk_gMLP, self).__init__(outside_name=outside_name)
        self.MLP = MLP(input_size, output_size, hidden, num_of_layer)
        self.MLP.apply(initialize_weights)
        if cuda_available:
            self.MLP = self.MLP.cuda()
        self.epoch, self.test_gap = 250, 1
        self.optimizer = optim.Adam(params=self.MLP.parameters(), lr=5e-5, weight_decay=1e-2)
        self.loss_function = nn.MSELoss()

    @staticmethod
    def separate_ls_acs(batch_x, step):
        ls = batch_x[0][:, -bert_dim * step:]
        acs = batch_x[0][:, :-bert_dim * step]
        l_target = ls[:, -bert_dim:]
        cond_target = torch.cat([acs[:, :bert_dim * step], ls[:, :-bert_dim]], dim=1)  # x, a0, l0,
        return l_target, cond_target

    def get_sample_prob(self, batch_x, sigma=1):
        mean = self.MLP(batch_x)
        normal_dist = dist.Normal(mean, sigma)
        L1_samples = normal_dist.sample((1,))
        log_prob = normal_dist.log_prob(L1_samples)
        prob = torch.exp(log_prob)
        total_prob = prob.prod(2)
        return L1_samples, total_prob

    def eval(self, data_loader, **kwargs):
        step = kwargs.get('step', None)
        assert step is not None
        self.MLP.eval()
        accumulate_samples = []
        accumulate_prob = []
        for batch_X in data_loader:
            if cuda_available:
                batch_X[0] = batch_X[0].cuda()

            l_target, cond_target = self.separate_ls_acs(batch_X, step)
            # print("test",l_target.shape, a_target.shape)
            samples, prob = self.get_sample_prob(cond_target)
            accumulate_samples.append(samples.clone().detach().cpu())
            accumulate_prob.append(prob.clone().detach().cpu())

        return torch.cat(accumulate_samples, dim=1).squeeze(), torch.cat(accumulate_prob, dim=1).squeeze()

    def fit(self, x, y=None, **kwargs):
        dataloader = self.numpy2tensor(x)
        step = kwargs.get('step', None)
        assert step is not None
        self.MLP.train()
        best_f1 = 100000
        best_epoch = 0
        loss_list = []
        for i in tqdm(range(self.epoch)):
            batch_loss = 0
            for batch_X in dataloader:
                if cuda_available:
                    batch_X[0] = batch_X[0].cuda()

                l_target, cond_target = self.separate_ls_acs(batch_X, step)
                # print("train",l_target.shape, a_target.shape)
                output = self.MLP(cond_target)
                loss = self.loss_function(output, l_target)

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
                    torch.save(self.MLP.state_dict(),
                               ckpt_dir + self.outside_name + '_temp_gMLP_epoch_' + str(i) + '.pt')
        plt.plot(loss_list)
        plt.savefig(plot_dir + 'gMLP_loss_list.png')
        plt.close()
        self.MLP.load_state_dict(
            torch.load(ckpt_dir + self.outside_name + '_temp_gMLP_epoch_' + str(best_epoch) + '.pt'))

    def predict(self, x, **kwargs):
        step = kwargs.get('step', None)
        assert step is not None
        dataloader = self.numpy2tensor(x)
        return self.eval(dataloader, step=step)

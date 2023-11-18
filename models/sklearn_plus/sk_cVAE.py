from models.sklearn_plus.sk_vanilla_VAE import sk_vanilla_VAE
from models.sklearn_plus.backbone.cVAE import cVAE
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils import cuda_available, ckpt_dir, plot_dir
import matplotlib.pyplot as plt

class sk_cVAE(sk_vanilla_VAE):
    def __init__(self, n_components=50, outside_name=''):
        super(sk_cVAE, self).__init__(n_components=n_components, outside_name=outside_name)
        self.learning_rate = 1e-4
        self.epochs = 500
        self.criterion = nn.MSELoss()
        self.kld_weight = 0

    def eval(self, data_loader, **kwargs):
        self.vae.eval()
        all_decoded_x = []
        for batch_x, condition in data_loader:
            if cuda_available:
                batch_x = batch_x.cuda()
                condition = condition.cuda()
            output, _ = self.vae.encode(torch.cat((batch_x, condition), dim=1))
            all_decoded_x.append(output.clone().detach().cpu())
        return torch.cat(all_decoded_x, dim=0).numpy()

    def fit(self, x, y=None, **kwargs):
        assert y is None
        condition = kwargs.get('condition', None)
        assert condition is not None
        # fit into dataloader
        dataloader = self.numpy2tensor(x, **kwargs)

        # initialize model
        self.vae = cVAE(x.shape[1], condition.shape[1], self.n_components)
        if cuda_available:
            self.vae = self.vae.cuda()
        self.vae.train()
        optimizer = optim.Adam(self.vae.parameters(), lr=self.learning_rate)

        best_loss, best_epoch, loss_list, mse_loss_list, kl_loss_list = 100000, 0, [], [], []
        for epoch in tqdm(range(self.epochs)):
            batch_loss, mse_loss, kl_loss = 0, 0, 0
            for batch_x, condition in dataloader:
                if cuda_available:
                    batch_x = batch_x.cuda()
                    condition = condition.cuda()
                optimizer.zero_grad()
                encoded_mean, encoded_var, outputs = self.vae(batch_x, condition)
                # loss = self.criterion(outputs, batch_x[0])
                loss = self.vae_loss(outputs, batch_x, encoded_mean, encoded_var)
                loss['backward'].backward()
                optimizer.step()
                batch_loss += loss['backward'].item()
                mse_loss += loss['mse'].item()
                kl_loss += loss['kl'].item()
            loss_list.append(batch_loss / len(dataloader))
            mse_loss_list.append(mse_loss / len(dataloader))
            kl_loss_list.append(kl_loss / len(dataloader))
            if batch_loss < best_loss:
                best_epoch = epoch
                best_loss = batch_loss
                torch.save(self.vae.state_dict(),
                           ckpt_dir + self.outside_name + '_temp_cvae_epoch_' + str(epoch) + '.pt')

        plt.plot(loss_list)
        plt.savefig(plot_dir + 'cvae_loss_list.png')
        plt.close()

        plt.plot(mse_loss_list)
        plt.savefig(plot_dir + 'cvae_mse_loss_list.png')
        plt.close()

        plt.plot(kl_loss_list)
        plt.savefig(plot_dir + 'cvae_kl_loss_list.png')
        plt.close()

        print(mse_loss_list[-1], kl_loss_list[-1])

        self.vae.load_state_dict(
            torch.load(ckpt_dir + self.outside_name +'_temp_cvae_epoch_' + str(best_epoch) + '.pt'))

        return self

    def transform(self, x, **kwargs):
        if self.vae is None:
            print("Not fitted yet!")
            raise NotImplementedError
        condition = kwargs.get('condition', None)
        assert condition is not None

        dataloader = self.numpy2tensor(x, **kwargs)
        encoded_data = self.eval(dataloader)
        return encoded_data

    def predict(self, x, **kwargs):
        return self.transform(x)

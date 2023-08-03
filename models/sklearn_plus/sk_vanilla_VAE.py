from models.sklearn_plus.sk_plus import sk_plus
from models.sklearn_plus.backbone.VAE import VAE
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils import cuda_available, ckpt_dir, plot_dir
import matplotlib.pyplot as plt


class sk_vanilla_VAE(sk_plus):
    def __init__(self, n_components=50, outside_name=''):
        super(sk_vanilla_VAE, self).__init__(outside_name=outside_name)
        self.n_components = n_components
        self.vae = None
        self.learning_rate = 3e-3
        self.epochs = 500
        self.criterion = nn.MSELoss()
        self.kld_weight = 0

    def eval(self, data_loader, **kwargs):
        self.vae.eval()
        all_decoded_x = []
        for batch_x in data_loader:
            if cuda_available:
                batch_x[0] = batch_x[0].cuda()
            output = self.vae.encoder_mean(batch_x[0])
            all_decoded_x.append(output.clone().detach().cpu())
        return torch.cat(all_decoded_x, dim=0).numpy()

    def vae_loss(self, decoded, x, encoded_mean, encoded_var):
        mse_loss = self.criterion(decoded, x)
        kl_loss = -0.5 * torch.mean(torch.sum(1 + encoded_var - encoded_mean ** 2 - encoded_var.exp(), dim=1))
        return {'backward': mse_loss + self.kld_weight * kl_loss, 'mse': mse_loss, 'kl': kl_loss}

    def fit(self, x, y=None, **kwargs):
        assert y is None
        # fit into dataloader
        dataloader = self.numpy2tensor(x)

        # initialize model
        self.vae = VAE(x.shape[1], self.n_components)
        if cuda_available:
            self.vae = self.vae.cuda()
        self.vae.train()
        optimizer = optim.Adam(self.vae.parameters(), lr=self.learning_rate)

        best_loss, best_epoch, loss_list, mse_loss_list, kl_loss_list = 100000, 0, [], [], []
        for epoch in tqdm(range(self.epochs)):
            batch_loss, mse_loss, kl_loss = 0, 0, 0
            for batch_x in dataloader:
                if cuda_available:
                    batch_x[0] = batch_x[0].cuda()
                optimizer.zero_grad()
                encoded_mean, encoded_var, outputs = self.vae(batch_x[0])
                # loss = self.criterion(outputs, batch_x[0])
                loss = self.vae_loss(outputs, batch_x[0], encoded_mean, encoded_var)
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
                           ckpt_dir + self.outside_name + '_temp_vae_epoch_' + str(epoch) + '.pt')

        plt.plot(loss_list)
        plt.savefig(plot_dir + 'vae_loss_list.png')
        plt.close()

        plt.plot(mse_loss_list)
        plt.savefig(plot_dir + 'vae_mse_loss_list.png')
        plt.close()

        plt.plot(kl_loss_list)
        plt.savefig(plot_dir + 'vae_kl_loss_list.png')
        plt.close()

        print(mse_loss_list[-1], kl_loss_list[-1])

        self.vae.load_state_dict(
            torch.load(ckpt_dir + self.outside_name + '_temp_vae_epoch_' + str(best_epoch) + '.pt'))

        return self

    def transform(self, x, **kwargs):
        if self.vae is None:
            print("Not fitted yet!")
            raise NotImplementedError

        dataloader = self.numpy2tensor(x)
        encoded_data = self.eval(dataloader)
        return encoded_data

    def predict(self, x, **kwargs):
        return self.transform(x)

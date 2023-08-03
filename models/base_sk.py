from models.sk_model import sk_model
from joblib import dump
import torch
from utils import bert_dim


class base_sk(sk_model):
    def __init__(self, config):
        super(base_sk, self).__init__(config)

    def get_input_feat(self, data_iter, mode):
        acs = data_iter[mode]['acs']
        ls = data_iter[mode]['ls']
        acc_cond = data_iter[mode]['acc_cond']
        condition_dict = {'condition': torch.cat([acs[:, :-1, :], ls], dim=1).reshape(len(acs), -1)}

        if self.decompose_cond:
            if mode == 'train':
                condition_dict['condition'] = torch.tensor(self.pca.fit_transform(condition_dict['condition'].numpy()))
            else:
                condition_dict['condition'] = torch.tensor(self.pca.transform(condition_dict['condition'].numpy()))

        if self.decompose_a:
            a_target = acs[:, -1, :]
            if mode == 'train':
                a_target = torch.tensor(self.vae.fit_transform(a_target.numpy(), **condition_dict))
            else:
                a_target = torch.tensor(self.vae.transform(a_target.numpy(), **condition_dict))
            acs = torch.cat([acs[:, :-1, :].reshape(len(acs), -1), a_target.reshape(len(a_target), -1)], dim=1)
        else:
            acs = acs.reshape(len(acs), -1)

        input_feat = None
        # condition + target_A or learned latent
        if self.z_mode == 0:
            input_feat = torch.cat([acs, ls.reshape(len(ls), -1)], dim=1)

        # condition
        if self.z_mode == 1:
            pre_a = acs[:, :self.step * bert_dim]
            input_feat = torch.cat([pre_a, ls.reshape(len(ls), -1)], dim=1).reshape(len(acs), -1)

        # only as
        if self.z_mode == 2:
            input_feat = acs

        # only target A
        if self.z_mode == 3:
            input_feat = acs[:, self.step * bert_dim:]

        # only initial prompt
        if self.z_mode == 4:
            input_feat = acs[:, :bert_dim]

        # only ls
        if self.z_mode == 5:
            input_feat = ls

        if self.z_mode == 6:
            input_feat = torch.cat([condition_dict['condition'], acs[:, self.step * bert_dim:]], dim=1)

        if self.z_mode == 7:
            rand_feat = torch.rand(acs.shape[0], self.step * bert_dim)
            input_feat = torch.cat([rand_feat, acs[:, self.step * bert_dim:]], dim=1)

        if self.z_mode == 8:
            input_feat = acc_cond

        return input_feat

    def save_model(self, path):
        dump(self.model, path.replace('.pt', ''))

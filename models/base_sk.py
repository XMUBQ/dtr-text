from models.sk_model import sk_model
from joblib import dump
import torch
from utils import bert_dim, cuda_available


class base_sk(sk_model):
    def __init__(self, config):
        super(base_sk, self).__init__(config)

    def get_input_feat(self, data_iter, mode, test_counterfactual=False):
        acs = data_iter[mode]['acs'] if not test_counterfactual else torch.cat(
            [data_iter[mode]['acs'][:, :1, :], data_iter[mode]['counter_a']], dim=1)
        assert acs.shape==data_iter[mode]['acs'].shape
        ls = data_iter[mode]['ls']
        acc_cond = data_iter[mode]['acc_cond']

        condition_list = []
        for i in range(self.step):
            condition_list.append(
                {'condition': torch.cat([acs[:, :i + 1, :], ls[:, :i + 1, :]], dim=1).reshape(len(acs), -1)})

        # condition
        if self.decompose_cond:
            for i in range(self.step):
                if mode == 'train' and (not test_counterfactual):
                    condition_list[i]['condition'] = torch.tensor(
                        self.decompose_cond_pca.fit_transform(condition_list[i]['condition'].numpy()))
                else:
                    condition_list[i]['condition'] = torch.tensor(
                        self.decompose_cond_pca.transform(condition_list[i]['condition'].numpy()))

        # treatment
        learned_z = []
        if self.decompose_a:
            a_target = []
            for i in range(self.step):
                step_a_target = acs[:, i + 1, :]
                if mode == 'train' and (not test_counterfactual):
                    print("fit", i + 1, "th cVAE")
                    step_a_target = torch.tensor(
                        self.vae_list[i].fit_transform(step_a_target.numpy(), **condition_list[i]))
                else:
                    step_a_target = torch.tensor(self.vae_list[i].transform(step_a_target.numpy(), **condition_list[i]))
                a_target.append(step_a_target)
                learned_z.append(step_a_target.clone().cpu())
            a_target = torch.cat(a_target, dim=1)
            acs = torch.cat([acs[:, 0, :].reshape(len(acs), -1), a_target.reshape(len(a_target), -1)], dim=1)
        else:
            acs = acs.reshape(len(acs), -1)

        input_feat = None
        # acs + ls
        if self.z_mode == 0:
            input_feat = torch.cat([acs, ls.reshape(len(ls), -1)], dim=1)

        # condition
        if self.z_mode == 1:
            pre_a = acs[:, :-bert_dim if not self.decompose_a else self.decompose_a_dim]
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
            input_feat = acs[:, bert_dim:self.step * bert_dim]
            print(input_feat.shape)

        if self.z_mode == 7:
            rand_feat = torch.rand(acs.shape[0], self.step * bert_dim)
            input_feat = torch.cat([rand_feat, acs[:, self.step * bert_dim:]], dim=1)

        if self.z_mode == 8:
            input_feat = acc_cond
        return input_feat, learned_z

    def save_model(self, path):
        dump(self.model, path.replace('.pt', ''))
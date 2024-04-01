from utils import bert_dim
import numpy as np
import torch
from sklearn.decomposition import PCA
from models.sklearn_plus.sk_gMLP import sk_gMLP
from tqdm import tqdm
from utils import decompose_a_modelptr
from joblib import dump
from models.base_model import base_model


class g_adjust_model(base_model):
    def __init__(self, config):
        super(g_adjust_model, self).__init__(config)
        # cvae list
        self.decompose_a = config['decompose_a']
        self.decompose_a_dim = config['decompose_a_dim']
        self.decompose_a_model = config['decompose_a_model']
        self.vae_list = []
        for i in range(self.step):
            step_i_condition_size = (i + 1) * bert_dim * 2
            self.vae_list.append(decompose_a_modelptr[self.decompose_a_model](
                n_components=self.decompose_a_dim, outside_name=config['save_name'] + '_step_' + str(i),
                input_size=bert_dim, condition_size=step_i_condition_size, output_size=bert_dim))

        # decompose condition
        self.decompose_cond_pca = PCA(n_components=config['decompose_cond_dim'])  # this is just for decompose condition
        self.decompose_cond = config['decompose_cond']

        self.g_estimate_model = sk_gMLP(
            (self.decompose_a_dim if self.decompose_a else bert_dim) * (self.step - 1) + bert_dim * self.step, bert_dim,
            outside_name=config['save_name'])
        self.sample_bank = []

    def save_model(self, path):
        dump(self.model, path.replace('.pt', ''))

    def get_input_feat(self, data_iter, mode, test_counterfactual=False):
        acs = data_iter[mode]['acs'] if not test_counterfactual else torch.cat(
            [data_iter[mode]['acs'][:, :1, :], data_iter[mode]['counter_a']], dim=1)
        assert acs.shape == data_iter[mode]['acs'].shape
        ls = data_iter[mode]['ls']

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
        input_feat = torch.cat([acs, ls.reshape(len(ls), -1)], dim=1)

        return input_feat, learned_z

    def cal_g_estimate(self, input_feat, sample_bank, sample_times=50):
        g_estimation = []
        for i in tqdm(range(input_feat.shape[0])):
            acc_sample_prob = np.zeros((sample_times, 2))

            indices = torch.randint(low=0, high=sample_bank.size(0), size=(sample_times,))
            sampled_xl0 = sample_bank[indices]
            sample_x = sampled_xl0[:, :bert_dim]
            sample_l0 = sampled_xl0[:, bert_dim:]

            ls_i = input_feat[i, -bert_dim * self.step:]
            l_bar = torch.cat([sample_l0, ls_i[bert_dim:].repeat(sample_times, 1)], dim=1)
            acs_i = input_feat[i, :-bert_dim * self.step]
            a_bar = torch.cat([sample_x, acs_i[bert_dim:].repeat(sample_times, 1)], dim=1)
            new_input_feat = torch.cat([a_bar, l_bar], dim=1)

            for j in range(sample_times):
                samples_l_target, _ = self.g_estimate_model.predict(new_input_feat, step=self.step,
                                                                    decompose_a=self.decompose_a,
                                                                    decompose_a_dim=self.decompose_a_dim)
                ls = new_input_feat[:, -bert_dim * self.step:]
                acs = new_input_feat[:, :-bert_dim * self.step]
                sample_new_feat = torch.cat([acs, ls[:, :bert_dim * (self.step - 1)], samples_l_target], dim=1)
                sample_prob = self.model.predict_proba(sample_new_feat)
                acc_sample_prob += sample_prob
            acc_sample_prob /= sample_times
            g_estimation.append(acc_sample_prob.mean(axis=0))
        return np.array(g_estimation)

    def forward(self, data_iter, mode, test_counterfactual=False):
        data_iter = self.rebuild_data(data_iter)
        input_feat, learned_z = self.get_input_feat(data_iter, mode, test_counterfactual)
        input_feat = input_feat.reshape(len(input_feat), -1)
        y = np.array(data_iter[mode]['label'] if not test_counterfactual else data_iter[mode]['counter_label'])
        articles = data_iter[mode]['article']
        if mode == 'train' and (not test_counterfactual):
            print("fitting LR")
            self.model.fit(input_feat, y)
            # sample bank for g_estimation
            ls = input_feat[:, -bert_dim * self.step:]
            acs = input_feat[:, :-bert_dim * self.step]
            self.sample_bank = torch.cat([acs[:, :bert_dim], ls.reshape(len(ls), -1)[:, :bert_dim]], dim=1)

        output = self.model.predict(input_feat).tolist()
        prob = self.model.predict_proba(input_feat)

        g_estimation = []
        if mode == 'train' and (not test_counterfactual):
            print("train conditional generation for L_step")
            self.g_estimate_model.fit(input_feat, step=self.step, decompose_a=self.decompose_a,
                                      decompose_a_dim=self.decompose_a_dim)

        if mode == 'test':
            g_estimation = self.cal_g_estimate(input_feat, self.sample_bank)

        return output, prob, g_estimation, learned_z, y, articles

from utils import bert_dim, list2iter
import numpy as np
import torch
from sklearn.decomposition import PCA
from models.sklearn_plus.sk_gMLP import sk_gMLP
from tqdm import tqdm
from utils import learner2modelptr, learner2params, decompose_a_modelptr

class sk_model:
    def __init__(self, config):
        self.step = config['action_num']
        self.base_learner = config['learner']
        self.model = learner2modelptr[self.base_learner](**learner2params[self.base_learner])

        self.decompose_a = config['decompose_a']
        self.vae = decompose_a_modelptr[config['decompose_a_model']](
            n_components=config['decompose_a_dim'], outside_name=config['save_name'])  # this is for decompose a_step
        self.z_mode = config['z_mode']
        self.c_mode = config['c_mode']

        self.pca = PCA(n_components=config['decompose_cond_dim']) # this is just for decompose condition
        self.decompose_cond = config['decompose_cond']

        self.g_estimate = config['g_estimation']
        self.g_estimate_model = sk_gMLP(bert_dim * (2 * self.step - 1), bert_dim, outside_name=config['save_name'])
        self.sample_bank = []

    def save_model(self, path):
        raise NotImplementedError

    def get_input_feat(self, data_iter, mode):
        raise NotImplementedError

    def get_counterfactual_feat(self, data_iter):
        acs = data_iter['test']['acs']
        ls = data_iter['test']['ls']
        acc_cond = data_iter['test']['acc_cond']
        condition_dict = {'condition': torch.cat([acs[:, :-1, :], ls], dim=1).reshape(len(acs), -1)}
        if self.decompose_cond:
            condition_dict['condition'] = torch.tensor(self.pca.transform(condition_dict['condition'].numpy()))

        counter_a = data_iter['test']['counter_a']
        if self.decompose_a:
            counter_a = torch.tensor(self.vae.transform(counter_a.numpy(), **condition_dict))

        counter_y = data_iter['test']['counter_label']
        counter_feat = []

        # condition + counterfactual a
        if self.c_mode == 0:
            pre_a = acs[:, :-1, :]
            pre_feat = torch.cat([pre_a.reshape(len(acs), -1), counter_a], dim=1)
            counter_feat = torch.cat([pre_feat, ls.reshape(len(ls), -1)], dim=1)

        # condition
        if self.c_mode == 1:
            pre_a = acs[:, :-1, :]
            counter_feat = torch.cat([pre_a, ls], dim=1).reshape(len(acs), -1)

        # using only as
        if self.c_mode == 2:
            counter_feat = torch.cat([acs[:, :-1, :].reshape(len(acs), -1), counter_a], dim=1)

        # using only counterfactual target_A
        if self.c_mode == 3:
            counter_feat = counter_a

        # using initial prompt
        if self.c_mode == 4:
            counter_feat = acs[:, 0, :]

        # using only ls
        if self.c_mode == 5:
            counter_feat = ls

        if self.c_mode == 6:
            counter_feat = torch.cat([condition_dict['condition'], counter_a], dim=1)

        if self.c_mode == 7:
            rand_feat = torch.rand(acs.shape[0], self.step * bert_dim)
            counter_feat = torch.cat([rand_feat, counter_a], dim=1)

        if self.c_mode == 8:
            counter_feat = acc_cond

        return counter_feat.reshape(len(counter_feat), -1), counter_y

    def test_counterfactual(self, config, data_iter):
        data_iter = self.rebuild_data(data_iter)
        counter_feat, counter_y = self.get_counterfactual_feat(data_iter)
        counter_pred = self.model.predict(counter_feat).tolist()
        counter_prob = self.model.predict_proba(counter_feat)
        counter_g_estimation = []
        if self.g_estimate:
            counter_g_estimation = self.cal_g_estimate(counter_feat, self.sample_bank)
        print(len(counter_pred), len(counter_y))
        return counter_pred, counter_y, counter_prob, counter_g_estimation

    @staticmethod
    def rebuild_data(data_iter):
        train_list = data_iter['train']
        test_list = data_iter['test']
        return {'train': list2iter(train_list), 'test': list2iter(test_list)}

    def cal_g_estimate(self, input_feat, sample_bank, sample_times=50):
        g_estimation = []
        for i in tqdm(range(input_feat.shape[0])):
            acc_sample_prob=np.zeros((sample_times,2))

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
                samples_l_target, normal_prob = self.g_estimate_model.predict(new_input_feat, step=self.step)
                ls = new_input_feat[:, -bert_dim * self.step:]
                acs = new_input_feat[:, :-bert_dim * self.step]
                sample_new_feat = torch.cat([acs, ls[:, :bert_dim * (self.step - 1)], samples_l_target], dim=1)
                sample_prob = self.model.predict_proba(sample_new_feat)
                acc_sample_prob += sample_prob
            acc_sample_prob /= sample_times
            g_estimation.append(acc_sample_prob.mean(axis=0))
        return np.array(g_estimation)

    def optimal_solution(self, input_feat, raw_x):
        return []

    def forward(self, data_iter, mode):
        input_feat = self.get_input_feat(data_iter, mode)
        input_feat = input_feat.reshape(len(input_feat), -1)
        y = np.array(data_iter[mode]['label'])
        if mode == 'train':
            self.model.fit(input_feat, y)
            # sample bank for g_estimation
            ls = input_feat[:, -bert_dim * self.step:]
            acs = input_feat[:, :-bert_dim * self.step]
            self.sample_bank = torch.cat([acs[:, :bert_dim], ls.reshape(len(ls), -1)[:, :bert_dim]], dim=1)

        output = self.model.predict(input_feat).tolist()
        prob = self.model.predict_proba(input_feat)

        g_estimation = []
        if self.g_estimate:
            if mode == 'train':
                self.g_estimate_model.fit(input_feat, step=self.step)

            if mode == 'test':
                g_estimation = self.cal_g_estimate(input_feat, self.sample_bank)

        # optimal
        decision = self.optimal_solution(input_feat, data_iter[mode])
        return output, prob, g_estimation

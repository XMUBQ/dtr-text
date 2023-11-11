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

        # cvae list
        self.decompose_a = config['decompose_a']
        self.decompose_a_dim = config['decompose_a_dim']
        self.vae_list = []
        for i in range(self.step):
            self.vae_list.append(decompose_a_modelptr[config['decompose_a_model']](
                n_components=self.decompose_a_dim, outside_name=config['save_name'] + '_step_' + str(i)))
        self.z_mode = config['z_mode']

        # decompose condition
        self.decompose_cond_pca = PCA(n_components=config['decompose_cond_dim'])  # this is just for decompose condition
        self.decompose_cond = config['decompose_cond']

        self.g_estimate = config['g_estimation']
        self.g_estimate_model = sk_gMLP(
            (self.decompose_a_dim if self.decompose_a else bert_dim) * (self.step - 1) + bert_dim * self.step, bert_dim,
            outside_name=config['save_name'])
        self.sample_bank = []

    def save_model(self, path):
        raise NotImplementedError

    def get_input_feat(self, data_iter, mode, test_counterfactual=False):
        raise NotImplementedError

    def test_counterfactual(self, config, data_iter):
        data_iter = self.rebuild_data(data_iter)
        counter_y = data_iter['test']['counter_label']
        counter_feat, learned_counter_z = self.get_input_feat(data_iter, mode='test',
                                                                         test_counterfactual=True)
        counter_pred = self.model.predict(counter_feat).tolist()
        counter_prob = self.model.predict_proba(counter_feat)
        counter_g_estimation = []
        if self.g_estimate:
            counter_g_estimation = self.cal_g_estimate(counter_feat, self.sample_bank)
        return counter_pred, counter_y, counter_prob, counter_g_estimation, learned_counter_z

    @staticmethod
    def rebuild_data(data_iter):
        train_list = data_iter['train']
        test_list = data_iter['test']
        return {'train': list2iter(train_list), 'test': list2iter(test_list)}

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

    def optimal_solution(self, input_feat, raw_x):
        return []

    def forward(self, data_iter, mode):
        input_feat, learned_z = self.get_input_feat(data_iter, mode)
        input_feat = input_feat.reshape(len(input_feat), -1)
        y = np.array(data_iter[mode]['label'])
        if mode == 'train':
            print("fitting LR")
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
                print("train conditional generation for L_step")
                self.g_estimate_model.fit(input_feat, step=self.step, decompose_a=self.decompose_a,
                                          decompose_a_dim=self.decompose_a_dim)

            if mode == 'test':
                g_estimation = self.cal_g_estimate(input_feat, self.sample_bank)

        # optimal
        decision = self.optimal_solution(input_feat, data_iter[mode])
        return output, prob, g_estimation, learned_z

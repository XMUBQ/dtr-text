from skeleton.skeleton import skeleton
from models.model_factory import ModelFactory
from utils import ckpt_dir, output_dir
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tabulate import tabulate


class g_adjust_frame(skeleton):
    def __init__(self, config):
        super(g_adjust_frame, self).__init__(config)

    def update(self, model, config, data_iter):
        model.forward(data_iter, 'train')
        accumulate_pred, prob, g_estimation, test_learned_z, y, articles = model.forward(data_iter, 'test')
        return {'pred': accumulate_pred, 'true': y,
                'article': articles, 'g_estimation': g_estimation,
                'prob': prob, 'test_learned_z': test_learned_z}

    def train(self, config, fold_data, print_table=True):
        kf = fold_data['kfold']
        raw_list = fold_data['data']
        pred, true, g_estimation, prob, articles, test_learned_z = [], [], [], [], [], []
        counter_acc_output, counter_acc_true, counter_acc_prob, counter_acc_g_estimation, counter_acc_learned_z = [], [], [], [], []
        for i, (train_index, test_index) in enumerate(kf.split(raw_list)):
            print("Fold", i)
            config['fold'] = i
            model = ModelFactory.get_model(config)

            train_list = list(map(lambda idx: raw_list[idx], train_index))
            test_list = list(map(lambda idx: raw_list[idx], test_index))
            fold_iter = {'train': train_list, 'test': test_list}
            items = self.update(model, config, fold_iter)

            articles.extend(items['article'])
            pred.extend(items['pred'])
            true.extend(items['true'])
            prob.append(items['prob'])
            test_learned_z.append(items['test_learned_z'])
            if config['g_estimation']:
                g_estimation.append(items['g_estimation'])
                print(items['g_estimation'].mean(axis=0))

            model.save_model(ckpt_dir + config['save_name'] + '_' + str(i) + '.pt')

            counter_output, counter_prob, counter_g_estimation, learned_counter_z, counter_true, _ = model.forward(
                fold_iter, 'test', test_counterfactual=True)
            counter_acc_output.extend(counter_output)
            counter_acc_true.extend(counter_true)
            counter_acc_prob.append(counter_prob)
            if config['g_estimation']:
                counter_acc_g_estimation.append(counter_g_estimation)
            counter_acc_learned_z.append(learned_counter_z)

        # saving results
        prob = np.concatenate(prob, axis=0)
        df = pd.DataFrame(prob, columns=['p_casual', 'p_formal'])
        df['article'] = articles
        df['pred'] = pred
        df['true'] = true
        if config['g_estimation']:
            g_estimation = np.concatenate(g_estimation, axis=0)
            gdf = pd.DataFrame(g_estimation, columns=['gE0', 'gE1'])
            df = pd.concat([df, gdf], axis=1)

        if config['decompose_a']:
            for i in range(config['action_num']):
                step_learned_z = [zs[i] for zs in test_learned_z]
                step_learned_z = torch.cat(step_learned_z, dim=0)
                z_df = pd.DataFrame(step_learned_z.numpy(),
                                    columns=['z_' + str(i) + str(dim) for dim in range(step_learned_z.shape[1])])
                df = pd.concat([df, z_df], axis=1)

        print("Counterfactual performance")
        df['counter_pred'] = counter_acc_output
        df['counter_true'] = counter_acc_true
        counter_acc_prob = np.concatenate(counter_acc_prob, axis=0)
        cp_df = pd.DataFrame(counter_acc_prob, columns=['counter_p_casual', 'counter_p_formal'])
        df = pd.concat([df, cp_df], axis=1)

        if config['g_estimation']:
            counter_acc_g_estimation = np.concatenate(counter_acc_g_estimation, axis=0)
            counter_gdf = pd.DataFrame(counter_acc_g_estimation, columns=['counter_gE0', 'counter_gE1'])
            df = pd.concat([df, counter_gdf], axis=1)

        if config['decompose_a']:
            for i in range(config['action_num']):
                step_learned_counter_z = [zs[i] for zs in counter_acc_learned_z]
                step_learned_counter_z = torch.cat(step_learned_counter_z, dim=0)
                counter_z_df = pd.DataFrame(step_learned_counter_z.numpy(),
                                            columns=['counter_z_' + str(i) + str(dim) for dim in
                                                     range(step_learned_counter_z.shape[1])])
                df = pd.concat([df, counter_z_df], axis=1)

        df.to_csv(output_dir + config['save_name'] + '.tsv', sep='\t')

        if print_table and config['g_estimation']:
            headers = ["test - LR", "counterfactual - LR", "test - LR + G-E", "counterfactual - LR + G-E"]
            data = [nn.MSELoss()(torch.tensor(df.true.to_list()), torch.tensor(df.p_formal.to_list())),
                    nn.MSELoss()(torch.tensor(df.counter_true.to_list()), torch.tensor(df.counter_p_formal.to_list())),
                    nn.MSELoss()(torch.tensor(df.true.to_list()), torch.tensor(df.gE1.to_list())),
                    nn.MSELoss()(torch.tensor(df.counter_true.to_list()), torch.tensor(df.counter_gE1.to_list()))]

            formatted_data = [f"{value:.4f}" for value in data]
            print(tabulate([formatted_data], headers=headers, tablefmt="grid"))

        return

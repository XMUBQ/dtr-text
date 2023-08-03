import os
import json
from tqdm import tqdm
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import defaultdict
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from models.base_sk import base_sk
from argsparser import args
from skeleton.sk_frame import sk_frame
from utils import data_dir, cuda_available, eval_dir, save_all_step, ckpt_dir, get_bert_rep, meta_path, output_dir, \
    add_noise


def kfold_train(config, fold_data):
    kf = fold_data['kfold']
    raw_list = fold_data['data']
    pred, true, scores, g_estimation, prob, articles = [], [], [], [], [], []
    counter_acc_output, counter_acc_true, counter_scores, counter_acc_prob, counter_acc_g_estimation = [], [], [], [], []

    for i, (train_index, test_index) in enumerate(kf.split(raw_list)):
        print("Fold", i)
        config['fold'] = i
        model = load_model(config)
        if cuda_available and isinstance(model, nn.Module):
            model = model.cuda()

        train_list = list(map(lambda idx: raw_list[idx], train_index))
        test_list = list(map(lambda idx: raw_list[idx], test_index))
        fold_iter = {'train': train_list, 'test': test_list}
        items = config['skeleton'].update(model, config, fold_iter)

        articles.extend(items['article'])
        pred.extend(items['pred'])
        true.extend(items['true'])
        prob.append(items['prob'])
        scores.append(items['test_result']['weighted avg']['f1-score'])
        print(items['test_result'])
        if config['g_estimation']:
            g_estimation.append(items['g_estimation'])
            print(items['g_estimation'].mean(axis=0))
        config['skeleton'].save_model(model, ckpt_dir + config['save_name'] + '_' + str(i) + '.pt')

        if config['save_counterfactual']:
            counter_output, counter_true, counter_prob, counter_g_estimation = config['skeleton'].test_counterfactual(
                model, config,
                fold_iter)
            counter_acc_output.extend(counter_output)
            counter_acc_true.extend(counter_true)
            counter_acc_prob.append(counter_prob)
            counter_scores.append(
                classification_report(counter_true, counter_output, output_dict=True)['weighted avg']['f1-score'])
            if config['g_estimation']:
                counter_acc_g_estimation.append(counter_g_estimation)

    print("Observed performance")
    print(classification_report(true, pred, digits=3))
    print("Average score:", np.mean(scores), "Standard deviation:", np.std(scores))

    # saving results
    prob = np.concatenate(prob, axis=0)
    df = pd.DataFrame(prob, columns=['p_casual', 'p_formal'])
    df['article'] = articles
    df['pred'] = pred
    df['true'] = true
    if config['g_estimation']:
        g_estimation = np.concatenate(g_estimation, axis=0)
        mean_g_estimation = g_estimation.mean(axis=0)
        print("Average g_estimation:", mean_g_estimation)
        gdf = pd.DataFrame(g_estimation, columns=['gE0', 'gE1'])
        df = pd.concat([df, gdf], axis=1)

    if config['save_counterfactual']:
        print("Counterfactual performance")
        print(classification_report(counter_acc_true, counter_acc_output, digits=3))
        print("Average score:", np.mean(counter_scores), "Standard deviation:", np.std(counter_scores))
        df['counter_pred'] = counter_acc_output
        df['counter_true'] = counter_acc_true
        counter_acc_prob = np.concatenate(counter_acc_prob, axis=0)
        cp_df = pd.DataFrame(counter_acc_prob, columns=['counter_p_casual', 'counter_p_formal'])
        df = pd.concat([df, cp_df], axis=1)

        if config['g_estimation']:
            counter_acc_g_estimation = np.concatenate(counter_acc_g_estimation, axis=0)
            mean_counter_acc_g_estimation = counter_acc_g_estimation.mean(axis=0)
            print("Average counter g_estimation:", mean_counter_acc_g_estimation)
            counter_gdf = pd.DataFrame(counter_acc_g_estimation, columns=['counter_gE0', 'counter_gE1'])
            df = pd.concat([df, counter_gdf], axis=1)

    df.to_csv(output_dir + config['save_name'] + '.tsv', sep='\t')
    return classification_report(true, pred, digits=3, output_dict=True)


def load_tensor(config):
    articles = os.listdir(data_dir + eval_dir)

    meta_df = pd.read_csv(meta_path, sep='\t', index_col=0)
    session2user = defaultdict(lambda: -1, dict(zip(list(meta_df.session_id), list(meta_df.user_id))))
    session2topic = defaultdict(lambda: 19, dict(zip(list(meta_df.session_id), list(meta_df.topic_id))))

    tensor_list = []
    for article in tqdm(articles):
        df = pd.read_csv(data_dir + eval_dir + article, sep='\t', index_col=0).fillna('')
        if len(df) > config['action_num']:
            # quality label
            label = df.iloc[config['action_num']][config['reward']]

            # user_id
            user_id = session2user[article.replace('.tsv', '')]

            # topic id
            topic_id = session2topic[article.replace('.tsv', '')]

            # a_i
            cond_acs = list(df.a[:config['action_num']])
            cond_acs = [a.strip() for a in cond_acs]
            a = [df[config['target_a']][config['action_num']].strip()]
            acs = cond_acs + a

            # l_i
            raw_ls = [sorted(json.loads(l), key=lambda x: x['index'], reverse=False) for l in
                      list(df.l[:config['action_num']])]  # 'probability' if not seq_action else

            # user selection
            selects = list(df.select[:config['action_num']])

            # accumulative context
            acc_cond = df.acc_text[config['action_num'] - 1]

            # l_i, a_i representation
            ls = ['[SEP]'.join([op['trimmed'] for op in l]) for idx, l in enumerate(raw_ls)]

            acs_rep = get_bert_rep(acs).detach().cpu()
            if config['noise_on_treatment']:
                acs_rep[-1, :] = add_noise(acs_rep[-1, :])

            lcs_rep = get_bert_rep(ls).detach().cpu()

            # separate l_i representation and accumulative context representation
            sep_ls_rep = []
            for l in ls:
                sep_ls_rep.append(get_bert_rep(l.split('[SEP]')).detach().cpu())
            acc_rep = get_bert_rep([acc_cond]).detach().cpu()

            # counterfactual
            counter_a = df[config['target_counter_a']][config['action_num']]
            counter_a_rep = get_bert_rep([counter_a]).detach().cpu()
            counter_a_label = df[config['reward_counter']][config['action_num']]

            outputs = {'acs': acs_rep, 'ls': lcs_rep, 'sep_ls': sep_ls_rep, 'acc_cond': acc_rep,
                       'counter_a_rep': counter_a_rep}
            tensor_list.append((outputs, label, selects, article, user_id, topic_id, counter_a_label))

    config['total_user'] = len(set(session2user.values()))

    return tensor_list


def load_kfold_data(config, kfold=5):
    print("Loading experimental data...")
    raw_list = load_tensor(config)
    kf = KFold(n_splits=kfold, shuffle=True, random_state=16)
    kf.get_n_splits(raw_list)
    return {'data': raw_list, 'kfold': kf}


def load_model(config):
    model = None
    if config['model'] == 'vanilla':
        model = base_sk(config)
    # elif config['model'] == 'Q':
    #     model = reg_q_learning(config['action_num'])
    return model


def load_para():
    config = {'epoch': args.epoch, 'testgap': args.testgap, 'model': args.model, 'learner': args.learner,
              'embedding': args.embedding, 'label': args.label, 'reward': args.reward, 'target_a': args.target_a,
              'action_num': args.action_num, 'z_mode': args.z_mode, 'g_estimation': bool(args.g_estimation),

              'decompose_a': bool(args.decompose_a), 'decompose_a_dim': args.decompose_a_dim,
              'decompose_a_model': args.decompose_a_model,

              'decompose_cond': bool(args.decompose_cond),
              'decompose_cond_dim': args.decompose_cond_dim,

              'c_mode': args.c_mode, 'save_counterfactual': bool(args.save_counterfactual),
              'target_counter_a': args.target_counter_a, 'reward_counter': args.reward_counter,

              'noise_on_treatment': bool(args.noise_on_treatment)}

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam
    config['org_optim_func'] = optimizer
    config['loss_func'] = loss_function
    config['skeleton'] = sk_frame(config)
    config['train_func'] = kfold_train
    config['save_all_step'] = save_all_step[config['model']]

    save_name = '_'.join(
        [config['model'], config['learner'], config['embedding'], config['label'], config['reward'],
         str(config['action_num']), config['target_a'], 'decompose_a', str(config['decompose_a']), 'decompose_cond',
         str(config['decompose_cond']), 'z_mode',
         str(config['z_mode']), ])
    if config['decompose_a']:
        save_name += '_decompose_a_model_' + config['decompose_a_model'] + '_decompose_a_dim_' + str(
            config['decompose_a_dim'])
    if config['decompose_cond']:
        save_name += '_decompose_cond_dim_' + str(config['decompose_cond_dim'])
    if config['save_counterfactual']:
        save_name += '_' + config['target_counter_a'] + '_' + config['reward_counter']

    config['save_name'] = save_name
    print(config)
    return config


def load_pipelines():
    config = load_para()
    fold_data = load_kfold_data(config)
    return config, fold_data

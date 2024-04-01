import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from argsparser import args
from utils import save_all_step, eval_dir, meta_path, tokenizer, feature_model
from loader.loader_factory import DataLoaderFactory
from skeleton.frame_factory import FrameFactory

def load_kfold_data(config, kfold=5):
    print("Loading experimental data...")

    factory_loader= DataLoaderFactory.get_loader(config['data_name'], config, eval_dir, meta_path)
    raw_list=factory_loader.load_data()

    kf = KFold(n_splits=kfold, shuffle=True, random_state=16)
    kf.get_n_splits(raw_list)
    return {'data': raw_list, 'kfold': kf}

def load_para():
    config = {'model': args.model,
              'data_name': args.data_name, 'reward': args.reward,
              'target_a': args.target_a, 'action_num': args.action_num,

              'g_estimation': bool(args.g_estimation),

              'decompose_a': bool(args.decompose_a), 'decompose_a_dim': args.decompose_a_dim,
              'decompose_a_model': args.decompose_a_model,

              'decompose_cond': bool(args.decompose_cond),
              'decompose_cond_dim': args.decompose_cond_dim,

              'target_counter_a': args.target_counter_a, 'reward_counter': args.reward_counter,

              'noise_on_input': bool(args.noise_on_input), 'noise_level': args.noise_level, 'split': args.split}

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam
    config['org_optim_func'] = optimizer
    config['loss_func'] = loss_function
    config['skeleton'] = FrameFactory.get_frame(config)
    config['save_all_step'] = save_all_step[config['model']]
    config['tokenizer'] = tokenizer
    config['feature_model'] = feature_model

    save_name = '_'.join(
        [config['model'], config['data_name'], 'split', str(config['split']),
         config['reward'], str(config['action_num']), config['target_a'], 'decompose_a',
         str(config['decompose_a']), 'decompose_cond', str(config['decompose_cond'])])
    if config['decompose_a']:
        save_name += '_decompose_a_model_' + config['decompose_a_model'] + '_decompose_a_dim_' + str(
            config['decompose_a_dim'])
    if config['decompose_cond']:
        save_name += '_decompose_cond_dim_' + str(config['decompose_cond_dim'])
    if config['noise_on_input']:
        save_name += '_noise_level_' + str(config['noise_level'])

    save_name += '_random_seed_' + str(args.random_seed)

    config['save_name'] = save_name
    print(config['save_name'])
    return config


def load_pipelines():
    config = load_para()
    fold_data = load_kfold_data(config)
    return config, fold_data
import argparse

parser = argparse.ArgumentParser(description='No description')

# model TODO: future model extension
parser.add_argument('--model', type=str, default='vanilla')

# data info
parser.add_argument('--data_name', type=str, default='coauthor')
parser.add_argument('--action_num', type=int, default=2)
parser.add_argument('--target_a', type=str, default='causal_a')
parser.add_argument('--reward', type=str, default='causal_reward')
parser.add_argument('--decompose_a', type=int, default=0)
parser.add_argument('--decompose_a_model', type=str, default='cVAE')
parser.add_argument('--decompose_a_dim', type=int, default=50)
parser.add_argument('--decompose_cond', type=int, default=0)
parser.add_argument('--decompose_cond_dim', type=int, default=50)
parser.add_argument('--target_counter_a',type=str,default='counterfactual_a')
parser.add_argument('--reward_counter',type=str,default='counterfactual_reward')
parser.add_argument('--split',type=int,default=20)
parser.add_argument('--noise_on_input', type=int, default=1)
parser.add_argument('--noise_level',type=float,default=1)

# g estimation
parser.add_argument('--g_estimation', type=int, default=1)

# training parameters
parser.add_argument('--cuda', type=int, default=7)
parser.add_argument('--random_seed',type=int,default=80)

args = parser.parse_args()

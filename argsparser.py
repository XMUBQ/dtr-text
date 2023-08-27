import argparse

parser = argparse.ArgumentParser(description='No description')

# model
parser.add_argument('--model', type=str, default='vanilla')
parser.add_argument('--learner', type=str, default='LIN')

# data info
parser.add_argument('--data_name', type=str, default='coauthor')
parser.add_argument('--label', type=str, default='quality')
parser.add_argument('--action_num', type=int, default=2)
parser.add_argument('--target_a', type=str, default='causal_a')
parser.add_argument('--reward', type=str, default='causal_reward')
parser.add_argument('--decompose_a', type=int, default=0)
parser.add_argument('--decompose_a_model', type=str, default='cVAE')
parser.add_argument('--decompose_a_dim', type=int, default=50)
parser.add_argument('--decompose_cond', type=int, default=0)
parser.add_argument('--decompose_cond_dim', type=int, default=10)
parser.add_argument('--save_counterfactual', type=int, default=1)
parser.add_argument('--c_mode', type=int, default=0)
parser.add_argument('--target_counter_a',type=str,default='counterfactual_a')
parser.add_argument('--reward_counter',type=str,default='counterfactual_reward')

# z learning
parser.add_argument('--z_mode', type=int, default=0)

# g estimation
parser.add_argument('--g_estimation', type=int, default=1)

# training details
parser.add_argument('--cuda', type=int, default=7)
parser.add_argument('--epoch', type=int, default=1000)
parser.add_argument('--testgap', type=int, default=50)

# not used for now
parser.add_argument('--embedding', type=str, default='bert')

# temp args
parser.add_argument('--regularizer',type=float, default=1.0)
parser.add_argument('--split',type=int,default=20)
parser.add_argument('--noise_on_input', type=int, default=1)

# vae unused temporarily
parser.add_argument('--vae_arch', type=str, default='full')
parser.add_argument('--z_dim', type=int, default=10)
parser.add_argument('--z_cond', type=str, default='his')
parser.add_argument('--vae_loss', type=str, default='total')

args = parser.parse_args()

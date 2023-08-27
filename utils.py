from transformers import BertTokenizer, BertModel
import numpy as np
import torch
from collections import defaultdict
from argsparser import args

assert args.save_counterfactual == 0 or args.c_mode == args.z_mode

# constant
random_seed = 64  # or any of your favorite number
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

cuda_available = torch.cuda.is_available()
if cuda_available:
    dv = "cuda:" + str(args.cuda)
    device = torch.device(dv)
    torch.cuda.set_device(int(dv[-1]))
else:
    device = torch.device('cpu')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
feature_model = BertModel.from_pretrained('bert-base-uncased')
feature_model = feature_model.cuda()
feature_model.eval()

data_dir = 'data/'
output_dir = 'results/'
plot_dir = 'plot/'
ckpt_dir = 'ckpt/'

# coauthor related path
coauthor_eval_dir = str(args.split) + 'split-dtr-data/' # 'altera1-dtr-data/', 'temp-dtr-data/'
coauthor_meta_path = 'data/metadata.tsv'

# baize related path
baize_eval_dir = 'baize_temp_dir/'
baize_meta_path = 'data/baize/metadata.tsv'

bert_dim = 768
num_of_topic = 20

# nlp = stanza.Pipeline(lang='en', processors='tokenize')
seq_action_dict = defaultdict(lambda: True)
seq_action_dict['MLP'] = False


# function
def add_noise(outputs):
    noise = torch.randn(outputs.size())
    mean, std_dev = 0, 1.8
    adjusted_noise = noise * std_dev + mean
    outputs = outputs + adjusted_noise
    return outputs

def get_bert_rep(ss):
    input_feat = tokenizer.batch_encode_plus(ss, max_length=512,
                                             padding='longest',  # implements dynamic padding
                                             truncation=True,
                                             return_tensors='pt',
                                             return_attention_mask=True,
                                             return_token_type_ids=True
                                             )
    if cuda_available:
        input_feat['attention_mask'] = input_feat['attention_mask'].cuda()
        input_feat['input_ids'] = input_feat['input_ids'].cuda()

    with torch.no_grad():
        outputs = feature_model(input_feat['input_ids'],
                                attention_mask=input_feat['attention_mask']).pooler_output
    return outputs


def list2iter(tensor_list):
    tensor_rep, label, selects, article, user_id, topic_id, counter_a_label = zip(*tensor_list)
    all_acs = torch.cat([t['acs'].unsqueeze(0) for t in tensor_rep], dim=0)
    all_ls = torch.cat([t['ls'].unsqueeze(0) for t in tensor_rep], dim=0)
    all_acc_cond = torch.cat([t['acc_cond'] for t in tensor_rep])
    all_counter_a = torch.cat([t['counter_a_rep'] for t in tensor_rep])

    return {'acs': all_acs, 'ls': all_ls, 'acc_cond': all_acc_cond, 'counter_a': all_counter_a, 'label': label,
            'counter_label': counter_a_label,
            'selects': selects, 'article': article, 'user': user_id, 'topic': topic_id}


empty_embed = np.array(get_bert_rep(['']).squeeze().detach().cpu())

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from models.sklearn_plus.sk_MLP import sk_MLP
from models.sklearn_plus.sk_cVAE import sk_cVAE
from models.sklearn_plus.sk_vanilla_VAE import sk_vanilla_VAE
from models.sklearn_plus.sk_PCA import sk_PCA
from models.sklearn_plus.sk_reconsVAE import sk_reconsVAE

params = {'batch_size': 16, 'shuffle': False, 'drop_last': False, 'num_workers': 0}
learner2params = {'LIN': {'random_state': 42, "solver": 'liblinear', "C": args.regularizer},
                  'DET': {'random_state': 42, "criterion": "entropy", "max_depth": 10},
                  'MLP': {'input_size': (2 * args.action_num + 1) * bert_dim, 'output_size': 2, 'num_of_layer': 2,
                          'hidden': 128}}
learner2modelptr = {'LIN': LogisticRegression, 'DET': DecisionTreeClassifier, 'MLP': sk_MLP}
decompose_a_modelptr = {'VAE': sk_vanilla_VAE, 'cVAE': sk_cVAE, 'PCA': sk_PCA, 'reconsVAE': sk_reconsVAE}

save_all_step = defaultdict(lambda: False)

from transformers import BertTokenizer, BertModel
import numpy as np
import torch
from collections import defaultdict
from argsparser import args

# constant
random_seed = args.random_seed  # or any of your favorite number
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

# related path
eval_dir = args.data_name + str(args.split) + 'split/'
meta_path = 'data/' + args.data_name + '/metadata.tsv'

bert_dim = 768
num_of_topic = 20
seq_action_dict = defaultdict(lambda: True)
seq_action_dict['MLP'] = False


# function
def add_noise(outputs, noise_level=0):
    noise = torch.randn(outputs.size())
    mean, std_dev = 0, noise_level
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
        outputs = feature_model(input_feat['input_ids'], attention_mask=input_feat['attention_mask']).pooler_output
    return outputs
empty_embed = np.array(get_bert_rep(['']).squeeze().detach().cpu())


params = {'batch_size': 16, 'shuffle': False, 'drop_last': False, 'num_workers': 0}
from models.sklearn_plus.sk_cVAE import sk_cVAE
from models.sklearn_plus.sk_vanilla_VAE import sk_vanilla_VAE
from models.sklearn_plus.sk_PCA import sk_PCA

decompose_a_modelptr = {'VAE': sk_vanilla_VAE, 'cVAE': sk_cVAE, 'PCA': sk_PCA}

save_all_step = defaultdict(lambda: False)

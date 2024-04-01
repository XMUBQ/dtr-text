import torch
from sklearn.linear_model import LogisticRegression

class base_model:
    def __init__(self, config):
        self.step = config['action_num']
        self.model = LogisticRegression(**{'random_state': 42, "solver": 'liblinear', "C": 1.0})
        self.save_name = config['save_name']

    def save_model(self, path):
        raise NotImplementedError

    def get_input_feat(self, data_iter, mode, test_counterfactual=False):
        raise NotImplementedError

    @staticmethod
    def list2iter(tensor_list):
        tensor_rep, label, selects, article, user_id, topic_id, counter_a_label, acs_text, ls_text, counter_text = zip(
            *tensor_list)
        all_acs = torch.cat([t['acs'].unsqueeze(0) for t in tensor_rep], dim=0)
        all_ls = torch.cat([t['ls'].unsqueeze(0) for t in tensor_rep], dim=0)
        all_acc_cond = torch.cat([t['acc_cond'] for t in tensor_rep])
        all_counter_a = torch.cat([t['counter_a_rep'].unsqueeze(0) for t in tensor_rep], dim=0)

        return {'acs': all_acs, 'ls': all_ls, 'acc_cond': all_acc_cond, 'counter_a': all_counter_a, 'label': label,
                'counter_label': counter_a_label, 'counter_text': counter_text,
                'acs_text': acs_text, 'ls_text': ls_text,
                'selects': selects, 'article': article, 'user': user_id, 'topic': topic_id}

    def rebuild_data(self, data_iter):
        train_list = data_iter['train']
        test_list = data_iter['test']
        return {'train': self.list2iter(train_list), 'test': self.list2iter(test_list)}

    def forward(self, data_iter, mode, test_counterfactual=False):
        raise NotImplementedError

from loader.loader import loader
import json
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
from utils import get_bert_rep, add_noise


class LoadCoauthor(loader):
    def load_data(self):
        articles = self.read_articles()
        session2user = defaultdict(lambda: -1, dict(zip(list(self.meta_df.session_id), list(self.meta_df.user_id))))
        session2topic = defaultdict(lambda: 19, dict(zip(list(self.meta_df.session_id), list(self.meta_df.topic_id))))

        tensor_list = []
        for article in tqdm(articles):
            df = pd.read_csv(self.data_dir + self.eval_dir + article, sep='\t', index_col=0).fillna('')
            if len(df) > self.config['action_num']:
                # quality label
                label = df[self.config['reward']][self.config['action_num']]

                # user_id
                user_id = session2user[article.replace('.tsv', '')]

                # topic id
                topic_id = session2topic[article.replace('.tsv', '')]

                # a_i
                a_0 = [df.a[0]]
                obs_a = list(df[self.config['target_a']][1:self.config['action_num'] + 1])
                acs = a_0 + obs_a
                acs = [a.strip() for a in acs]

                # l_i
                raw_ls = [sorted(json.loads(l), key=lambda x: x['index'], reverse=False) for l in
                          list(df.l[:self.config['action_num']])]  # 'probability' if not seq_action else

                # user selection
                selects = list(df.select[:self.config['action_num']])

                # accumulative context
                acc_cond = df.acc_text[self.config['action_num'] - 1]

                # l_i, a_i representation
                ls = ['[SEP]'.join([op['trimmed'] for op in l]) for idx, l in enumerate(raw_ls)]

                acs_rep = get_bert_rep(acs).detach().cpu()
                if self.config['noise_on_input']:
                    acs_rep[1:, :] = add_noise(acs_rep[1:, :], noise_level=self.config['noise_level'])
                lcs_rep = get_bert_rep(ls).detach().cpu()
                acc_rep = get_bert_rep([acc_cond]).detach().cpu()

                # counterfactual
                counter_a = list(df[self.config['target_counter_a']][1: self.config['action_num'] + 1])
                counter_a_rep = get_bert_rep(counter_a).detach().cpu()
                if self.config['noise_on_input']:
                    counter_a_rep = add_noise(counter_a_rep, noise_level=self.config['noise_level'])
                counter_label = df[self.config['reward_counter']][self.config['action_num']]

                outputs = {'acs': acs_rep, 'ls': lcs_rep, 'acc_cond': acc_rep, 'counter_a_rep': counter_a_rep}
                tensor_list.append(
                    (outputs, label, selects, article, user_id, topic_id, counter_label, acs, ls, counter_a))

        self.config['total_user'] = len(set(session2user.values()))

        return tensor_list

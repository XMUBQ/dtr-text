from loader.loader import loader
import pandas as pd
from tqdm import tqdm
from utils import get_bert_rep, add_noise


class LoadDialcon(loader):
    def load_data(self):
        articles = self.read_articles()
        session2source = self.meta_df.set_index('dialogue_id')['source'].to_dict()
        session2minor = self.meta_df.set_index('dialogue_id')['TARGET'].to_dict()

        tensor_list = []
        for article in tqdm(articles):
            df = pd.read_csv(self.data_dir + self.eval_dir + article, sep='\t', index_col=0).fillna('')
            if len(df) >= max(self.config['action_num'], 2):
                # quality label
                label = df[self.config['reward']][self.config['action_num'] - 1]

                # source and target
                source = session2source[int(article.replace('.tsv', ''))]
                minor = session2minor[int(article.replace('.tsv', ''))]

                # a_i
                cond_acs = list(df[self.config['target_a']][:self.config['action_num']])
                acs = [minor] + cond_acs

                # l_i
                partial_ls = list(df.l[:self.config['action_num']])
                original_as = list(df.a[:self.config['action_num']])
                ls = [x + '[SEP]' + y for x, y in zip(partial_ls, original_as)]

                # accumulative context
                acc_cond = ' '.join([item for pair in zip(ls, acs) for item in pair][:-1])

                # representation of a_i, l_i, acc_cond
                acs_rep = get_bert_rep(acs).detach().cpu()
                if self.config['noise_on_input']:
                    acs_rep[1:, :] = add_noise(acs_rep[1:, :], noise_level=self.config['noise_level'])
                lcs_rep = get_bert_rep(ls).detach().cpu()
                acc_rep = get_bert_rep([acc_cond]).detach().cpu()

                # counterfactual
                counter_a = list(df[self.config['target_counter_a']][:self.config['action_num']])
                counter_a_rep = get_bert_rep(counter_a).detach().cpu()
                if self.config['noise_on_input']:
                    counter_a_rep = add_noise(counter_a_rep, noise_level=self.config['noise_level'])
                counter_label = df[self.config['reward_counter']][self.config['action_num'] - 1]

                outputs = {'acs': acs_rep, 'ls': lcs_rep, 'acc_cond': acc_rep, 'counter_a_rep': counter_a_rep}
                tensor_list.append((outputs, label, source, article, None, minor, counter_label, acs, ls, counter_a))

        return tensor_list

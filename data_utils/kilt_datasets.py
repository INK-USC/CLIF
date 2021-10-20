from .lamol_datasets import LAMOLDataset
import os

DATA_DIR='datasets/kilt/unified_individual'
PROMPTS = {
    'fever': 'does the paragraph support or refute the claim? ',
    'trex': 'which word to fill in? ',
    'structured_zeroshot': 'which word to fill in? ',
    'hotpotqa': '',
    'nq': '',
    'eli5':'',
    'wow': 'how to respond?',
    'aidayago2': '',
    'triviaqa': ''
}

class KILTDatasetWOContext(LAMOLDataset):
    def __init__(self, args, task_name, split, tokenizer, gen_token, full_init=True, **kwargs):
        super().__init__(args, task_name, split, tokenizer, gen_token, full_init=False, **kwargs)
        if split == 'test':
            self.split = 'dev'
        if full_init:
            self.init_data()

    def init_data(self):
        base_dir = os.path.join(DATA_DIR, 'kilt_{}_wctx'.format(self.task_name), self.task_name)
        query_file, context_file, answer_file = [os.path.join(base_dir,'{}-{}-{}.tsv'.format(self.task_name, self.split, x))
                                                 for x in ['query','ctx','answer']]
        query_lines, context_lines, answer_lines = open(query_file).readlines(), open(context_file).readlines(), \
                                                   open(answer_file).readlines()
        data = []
        prompt = PROMPTS[self.task_name]
        for query, context, answer in zip(query_lines, context_lines, answer_lines):
            query, context, answer = query.strip(), context.strip(), answer.strip()
            if prompt:
                entry = {
                    'context': query,
                    'qas':[{'question': ' ' + prompt, 'answers': [{'text': answer}]}]
                }
            else:
                entry = {
                    'context': '',
                    'qas':[{'question': query, 'answers': [{'text': answer}]}]
                }
            data.append(entry)
        self.data = data
        self.data_tokenization(self.data)


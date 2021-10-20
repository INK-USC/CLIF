from .lamol_datasets import *

DATA_DIR = 'datasets/leopard'

def get_label_space(train_instances):
    label_space = set()
    for instance in train_instances:
        label_space.add(instance['label'].lower())
    return label_space

def get_leopard_prompt(train_instances):
    item = train_instances[0]
    label_space = get_label_space(train_instances)
    if 'sentence1' and 'sentence2' in item:
        prompt = 'are two sentences {}?'.format(' or '.join(label_space))
    elif 'sentence1' in item:
        prompt = 'is this sentence {}?'.format(' or '.join(label_space))
    else:
        raise ValueError
    return prompt

def task_name_in_file(x):
    if x == 'conll':
        return 'conll_c'
    return x

class LeopardDataset(LAMOLDataset):
    def __init__(self, args, task_name, split, tokenizer, gen_token, split_id, k_shot, full_init=True, use_vocab_space=True, **kwargs):
        super().__init__(args, task_name, split, tokenizer, gen_token, full_init=False, use_vocab_space=use_vocab_space, **kwargs)
        if split in ['test','validation','eval']:
            self.split = 'eval'
        self.split_id = split_id
        self.k_shot = k_shot
        if full_init:
            self.init_data()

    def init_data(self):
        with open(os.path.join(DATA_DIR, self.task_name, '{}_train_{}_{}.json'.format(task_name_in_file(self.task_name), self.split_id, self.k_shot))) as f:
            train_instances = json.load(f)
        with open(os.path.join(DATA_DIR, self.task_name, '{}_eval.json'.format(task_name_in_file(self.task_name)))) as f:
            eval_instances = json.load(f)
        data = []
        sep_token = self.tokenizer.sep_token
        prompt = get_leopard_prompt(train_instances)
        instances = train_instances if self.split == 'train' else eval_instances

        label_set = set()
        for item in instances:
            label_set.add(item['label'].lower())
        label_set = list(label_set)

        for item in instances:
            if 'sentence2' in item:
                context = item['sentence1'] + ' ' + sep_token + ' ' + item['sentence2']
            else:
                context = item['sentence1']

            context = context.lower()
            answer = item['label'].lower()

            if prompt:
                entry = {
                    'context': context,
                    'qas':[{'question': prompt, 'answers': [{'text': answer, 'label': label_set.index(answer)}]}]
                }
            else:
                entry = {
                    'context': '',
                    'qas':[{'question': context, 'answers': [{'text': answer, 'label': label_set.index(answer)}]}]
                }
            data.append(entry)
        self.data = data
        self.data_tokenization(self.data)

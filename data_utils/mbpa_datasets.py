from .lamol_datasets import LAMOLDataset
import csv
import os
import pickle

DATA_PATH = 'datasets/mbpa_data/ordered_data'

INDIVIDUAL_CLASS_LABELS = {
    'yelp': {1: '1', 2: '2', 3: '3', 4: '4', 5: '5'},
    'dbpedia': {1: 'Company', 2: 'EducationalInstitution', 3: 'Artist',
                4: 'Athlete', 5: 'OfficeHolder', 6: 'MeanOfTransportation', 7: 'Building',
                8: 'NaturalPlace', 9: 'Village', 10: 'Animal', 11: 'Plant', 12: 'Album',
                13: 'Film', 14: 'WrittenWork'},
    'yahoo': {1: 'Society & Culture', 2: 'Science & Mathematics', 3: 'Health',
              4: 'Education & Reference', 5: 'Computers & Internet', 6: 'Sports',
              7: 'Business & Finance', 8: 'Entertainment & Music',
              9: 'Family & Relationships', 10: 'Politics & Government'},
    'amazon': {1: '1', 2: '2', 3: '3', 4: '4', 5: '5'},
    'agnews': {1: 'World', 2: 'Sports', 3: 'Business', 4: 'Sci/Tech'}
}

TC_ORDER = {
    1: ['yelp', 'agnews', 'dbpedia', 'amazon', 'yahoo'],
    2: ['dbpedia', 'yahoo', 'agnews', 'amazon', 'yelp'],
    3: ['yelp', 'yahoo', 'amazon', 'dbpedia', 'agnews'],
    4: ['agnews', 'yelp', 'amazon', 'yahoo', 'dbpedia']
}

def get_task(y, label_map):
    y = int(y)
    class_name = label_map[y]
    for task_name, dic in INDIVIDUAL_CLASS_LABELS:
        if class_name in dic.values():
            return task_name

class MBPADataset(LAMOLDataset):
    def __init__(self, args, task_name, split, tokenizer, gen_token, full_init=True, use_vocab_space=True, **kwargs):
        super().__init__(args, task_name, split, tokenizer, gen_token, full_init=False, use_vocab_space=use_vocab_space,
                         **kwargs)
        if split in ['validation']:
            self.split = 'test'
        self.split_id = args.split_id
        if full_init:
            self.init_data()

    def init_data(self,):
        if self.split == 'test':
            split_id = 4
        else:
            split_id = self.split_id
        with open(os.path.join(DATA_PATH, self.split, '{}.csv'.format(split_id))) as f:
            data_tab = csv.DictReader(f)
            items = [_ for _ in data_tab]
        with open(os.path.join(DATA_PATH, self.split, '{}.pkl'.format(split_id)), 'rb') as f:
            label_map = pickle.load(f)

        if ':' in self.task_name:
            _, task_id = self.task_name.split(':')
            task_id = int(task_id)
            if self.split == 'train':
                items = items[115000 * task_id: 115000 * (task_id + 1)]
            if self.split == 'test':
                task_name = TC_ORDER[self.split_id][task_id]
                test_task_id = TC_ORDER[4].index(task_name)
                items = items[7600 * test_task_id: 7600 * (test_task_id + 1)]

        data = []
        for item in items:
            #if item_task_name == self.task_name:
            content = item['content']
            label = int(item['labels'])
            label_name = label_map[label]
            entry = {
                'context': content,
                'qas': [{'question': '', 'answers': [{'text': label_name, 'label': label}]}]
            }
            data.append(entry)
        self.data = data
        self.data_tokenization(self.data)
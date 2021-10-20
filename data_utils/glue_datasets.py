from .lib import huggingface_datasets
from .lamol_datasets import LAMOLDataset
import os

# PROMPTS = {
#     'cola': 'is the text acceptable? ',
#     'sst2': 'is this review negative or positive? ',
#     'mrpc': 'are two sentences paraphrase? ',
#     'qqp': 'are two sentences paraphrase? ',
#     'stsb': 'how are two sentences similar? ',
#     'mnli': 'does one sentence entail or contradict another? ',
#     'qnli': 'does one sentence entail another? ',
#     'wnli': 'does one sentence entail another? ',
#     'rte': 'does one sentence entail another? '
# }
#
# LABEL_MAPPINGS = {
#     'cola': ['reject','accept'],
#     'sst2': ['negative','positive'],
#     'mrpc': ['paraphrase','not paraphrase'],
#     'qqp': ['paraphrase','not paraphrase'],
#     'stsb': ['not similar','similar'],
#     'mnli': ['contradict','neutral','entailment'],
#     'qnli': ['not entailment', 'entailment'],
#     'wnli': ['not entailment', 'entailment'],
#     'rte': ['not entailment', 'entailment']
# }

CHOICE_PROMPTS = {
    'cola': 'is this sentence unacceptable or acceptable? ',
    'sst2': 'is this sentence negative or positive? ',
    'mrpc': 'are two sentences equivalent or not equivalent? ',
    'qqp': 'are two sentences duplicate or not duplicate? ',
    'stsb': 'are two sentences not similar or similar? ',
    'mnli': 'are two sentences entailment or neutral or contradiction? ',
    'qnli': 'are two sentences entailment or not entailment? ',
    'wnli': 'are two sentences entailment or not entailment? ',
    'rte': 'are two sentences entailment or not entailment? '
}

CHOICE_LABEL_MAPPINGS = {
    'cola': ['not acceptable','acceptable'],
    'sst2': ['negative','positive'],
    'mrpc': ['not equivalent','equivalent'],
    'qqp': ['not duplicate','duplicate'],
    'stsb': ['not similar','similar'],
    'mnli': ['entail','neutral','contradict'],
    'qnli': ['entailment', 'not entailment'],
    'wnli': ['not entailment', 'entailment'],
    'rte': ['entailment', 'not entailment']
}

BIN_PROMPTS = {
    'cola': 'is this text acceptable? ',
    'sst2': 'is this review negative or positive? ',
    'mrpc': 'are two sentences paraphrase or not paraphrase? ',
    'qqp': 'are two sentences paraphrase or not paraphrase? ',
    'stsb': 'how are two sentences similar or not similar? ',
    'mnli': 'does one sentence entail, neutral or contradict another? ',
    'qnli': 'does one sentence entail or not entail another? ',
    'wnli': 'does one sentence entail or not entail another? ',
    'rte': 'does one sentence entail or not entail another? '
}

BIN_LABEL_MAPPINGS = {
    'cola': ['no','accepted'],
    'sst2': ['negative','positive'],
    'mrpc': ['paraphrase','not paraphrase'],
    'qqp': ['paraphrase','not paraphrase'],
    'stsb': ['similar','not similar'],
    'mnli': ['entail','neutral','contradict'],
    'qnli': ['not entail', 'entail'],
    'wnli': ['not entail', 'entailment'],
    'rte': ['not entail', 'entail']
}



class GLUEDataset(LAMOLDataset):
    def __init__(self, args, task_name, split, tokenizer, gen_token, full_init=True, use_vocab_space=True, **kwargs):
        super().__init__(args, task_name, split, tokenizer, gen_token, full_init=False, use_vocab_space=use_vocab_space, **kwargs)
        if split in ['test']:
            self.split = 'validation'
        if full_init:
            self.init_data()

    def init_data(self):
        #base_dir = os.path.join(DATA_DIR, 'kilt_{}_wctx'.format(self.task_name), self.task_name)
        #query_file, context_file, answer_file = [os.path.join(base_dir,'{}-{}-{}.tsv'.format(self.task_name, self.split, x))
        #                                         for x in ['query','ctx','answer']]
        #query_lines, context_lines, answer_lines = open(query_file).readlines(), open(context_file).readlines(), \
        #                                           open(answer_file).readlines()
        data = []
        prompt = CHOICE_PROMPTS[self.task_name]
        sep_token = self.tokenizer.sep_token
        dataset_all_splits = huggingface_datasets.load_dataset('glue', self.task_name)

        if self.task_name == 'mnli' and self.split == 'validation':
            dataset = [_ for _ in dataset_all_splits['validation_matched']] + [_ for _ in dataset_all_splits['validation_mismatched']]
        else:
            dataset = dataset_all_splits[self.split]

        for item in dataset:
            if 'sentence' in item:
                context = item['sentence']
            if 'sentence1' in item:
                context = item['sentence1'] + ' ' + sep_token + ' ' + item['sentence2']
            if 'question' in item:
                if 'sentence' in item:
                    context = item['question'] + ' ' + sep_token + ' ' + item['sentence']
                else:
                    context = item['question']
            if 'question1' in item:
                context = item['question1']  + ' ' + sep_token + ' ' + item['question2']
            if 'hypothesis' in item:
                context = item['hypothesis'] + ' ' + sep_token + ' ' + item['premise']

            if self.task_name == 'stsb':
                label_id = 0 if item['label'] < 2.5 else 1
            else:
                label_id = item['label']

            answer = CHOICE_LABEL_MAPPINGS[self.task_name][label_id]

            if prompt:
                entry = {
                    'context': context,
                    'qas':[{'question': prompt, 'answers': [{'text': answer, 'label': label_id}]}]
                }
            else:
                entry = {
                    'context': '',
                    'qas':[{'question': context, 'answers': [{'text': answer, 'label': label_id}]}]
                }
            data.append(entry)
        self.data = data
        self.data_tokenization(self.data)

from .lamol_datasets import LAMOLDataset
import os
import csv
from collections import defaultdict
import random

DATA_DIR = 'datasets/crossfit_data/data/'
SPLIT_SEEDS = [13, 21, 42, 87, 100]

PARTITIONS = {
    "train": [
        "superglue-rte", "tweet_eval-sentiment", "discovery", "glue-rte", "superglue-wsc", "scicite",
        "glue-mrpc", "tweet_eval-stance_hillary", "tweet_eval-offensive", "emotion", "hatexplain", "glue-cola", "sick",
        "paws", "ethos-sexual_orientation", "glue-qqp", "tweet_eval-emotion", "sms_spam",
        "health_fact", "glue-mnli", "imdb", "ethos-disability", "glue-wnli", "scitail", "trec-finegrained",
        "yahoo_answers_topics", "liar", "glue-sst2", "tweet_eval-stance_abortion", "circa", "tweet_eval-stance_climate",
        "glue-qnli", "tweet_eval-emoji", "ethos-directed_vs_generalized", "ade_corpus_v2-classification",
        "wiki_auto", "hate_speech_offensive", "superglue-wic", "google_wellformed_query",
        "tweet_eval-irony", "ethos-gender", "onestop_english", "trec", "rotten_tomatoes", "kilt_fever"
    ],
    "dev": ["tweet_eval-stance_feminist", "ethos-national_origin", "tweet_eval-hate", "ag_news", "amazon_polarity",
            "hate_speech18", "poem_sentiment", "climate_fever", "medical_questions_pairs", "tweet_eval-stance_atheism"],
    "test":
        ["superglue-cb", "dbpedia_14", "wiki_qa", "emo", "yelp_polarity", "ethos-religion","financial_phrasebank", "tab_fact", "anli", "ethos-race"]
}



class CrossFitCLSDataset(LAMOLDataset):
    def __init__(self, args, task_name, split, tokenizer, gen_token, split_id, full_init=True, use_vocab_space=True,
                 **kwargs):
        super().__init__(args, task_name, split, tokenizer, gen_token, full_init=False, use_vocab_space=use_vocab_space,
                         **kwargs)
        self.split = split
        self.split_id = split_id
        self.split_seed = SPLIT_SEEDS[self.split_id]
        self.merge_split = args.merge_split
        self.crossfit_k_shot = args.crossfit_k_shot

        if full_init:
            self.init_data()

    def trim_train_set(self, examples):
        label2examples = defaultdict(list)
        for x, y in examples:
            label2examples[y].append([x,y])
        for v in label2examples.values():
            random.Random(0).shuffle(v)
        ret = []
        for k, v in label2examples.items():
            ret.extend(v[:self.crossfit_k_shot])
        random.Random(0).shuffle(ret)
        return ret

    def init_data(self):
        cand_k = [16, 32, 64]
        f = None

        if not self.merge_split or self.split != 'train':
            data = []
            for k in cand_k:
                filename = os.path.join(DATA_DIR, self.task_name,
                                        '{}_{}_{}_{}.tsv'.format(self.task_name, k, self.split_seed, self.split))
                if os.path.isfile(filename):
                    f = open(filename)
            #reader = csv.reader(f, delimiter='\t')
            #data = [_ for _ in reader]
            lines = f.readlines()
            data.extend([line.strip().split('\t') for line in lines])
            #assert all([len(x) == 2 for x in data])
            data = self.filter_and_add_task(data)

        else:
            data = []
            for split_seed in SPLIT_SEEDS:
                for k in cand_k:
                    filename = os.path.join(DATA_DIR, self.task_name,
                                            '{}_{}_{}_{}.tsv'.format(self.task_name, k, split_seed, self.split))
                    if os.path.isfile(filename):
                        f = open(filename)
                #reader = csv.reader(f, delimiter='\t')
                lines = f.readlines()
                this_data = [line.strip().split('\t') for line in lines]

                this_data = self.filter_and_add_task(this_data)
                data.extend(this_data)
                assert all([len(x) == 2 for x in data])
        assert len(data)
        self.data_tokenization(data)

    def filter_and_add_task(self, data):
        ret = []
        for x in data:
            if len(x) > 2:
                x = [' '.join(x[:-1]), x[-1]]

            if len(x) == 2:
                ret.append([' [ {} ]{}'.format(self.task_name, x[0]), x[1]])
            else:
                print('error in', x)
        if self.split in ['train'] and self.crossfit_k_shot:
            ret = self.trim_train_set(ret)
        return ret

    def data_tokenization(self, data):
        self.data = self.tokenization_batch(data)

    def tokenization_batch(self, cqa_examples):
        context_questions = [_[0] for _ in cqa_examples]
        answers = [_[1] for _ in cqa_examples]
        labels = [-1 for _ in cqa_examples]
        if self.add_space:
            context_questions = [' ' + x for x in context_questions]
            answers = [' ' + x for x in answers]
        cq_inputs = self.tokenizer.batch_encode_plus(context_questions,
                                                     pad_to_max_length=True,
                                                     max_length=self.max_input_length,
                                                     )
        answer_inputs = self.tokenizer.batch_encode_plus(answers,
                                                         pad_to_max_length=True,
                                                         max_length=self.max_output_length,
                                                         )

        encoded_examples = []
        for i, (cq_input_ids, cq_input_masks, ans_input_ids, ans_input_masks, label) in \
                enumerate(zip(cq_inputs['input_ids'], cq_inputs['attention_mask'], answer_inputs['input_ids'],
                              answer_inputs['attention_mask'], labels)):
            # if self.add_space:
            #    ans_input_ids, ans_input_masks = ans_input_ids[1:], ans_input_masks[1:]
            # ans_input_ids = [1] * (len(cq_input_ids) - len(ans_input_ids)) + ans_input_ids
            # ans_input_ids = ans_input_ids[:len(cq_input_ids)]
            info = [cq_input_ids, cq_input_masks, ans_input_ids, ans_input_masks, label]
            encoded_examples.append(info)

        return encoded_examples

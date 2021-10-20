from .lamol_datasets import LAMOLDataset
import os
import csv

DATA_DIR = 'datasets/crossfit_data/data/'
SPLIT_SEEDS = [13,21,42,87,100]

PARTITIONS = {
    "train": [
        "ai2_arc",
        "aqua_rat",
        "boolq",
        "codah",
        "commonsense_qa",
        "cosmos_qa",
        "dream",
        "eli5-askh",
        "eli5-asks",
        "eli5-eli5",
        "freebase_qa",
        "hellaswag",
        "jeopardy",
        "kilt_hotpotqa",
        "kilt_nq",
        "kilt_trex",
        "kilt_zsre",
        "lama-conceptnet",
        "lama-google_re",
        "lama-squad",
        "lama-trex",
        "math_qa",
        "mc_taco",
        "numer_sense",
        "openbookqa",
        "qasc",
        "quail",
        "quarel",
        "quartz-no_knowledge",
        "quartz-with_knowledge",
        "race-high",
        "race-middle",
        "sciq",
        "search_qa",
        "social_i_qa",
        "squad-no_context",
        "superglue-copa",
        "superglue-multirc",
        "swag",
        "web_questions",
        "wino_grande",
        "wiqa"
    ],
    "dev": [],
    "test": [
        "adversarialqa",
        "biomrc",
        "duorc",
        "hotpot_qa",
        "quoref",
        "ropes",
        "squad-with_context",
        "superglue-record",
        "tweet_qa"
    ],
}


class CrossFitQADataset(LAMOLDataset):
    def __init__(self, args, task_name, split, tokenizer, gen_token, split_id, full_init=True, use_vocab_space=True, **kwargs):
        super().__init__(args, task_name, split, tokenizer, gen_token, full_init=False, use_vocab_space=use_vocab_space, **kwargs)
        self.split = split
        self.split_id = split_id
        self.split_seed = SPLIT_SEEDS[self.split_id]
        self.merge_split = args.merge_split

        if full_init:
            self.init_data()

    def init_data(self):
        cand_k = [16, 32, 64]
        f = None

        if not self.merge_split or self.split != 'train':
            for k in cand_k:
                filename = os.path.join(DATA_DIR, self.task_name, '{}_{}_{}_{}.tsv'.format(self.task_name, k, self.split_seed, self.split))
                if os.path.isfile(filename):
                    f = open(filename)
            reader = csv.reader(f, delimiter='\t')
            data = [_ for _ in reader]
        else:
            data = []
            for split_seed in SPLIT_SEEDS:
                for k in cand_k:
                    filename = os.path.join(DATA_DIR, self.task_name,
                                            '{}_{}_{}_{}.tsv'.format(self.task_name, k, split_seed, self.split))
                    if os.path.isfile(filename):
                        f = open(filename)
                reader = csv.reader(f, delimiter='\t')
                data.extend([_ for _ in reader])
        self.data_tokenization(data)


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
                                                     max_length=self.max_input_length)
        answer_inputs = self.tokenizer.batch_encode_plus(answers,
                                                         pad_to_max_length=True,
                                                         max_length=self.max_output_length,
                                                         )


        encoded_examples = []
        for i, (cq_input_ids, cq_input_masks, ans_input_ids, ans_input_masks, label) in \
                enumerate(zip(cq_inputs['input_ids'], cq_inputs['attention_mask'],answer_inputs['input_ids'],
                              answer_inputs['attention_mask'], labels)):
            #if self.add_space:
            #    ans_input_ids, ans_input_masks = ans_input_ids[1:], ans_input_masks[1:]
            #ans_input_ids = [1] * (len(cq_input_ids) - len(ans_input_ids)) + ans_input_ids
            #ans_input_ids = ans_input_ids[:len(cq_input_ids)]
            info = [cq_input_ids, cq_input_masks, ans_input_ids, ans_input_masks, label]
            encoded_examples.append(info)

        return encoded_examples

import os
from torch.utils.data import Dataset
from transformers.tokenization_utils import PreTrainedTokenizer
import json
from typing import Optional, List
import re
import logging
from collections import OrderedDict
import random

logger = logging.getLogger(__name__)

DATA_DIR = 'datasets/lamol/'

TASK_DICT = {
    "squad1": {
               "train":os.path.join(DATA_DIR,"squad-train-v1.1.json"),
               "eval":os.path.join(DATA_DIR,"squad-dev-v1.1.json"),
               "test":os.path.join(DATA_DIR,"squad-dev-v1.1.json"),

    },
    "squad2": {
               "train":os.path.join(DATA_DIR,"squad-train-v2.0.json"),
               "eval":os.path.join(DATA_DIR,"squad-dev-v2.0.json"),
               "test":os.path.join(DATA_DIR,"squad-dev-v2.0.json"),

    },
    "iwslt.en.de": {
               "train":os.path.join(DATA_DIR,"iwslt.en.de_to_squad-train-v2.0.json"),
               "eval":os.path.join(DATA_DIR,"iwslt.en.de_to_squad-dev-v2.0.json"),
               "test":os.path.join(DATA_DIR,"iwslt.en.de_to_squad-test-v2.0.json"),

    },
    "cnn_dailymail": {
               "train":os.path.join(DATA_DIR,"cnn_dailymail_to_squad-train-v2.0.json"),
               "eval":os.path.join(DATA_DIR,"cnn_dailymail_to_squad-dev-v2.0.json"),
               "test":os.path.join(DATA_DIR,"cnn_dailymail_to_squad-test-v2.0.json"),

    },
    "multinli.in.out": {
               "train":os.path.join(DATA_DIR,"multinli.in.out_to_squad-train-v2.0.json"),
               "eval":os.path.join(DATA_DIR,"multinli.in.out_to_squad-dev-v2.0.json"),
               "test":os.path.join(DATA_DIR,"multinli.in.out_to_squad-dev-v2.0.json"),

    },
    "sst": {
               "train":os.path.join(DATA_DIR,"sst_to_squad-train-v2.0.json"),
               "eval":os.path.join(DATA_DIR,"sst_to_squad-dev-v2.0.json"),
               "test":os.path.join(DATA_DIR,"sst_to_squad-test-v2.0.json"),

    },
    "srl": {
               "train":os.path.join(DATA_DIR,"srl_to_squad-train-v2.0.json"),
               "eval":os.path.join(DATA_DIR,"srl_to_squad-dev-v2.0.json"),
               "test":os.path.join(DATA_DIR,"srl_to_squad-test-v2.0.json"),

    },
    "zre": {
               "train":os.path.join(DATA_DIR,"zre_to_squad-train-v2.0.json"),
               "eval":os.path.join(DATA_DIR,"zre_to_squad-dev-v2.0.json"),
               "test":os.path.join(DATA_DIR,"zre_to_squad-test-v2.0.json"),

    },
    "woz.en": {
               "train":os.path.join(DATA_DIR,"woz.en_to_squad-train-v2.0.json"),
               "eval":os.path.join(DATA_DIR,"woz.en_to_squad-dev-v2.0.json"),
               "test":os.path.join(DATA_DIR,"woz.en_to_squad-test-v2.0.json"),

    },
    "wikisql": {
               "train":os.path.join(DATA_DIR,"wikisql_to_squad-train-v2.0.json"),
               "eval":os.path.join(DATA_DIR,"wikisql_to_squad-dev-v2.0.json"),
               "test":os.path.join(DATA_DIR,"wikisql_to_squad-test-v2.0.json"),

    },
    "schema": {
               "train":os.path.join(DATA_DIR,"schema_to_squad-train-v2.0.json"),
               "eval":os.path.join(DATA_DIR,"schema_to_squad-dev-v2.0.json"),
               "test":os.path.join(DATA_DIR,"schema_to_squad-test-v2.0.json"),

    },
    "ag": {
               "train":os.path.join(DATA_DIR,"ag_to_squad-train-v2.0.json"),
               "eval":os.path.join(DATA_DIR,"ag_to_squad-test-v2.0.json"),
               "test":os.path.join(DATA_DIR,"ag_to_squad-test-v2.0.json"),

    },
    "dbpedia": {
               "train":os.path.join(DATA_DIR,"dbpedia_to_squad-train-v2.0.json"),
               "eval":os.path.join(DATA_DIR,"dbpedia_to_squad-test-v2.0.json"),
               "test":os.path.join(DATA_DIR,"dbpedia_to_squad-test-v2.0.json"),

    },
    "yahoo": {
               "train":os.path.join(DATA_DIR,"yahoo_to_squad-train-v2.0.json"),
               "eval":os.path.join(DATA_DIR,"yahoo_to_squad-test-v2.0.json"),
               "test":os.path.join(DATA_DIR,"yahoo_to_squad-test-v2.0.json"),

    },
    "amazon": {
               "train":os.path.join(DATA_DIR,"amazon_to_squad-train-v2.0.json"),
               "eval":os.path.join(DATA_DIR,"amazon_to_squad-test-v2.0.json"),
               "test":os.path.join(DATA_DIR,"amazon_to_squad-test-v2.0.json"),

    },
    "yelp": {
               "train":os.path.join(DATA_DIR,"yelp_to_squad-train-v2.0.json"),
               "eval":os.path.join(DATA_DIR,"yelp_to_squad-test-v2.0.json"),
               "test":os.path.join(DATA_DIR,"yelp_to_squad-test-v2.0.json"),

    },
}


class LAMOLDataset(Dataset):
    def __init__(self, args, task_name, split, tokenizer: PreTrainedTokenizer,
                 gen_token=None, full_init=True, use_vocab_space=True, **kwargs):
        self.gen_token = gen_token
        self.use_vocab_space = use_vocab_space
        #if args.use_sep:
        #    self.sep_token = SPECIAL_TOKEN_IDS["sep_token"]
        #self.ans_token = SPECIAL_TOKEN_IDS["ans_token"]
        self.tokenizer = tokenizer
        self.eos_token = tokenizer.eos_token
        self.pad_token = tokenizer.pad_token
        self.sep_token = tokenizer.sep_token
        self.ans_token = args.ans_token
        self.use_sep = True
        self.max_input_length = args.max_input_length
        self.max_output_length = args.max_output_length
        self.debug = args.debug
        self.task_name = task_name
        self.split = split
        self.data = []
        self.max_a_len = []
        self.add_space = args.add_space
        self.lamol_format = kwargs.get('lamol_format', False)
        if full_init:
            self.init_data()

    def init_data(self):
        data_paths = [TASK_DICT[self.task_name][self.split]]

        if not isinstance(data_paths, list):
            data_paths = [data_paths]

        data = []
        for data_path in data_paths:
            if not data_path:
                continue
            with open(data_path, "r") as f:
                raw_ds = json.load(f)
            raw_ds = map(lambda x: x["paragraphs"], raw_ds["data"])
            d = []
            for raw_d in raw_ds:
                d.extend(raw_d)
            data += d

        self.data = []
        self.max_a_len = 0
        if len(data_paths) == 1 and data_paths[0] is not None and ('wiki' in data_paths[0] or 'woz' in data_paths[0]):
            # data = self._sort_by_index(data)
            # args.n_workers = 1
            if 'wiki' in data_paths[0]:
                answers_file = "wikisql_answers.json"
            elif 'woz' in data_paths[0]:
                answers_file = "woz.en_answers.json"
            with open(os.path.join(DATA_DIR, answers_file), "r") as f:
                self.answers = json.load(f)
        if len(data) > 0:
            self.data_tokenization(data)



    def etl_single_extra_data(self, data):
        gen_token = data[0]
        data = ' '.join([str(datum) for datum in data[1:]])
        try:
            if self.use_sep:
                context, qa = re.split(str(), data)
            else:
                context = ""
                qa = data
            question, answer = re.split(self.ans_token, qa)
            context = [int(c) for c in context.strip().split()]
            question = [int(q) for q in question.strip().split()]
            answer = [int(a) for a in re.sub(self.eos_token, "", answer).strip().split()]
            uid = uuid.uuid1().hex
            data = self.parse_example(gen_token, context, question, answer, uid)
        except ValueError:
            return
        return data

    def concat_example(self, gen_token, c, sep_token, q, ans_token, a, eos_token):
        example = sep_token + q + ans_token + a
        if len(example) + 1 > self.max_input_length:
            logger.warning('an example with len {} is too long!'.format(len(example) + 1))
            return
        example = gen_token + c[:self.max_input_length - len(example) - 1] + example + eos_token
        return example

    def parse_example(self, gen_token, context, question, answer, idx):
        if self.use_sep:
            cq_example = self.concat_example([], context, [self.sep_token], question, [self.ans_token], [], [])
            cqa_example = self.concat_example([], context, [self.sep_token], question, [self.ans_token], answer, [])
        else:
            cq_example = self.concat_example([], context, [], question, [self.ans_token], [], [])
            cqa_example = self.concat_example([], context, [], question, [self.ans_token], answer, [])
        Y_example = self.concat_example([], [], [], [], [], answer, [self.eos_token])
        Y_example = [-1] * (len(cqa_example) - len(Y_example)) + Y_example
        if self.use_sep:
            gen_X_example = self.concat_example([gen_token], context, [self.sep_token], question, [self.ans_token],
                                                answer, [])
            gen_Y_example = self.concat_example([], context, [self.sep_token], question, [self.ans_token], answer,
                                                [self.eos_token])
        else:
            gen_X_example = self.concat_example([gen_token], context, [], question, [self.ans_token], answer, [])
            gen_Y_example = self.concat_example([], context, [], question, [self.ans_token], answer, [self.eos_token])
        return cq_example, len(cq_example), cqa_example, len(cqa_example), Y_example, gen_X_example, gen_Y_example, idx

    def parallel_tokenization(self, d):
        examples = []
        context = self.tokenizer.encode(d["context"])
        max_a_len = 0
        for qa in d["qas"]:
            question = self.tokenizer.encode(qa["question"])

            raw_answers = qa["answers"]
            if len(raw_answers) == 0:
                assert qa["is_impossible"]
                raw_answers.append({"text": ""})

            answer = []
            for i, raw_answer in enumerate(raw_answers):
                answer.extend(self.tokenizer.encode(raw_answer["text"]))
                if i != len(raw_answers) - 1:
                    answer.append(self.pad_token)
            max_a_len = max(max_a_len, len(answer))

            examples.append(self.parse_example(self.gen_token, context, question, answer, qa.get("id", 0)))
        return examples, max_a_len

    def tokenization_batch_lamol(self, data):
        cqa_examples = []
        for d in data:
            context = d['context']
            for qa in d['qas']:
                question = qa['question']
                answer = []
                for i, raw_answer in enumerate(qa['answers']):
                    answer.append(raw_answer['text'])
                    if i != len(qa['answers']) - 1:
                        answer.append(self.pad_token)

                answer = ' '.join(answer)
                context_question = 'question: {} context: {} answer:'.format(question, context)
                context_question_answer = 'question: {} context: {} answer: {}'.format(question, context, answer)

                gen_x_example = 'gen ' + context_question_answer
                gen_y_example = context_question_answer # dummy

                if self.add_space:
                    context_question = ' ' + context_question
                    context_question_answer = ' ' + context_question_answer
                    answer = ' ' + answer
                    gen_x_example, gen_y_example = ' ' + context_question, ' ' + context_question_answer


                info = [context, question, context_question, answer, None, context_question_answer, gen_x_example, gen_y_example]

                # additional info
                if 'label' in qa['answers'][0]:
                    assert len(qa['answers']) == 1
                    info.append(qa['answers'][0]['label'])
                else:
                    info.append(-1)
                cqa_examples.append(info)

        context_questions = [_[2] for _ in cqa_examples]
        answers = [_[3] for _ in cqa_examples]
        labels = [_[4] for _ in cqa_examples]
        context_question_answers, gen_x_examples, gen_y_examples = [_[5] for _ in cqa_examples], [_[6] for _ in cqa_examples],\
                                                [_[7] for _ in cqa_examples]

        cq_inputs = self.tokenizer.batch_encode_plus(context_questions,
                                                     pad_to_max_length=True,
                                                     max_length=self.max_input_length)
        answer_inputs = self.tokenizer.batch_encode_plus(answers,
                                                         pad_to_max_length=True,
                                                         max_length=self.max_output_length,
                                                         )
        cqa_inputs, gen_x_inputs, gen_y_inputs = [self.tokenizer.batch_encode_plus(x,
                                                 pad_to_max_length=True,
                                                 max_length=self.max_input_length) for x in [context_question_answers,
                                                                            gen_x_examples,
                                                                            gen_y_examples]]
        encoded_examples = []
        for i, (cq_input_ids, cq_input_masks, ans_input_ids, ans_input_masks, cqa_input_ids,
                gen_x_input_ids, gen_y_input_ids) in \
                enumerate(zip(cq_inputs['input_ids'], cq_inputs['attention_mask'],answer_inputs['input_ids'],
                              answer_inputs['attention_mask'], cqa_inputs['input_ids'], gen_x_inputs['input_ids'],
                              gen_y_inputs['input_ids'])):
            if self.add_space:
                ans_input_ids, ans_input_masks = ans_input_ids[1:], ans_input_masks[1:]
            info = []
            encoded_examples.append([
                cq_input_ids, len(cq_input_ids), cqa_input_ids, len(cqa_input_ids), ans_input_ids,
                gen_x_input_ids, gen_y_input_ids
            ])

        return encoded_examples

    def tokenization_batch(self, data):
        cqa_examples = []
        for d in data:
            context = d['context']
            for qa in d['qas']:
                question = qa['question']
                answer = []
                for i, raw_answer in enumerate(qa['answers']):
                    answer.append(raw_answer['text'])
                    if i != len(qa['answers']) - 1:
                        answer.append(self.pad_token)

                answer = ' '.join(answer)
                context_question = 'question: {} context: {}'.format(question, context)
                if self.add_space:
                    context_question = ' ' + context_question
                    answer = ' ' + answer

                info = [context, question, context_question, answer]

                # additional info
                if 'label' in qa['answers'][0]:
                    assert len(qa['answers']) == 1
                    info.append(qa['answers'][0]['label'])
                else:
                    info.append(-1)
                cqa_examples.append(info)

        context_questions = [_[2] for _ in cqa_examples]
        answers = [_[3] for _ in cqa_examples]
        labels = [_[4] for _ in cqa_examples]

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
            if self.add_space:
                ans_input_ids, ans_input_masks = ans_input_ids[1:], ans_input_masks[1:]
            #ans_input_ids = [1] * (len(cq_input_ids) - len(ans_input_ids)) + ans_input_ids
            #ans_input_ids = ans_input_ids[:len(cq_input_ids)]
            info = [cq_input_ids, cq_input_masks, ans_input_ids, ans_input_masks, label]
            encoded_examples.append(info)

        return encoded_examples

    def data_tokenization(self, data):
        if self.lamol_format:
            self.data = self.tokenization_batch_lamol(data)
        else:
            self.data = self.tokenization_batch(data)

    def get_indices(self):
        return [d[-1] for d in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
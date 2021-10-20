from torch.utils.data import DataLoader, Dataset
from .datasets import get_dataset, LAMOLDataset, LEOPARD_TASKS, CROSSFIT_QA_TRAIN_TASKS, CROSSFIT_CLS_TRAIN_TASKS, \
    CROSSFIT_QA_TEST_TASKS, CROSSFIT_CLS_TEST_TASKS
from collections import defaultdict
import os, pickle
import random as _random
import logging
import torch
import numpy as np
import torch.nn.functional as F

logger = logging.getLogger(__name__)

DATA_SEED = 0
random = _random.Random(DATA_SEED)

def pad_tensor_sequence(x, max_len, pad_value):
    # time dimension is at dim 1
    if x.size(1) == max_len:
        return x
    if x.size(1) > max_len:
        raise ValueError(str((x.size(), max_len)))
    target_size = [_ for _ in x.size()]
    target_size[1] = max_len - x.size(1)
    pad = torch.full(target_size, pad_value, dtype=x.dtype).to(x.device)
    ret = torch.cat([x, pad], 1)
    return ret

def trim_batch(
                input_ids,
                pad_token_id,
                attention_mask=None,
                axis=0,
        ):
            """Remove columns that are populated exclusively by pad_token_id"""
            keep_column_mask = input_ids.ne(pad_token_id).any(dim=axis)
            if attention_mask is None:
                return input_ids[:, keep_column_mask] if axis == 0 else input_ids[keep_column_mask, :]
            else:
                return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])

class TaskSequence(object):
    def __init__(self, args, task_keys, tokenizer, few_shot=False, lamol_format=False):
        self.args = args
        self.k_shot = args.k_shot
        self.max_split_id = args.max_split_id
        self.task_keys = task_keys
        self.tokenizer = tokenizer
        self.encoded_dataset = defaultdict(dict)
        self.data_loader_maps = defaultdict(dict)
        self.label_vocab_space_map = defaultdict(set)
        self.label_class_space_map = defaultdict(list)
        self.use_cache = not args.no_cache
        self.add_space = args.add_space
        self.lamol_format = lamol_format

        self.few_shot = few_shot
        self.example_limit = args.example_limit # this is only for few shot learning
        self.train_limit = args.train_limit # this is for regular training

        if any([x in CROSSFIT_QA_TRAIN_TASKS or x in CROSSFIT_CLS_TRAIN_TASKS for x in task_keys]) and (not args.task_collection or 'crossfit' in args.task_collection):
            splits = ['train', 'test', 'dev']
        else:
            splits = ['train', 'test']
        joint_task_keys = '_'.join(self.task_keys)
        if args.task_collection and 'crossfit' in args.task_collection:
            joint_task_keys = args.task_collection
            if self.args.crossfit_k_shot > 0:
                joint_task_keys += '_{}shot'.format(self.args.crossfit_k_shot)
        if self.add_space:
            joint_task_keys += '_space'


        if args.merge_split:
            joint_task_keys += '_merge_split'
        elif args.split_id != -1:
            joint_task_keys += '_split_{}'.format(args.split_id)


        if self.lamol_format:
            joint_task_keys += '_lamol'
        if any([x in LEOPARD_TASKS for x in task_keys]) and self.k_shot != 16:
            joint_task_keys += '_{}shot'.format(self.k_shot)

        cache_file = 'datasets/cache_{}.pkl'.format(joint_task_keys)
        if self.args.max_input_length != 256:
            cache_file = 'datasets/cache_{}_{}'.format(self.args.max_input_length, joint_task_keys)

        if args. task_collection and 'crossfit' in args.task_collection and args.split_id != 1:
            cache_file += '_split_{}'.format(args.split_id)

        seq_class = None
        if not self.use_cache or not os.path.isfile(cache_file):
            for task_key in self.task_keys:
                for split in splits:
                    if task_key in LEOPARD_TASKS and (args.task_collection in [None, 'leopard']): # few shot learning datasets, specify shot num
                        for split_id in range(self.max_split_id):
                            dataset = get_dataset(args, task_key, split, tokenizer, split_id=split_id, k_shot=self.k_shot,
                                                  lamol_format=self.lamol_format)
                            task_name = '{}_{}_{}'.format(task_key, split_id, self.k_shot)
                            self.encoded_dataset[task_name][split] = dataset
                    elif task_key in CROSSFIT_QA_TRAIN_TASKS or task_key in CROSSFIT_QA_TEST_TASKS \
                            or task_key in CROSSFIT_CLS_TRAIN_TASKS or task_key in CROSSFIT_CLS_TEST_TASKS:
                        if seq_class not in [None, 'crossfit']:
                            raise ValueError
                        dataset = get_dataset(args, task_key, split, tokenizer, split_id=args.split_id)
                        self.encoded_dataset[task_key][split] = dataset
                    else:
                        dataset = get_dataset(args, task_key, split, tokenizer, lamol_format=self.lamol_format)
                        self.encoded_dataset[task_key][split] = dataset
            if self.use_cache:
                with open(cache_file,'wb') as wf:
                    pickle.dump(self.encoded_dataset, wf)
        else:
            with open(cache_file, 'rb') as f:
                self.encoded_dataset = pickle.load(f)

        for task_key in self.encoded_dataset:
            for split in splits:
                shuffle = split == 'train'
                if task_key == 'mbpa':
                    shuffle = False
                if self.args.enforce_train_shuffle and split == 'train':
                    shuffle = True
                batch_size = args.train_batch_size if split == 'train' else args.predict_batch_size
                if self.few_shot and split == 'train':
                    self.trim_subset(self.encoded_dataset[task_key][split], self.example_limit)
                elif split == 'train' and self.train_limit != -1:
                    self.trim_subset(self.encoded_dataset[task_key][split], self.train_limit)
                data = self.encoded_dataset[task_key][split]
                data_loader = DataLoader(data, shuffle=shuffle, batch_size=batch_size)
                self.data_loader_maps[task_key][split] = data_loader


        # state the label space from the training set
        for task_key in self.encoded_dataset:
            train_examples = [_ for _ in self.encoded_dataset[task_key]['train']]
            train_ans = [x[2] for x in train_examples]
            tokens = set()

            for ans in train_ans:
                for word in ans:
                    tokens.add(word)

            spec_token_ids = self.tokenizer.convert_tokens_to_ids([_ for _ in self.tokenizer.special_tokens_map.items()])
            tokens.update(spec_token_ids)
            self.label_vocab_space_map[task_key] = list(tokens)

            labels = set()
            if len(train_examples[0]) > 4:
                for example in train_examples:
                    labels.add(example[4])
            self.label_class_space_map[task_key] = sorted(list(labels))


    def trim_subset(self, dataset: LAMOLDataset, n):
        #logger.info("trimmed subset")
        if len(dataset.data) < n:
            logger.warning('{} < {} when trimming dataset on {}'.format(len(dataset.data), n, dataset.task_name))
        else:
            subset = random.sample(dataset.data, n)
            dataset.data = subset

    def get_dataloader_sequence_iterator(self):
        for task_key in self.data_loader_maps:
            if 'dev' in self.data_loader_maps[task_key]:
                data_loaders = [self.data_loader_maps[task_key][split] for split in ['train', 'dev', 'test']]
            else:
                data_loaders = [self.data_loader_maps[task_key][split] for split in ['train','test','test']]
            yield task_key, data_loaders

    def __len__(self):
        return len(self.data_loader_maps)

    def get_data_loader(self, task_key, split):
        data_loader = self.data_loader_maps[task_key][split]
        return data_loader

    def get_label_space_map(self, t='ans'):
        if t == 'ans':
            return self.label_vocab_space_map
        else:
            return self.label_class_space_map

    def get_mtl_dataloader(self, split, task_num):
        dummy_dataset = list(self.encoded_dataset.values())[0][split]
        shuffle = split == 'train'
        mtl_bal_sampling = split == 'train'
        batch_size = self.args.train_batch_size if split == 'train' else self.args.predict_batch_size
        data_loaders = []
        for task_id, task_key in enumerate(self.data_loader_maps):
            if task_id < task_num:
                data_loader = self.data_loader_maps[task_key][split]
                data_loaders.append(data_loader)

        mtl_dataloader = MTLDataloader(dummy_dataset, shuffle=shuffle, batch_size=batch_size,
                                   data_loaders=data_loaders, task_keys=self.task_keys, mtl_bal_sampling=mtl_bal_sampling,
                                   task_num=task_num, pad_token_id=self.tokenizer.pad_token_id, sqrt=self.args.sqrt
                                    )
        return mtl_dataloader

class MTLDataloader(DataLoader):
    def __init__(self, dummy_dataset, *args, **kwargs):
        data_loaders = kwargs.pop('data_loaders')
        mtl_bal_sampling = kwargs.pop('mtl_bal_sampling')
        self.task_keys = kwargs.pop('task_keys')
        self.task_num = kwargs.pop('task_num')
        self.pad_token_id = kwargs.pop('pad_token_id')
        self.sqrt = kwargs.pop('sqrt')
        super().__init__(dummy_dataset, *args, **kwargs)
        self.data_loaders = data_loaders
        self.mtl_bal_sampling = mtl_bal_sampling
        self.loader_len = None

        # just for random sampling of examples from a task
        self.task_iterators = [iter(x) for x in self.data_loaders]

    def __iter__(self):
        def mtl_data_iterator():
            draws = []
            for i in range(self.task_num):
                draws.extend([i] * len(self.data_loaders[i]))
            iterators = [iter(_) for _ in self.data_loaders]
            random.shuffle(draws)
            self.loader_len = len(draws)
            for loader_id in draws:
                iterator = iterators[loader_id]
                yield next(iterator)

        def mtl_bal_data_iterator():
            draws = []
            max_dataloader_len = max([len(x) for x in self.data_loaders])
            for i in range(self.task_num):
                if self.sqrt:
                    # x : max_dataloader_len = sqrt(len(x)) : sqrt(len(max_dataloader_len))
                    batch_num = int(max_dataloader_len * (len(self.data_loaders[i]) ** 0.5) // (max_dataloader_len ** 0.5))
                    draws.extend([i] * batch_num)
                else:
                    draws.extend([i] * max_dataloader_len)
            iterators = [iter(_) for _ in self.data_loaders]
            random.shuffle(draws)
            self.loader_len = len(draws)
            for loader_id in draws:
                task_name = self.task_keys[loader_id]
                iterator = iterators[loader_id]
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterators[loader_id] = iter(self.data_loaders[loader_id])
                    iterator = iterators[loader_id]
                    batch = next(iterator)
                yield (loader_id, task_name), batch



        if self.mtl_bal_sampling:
            return mtl_bal_data_iterator()
        else:
            return mtl_data_iterator()

    def concat_tensors_with_pad(self, l, pad_value):
        if len(l[0].size()) < 2:
            return torch.cat(l, 0)
        max_len = max([x.size(1) for x in l])
        padded = []
        for x in l:
            pad_x = pad_tensor_sequence(x, max_len, pad_value)
            padded.append(pad_x)
        padded = torch.cat(padded, 0)
        return padded

    def get_full_batch_from_task(self, task_id, config):
        tmp_dataloader = DataLoader(self.data_loaders[task_id].dataset, batch_size=len(self.data_loaders[task_id].dataset))
        batch = next(iter(tmp_dataloader))
        cq_inputs_sample, cq_attention_mask_sample, ans_inputs_sample, ans_attention_mask_sample = [
            torch.stack(x, 0).transpose(0, 1).cuda() for x in batch[0:4]]
        cq_inputs_sample, cq_attention_mask_sample = trim_batch(cq_inputs_sample, config.pad_token_id,
                                                                cq_attention_mask_sample)
        ans_inputs_sample, ans_attention_mask_sample = trim_batch(ans_inputs_sample, config.pad_token_id,
                                                                  ans_attention_mask_sample)
        labels = batch[4].cuda()
        return cq_inputs_sample, cq_attention_mask_sample, ans_inputs_sample, ans_attention_mask_sample, labels

    def sample_batch_from_task(self, task_id, cat=False, config=None, k=1):
        if not cat and k != 1:
            raise NotImplementedError
        if not cat:
            return self._sample_batch_from_task(task_id, cat, config, k)
        else:
            l = []
            for i in range(k):
                sampled_batch = self._sample_batch_from_task(task_id, cat, config, k)
                l.append(sampled_batch)
            cat_batch = []
            # [encoder input, encoder attn, decoder input, decoder attn, label]
            pad_values = [self.pad_token_id, 0, self.pad_token_id, 0, -1]
            for i in range(len(l[0])):
                pad_value = pad_values[i] if i < len(pad_values) else -1
                x = self.concat_tensors_with_pad([x[i] for x in l], pad_value)
                cat_batch.append(x)
            return cat_batch

    def _sample_batch_from_task(self, task_id, cat=False, config=None, k=1):
        try:
            batch = next(self.task_iterators[task_id])
        except StopIteration:
            self.task_iterators[task_id] = iter(self.data_loaders[task_id])
            batch = next(self.task_iterators[task_id])
        if cat:
            cq_inputs_sample, cq_attention_mask_sample, ans_inputs_sample, ans_attention_mask_sample = [
                torch.stack(x, 0).transpose(0, 1).cuda() for x in batch[0:4]]
            cq_inputs_sample, cq_attention_mask_sample = trim_batch(cq_inputs_sample, config.pad_token_id, cq_attention_mask_sample)
            ans_inputs_sample, ans_attention_mask_sample = trim_batch(ans_inputs_sample, config.pad_token_id, ans_attention_mask_sample)
            labels = batch[4].cuda()
            return cq_inputs_sample, cq_attention_mask_sample, ans_inputs_sample, ans_attention_mask_sample, labels
        else:
            return batch

    def __len__(self):
        return self.loader_len
        #if self.mtl_bal_sampling:
        #    max_dataloader_len = max([len(x) for x in self.data_loaders])
        #    return max_dataloader_len * len(self.data_loaders)
        #else:
        #    return sum([len(_) for _ in self.data_loaders])



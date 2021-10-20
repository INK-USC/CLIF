import transformers
import torch.nn as nn
import csv
import os
import torch
import pickle


def add_special_tokens(model, tokenizer, args):
    special_tokens = {
        'sep_token': args.sep_token,
        'additional_special_tokens': [args.ans_token] if not args.ans_token in tokenizer.encoder else []
    }
    tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))
    return tokenizer


def freeze_params(model: nn.Module):
    """Set requires_grad=False for each of model.parameters()"""
    for par in model.parameters():
        par.requires_grad = False


def freeze_embeds(model):
    """Freeze token embeddings and positional embeddings for bart, just token embeddings for t5."""
    model_type = model.config.model_type

    if model_type == "t5":
        freeze_params(model.shared)
        for d in [model.encoder, model.decoder]:
            freeze_params(d.embed_tokens)
    elif model_type == "fsmt":
        for d in [model.model.encoder, model.model.decoder]:
            freeze_params(d.embed_positions)
            freeze_params(d.embed_tokens)
    else:
        freeze_params(model.model.shared)
        for d in [model.model.encoder, model.model.decoder]:
            freeze_params(d.embed_positions)
            freeze_params(d.embed_tokens)

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

def get_words(tokens):
    ret = []
    for token in tokens:
        if token == '</s>':
            break
        ret.append(token)
    return ret


def convert_to_single_gpu(state_dict):
    def _convert(key):
        if key.startswith('module.'):
            return key[7:]
        return key

    return {_convert(key): value for key, value in state_dict.items()}

def save_predictions(config, gts, preds, questions, tokenizer, filename):
    if type(preds[0]) is not int:
        preds = [tokenizer.convert_tokens_to_string(get_words(x)) for x in preds]
        gts = [tokenizer.convert_tokens_to_string(get_words(x))for x in gts]
    questions = [tokenizer.convert_tokens_to_string(get_words(x)) for x in questions]
    with open(os.path.join(config.output_dir, filename),'w') as wf:
        writer = csv.writer(wf)
        writer.writerows([(a,b,c) for a,b,c in zip(questions, gts, preds)])
        wf.close()


def save_state(args, model, optimizer, scheduler):
    model_state_dict = {k: v.cpu() for (k, v) in model.state_dict().items()}
    optimizer_state_dict = {k: v for (k,v) in optimizer.state_dict().items()}
    scheduler_state_dict = {k: v for (k,v) in scheduler.state_dict().items()}
    #torch.save(model_state_dict, os.path.join('/tmp', args.output_dir, "model.tmp.pt"))
    #torch.save(optimizer.state_dict(), os.path.join('/tmp', args.output_dir, 'optimizer.tmp.pt'))
    #torch.save(scheduler.state_dict(), os.path.join(args.output_dir, 'scheduler.tmp.pt'))
    all_states = [model_state_dict, optimizer_state_dict, scheduler_state_dict]
    return all_states

def load_state(args, model, optimizer, scheduler, all_states, strict=True):
    # load the saved model back
    model.load_state_dict(convert_to_single_gpu(all_states[0]), strict=strict)
    optimizer.load_state_dict(all_states[1])
    scheduler.load_state_dict(all_states[2])


def load_best_checkpoint(args, model, optimizer=None, scheduler=None, postfix='', use_tmp=False):
    if use_tmp:
        output_dir = os.path.join('/tmp/runs-continual-meta/', args.output_dir)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
    else:
        output_dir = args.output_dir
    # load the saved model back
    if postfix:
        model.load_state_dict(convert_to_single_gpu(torch.load(os.path.join(output_dir, 'best-model{}.pt'.format(postfix)))) )
        print()
    else:
        model.load_state_dict(convert_to_single_gpu(torch.load(os.path.join(output_dir, args.predict_checkpoint))) )
    if optimizer is not None:
        if os.path.isfile(os.path.join(output_dir, 'optimizer{}.pt'.format(postfix))):
            optimizer.load_state_dict(torch.load(os.path.join(output_dir, 'optimizer{}.pt'.format(postfix))))
        if os.path.isfile(os.path.join(output_dir, 'scheduler{}.pt'.format(postfix))):
            scheduler.load_state_dict(torch.load(os.path.join(output_dir, 'scheduler{}.pt'.format(postfix))))

def save_best_checkpoint(args, model, optimizer=None, scheduler=None, postfix='', use_tmp=False):
    if use_tmp:
        output_dir = os.path.join('/tmp/runs-continual-meta/', args.output_dir)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
    else:
        output_dir = args.output_dir

    model_state_dict = {k: v.cpu() for (k, v) in model.state_dict().items()}
    torch.save(model_state_dict, os.path.join(output_dir, "best-model{}.pt".format(postfix)))
    if optimizer is not None:
        torch.save(optimizer.state_dict(), os.path.join(output_dir, 'optimizer{}.pt'.format(postfix)))
        torch.save(scheduler.state_dict(), os.path.join(output_dir, 'scheduler{}.pt'.format(postfix)))

def remove_checkpoints_if_possible(args, output_dir, task_id):
    if args.skip_intermediate_ckpt:
        for i in range(task_id):
            for filename in ['best-model_task_{}.pt','optimizer_task_{}.pt','scheduler_task_{}.pt']:
                filename = os.path.join(output_dir, filename.format(i))
                if os.path.exists(filename):
                    os.remove(filename)


def lazy_save_best_checkpoint(args, model, optimizer=None, scheduler=None, postfix='', use_tmp=False, current_task_id=-1):
    if use_tmp:
        output_dir = os.path.join('/tmp/runs-continual-meta/', args.output_dir)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
    else:
        output_dir = args.output_dir

    remove_checkpoints_if_possible(args, output_dir, current_task_id)

    model_state_dict = {k: v.cpu() for (k, v) in model.state_dict().items()}
    save_args = {os.path.join(output_dir, "best-model{}.pt".format(postfix)): model_state_dict}
    if optimizer is not None:
        save_args[os.path.join(output_dir, 'optimizer{}.pt'.format(postfix))] = optimizer.state_dict()
        save_args[os.path.join(output_dir, 'scheduler{}.pt'.format(postfix))] = scheduler.state_dict()
    return save_args

def exec_save_best_checkpoint(save_args):
    for path, dic in save_args.items():
        torch.save(dic, path)

def adjust_learning_rate(optimizer: torch.optim.Optimizer, **kwargs):
    for param in optimizer.param_groups:
        for k, v in kwargs.items():
            param[k] = v

def freeze_layer_norm(model):
    for name, module in model.named_modules():
        if 'layer_norm' in name:
            print('Froze {}'.format(name))
            module.eval()

def get_trainable_params(args, model, train_all=False):
    no_decay = ['bias', 'LayerNorm.weight']

    if args.train_all or train_all:
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
    elif args.train_flex:
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.weight_generator.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.weight_generator.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0},
            #{'params': [p for n, p in model.basic_task_encoder.named_parameters() if not any(nd in n for nd in no_decay)],
            # 'weight_decay': args.weight_decay},
            #{'params': [p for n, p in model.bart_model.named_parameters() if not any(nd in n for nd in no_decay)],
            # 'weight_decay': args.weight_decay},
            #{'params': [p for n, p in model.bart_model.named_parameters() if any(nd in n for nd in no_decay)],
            # 'weight_decay': 0.0}
        ]
    else:
        if args.no_param_gen: # directly train adapters
            optimizer_grouped_parameters = [
                {'params': [p for n, p in model.named_parameters() if
                            not any(nd in n for nd in no_decay) and 'adapter' in n],
                 'weight_decay': args.weight_decay},
                {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)
                            and 'adapter' in n],
                 'weight_decay': 0.0}
            ]
        else:
            optimizer_grouped_parameters = [
                {'params': [p for n, p in model.weight_generator.named_parameters() if not any(nd in n for nd in no_decay)],
                 'weight_decay': args.weight_decay},
                {'params': [p for n, p in model.weight_generator.named_parameters() if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0}
            ]

    if args.train_task_embs or args.train_flex and not args.train_all:
        task_emb_params = [p for n, p in model.named_parameters() if 'stored_task_embs' in n]
        assert len(task_emb_params) == 1
        optimizer_grouped_parameters.append(
            {'params': task_emb_params,
             'weight_decay': 0}
        )

    return optimizer_grouped_parameters

def count_optimized_params(optimized_grouped_parameters):
    """
    returns the total number of parameters used by `m` (only counting
    shared parameters once); if `only_trainable` is True, then only
    includes parameters with `requires_grad = True`
    """
    all_params = []
    for dic in optimized_grouped_parameters:
        all_params.extend(dic['params'])
    unique = dict((p.data_ptr(), p) for p in all_params).values()

    return sum(p.numel() for p in unique)

def count_params(m: torch.nn.Module, only_trainable: bool = False):
    """
    returns the total number of parameters used by `m` (only counting
    shared parameters once); if `only_trainable` is True, then only
    includes parameters with `requires_grad = True`
    """
    parameters = m.parameters()
    if only_trainable:
        parameters = list(p for p in parameters if p.requires_grad)
    unique = dict((p.data_ptr(), p) for p in parameters).values()
    return sum(p.numel() for p in unique)

def get_batch_infinite(config, data_loader):
    while True:
        for batch in data_loader:
            cq_inputs, cq_attention_mask, ans_inputs, ans_attention_mask = [torch.stack(x, 0).transpose(0, 1).cuda() for
                                                                            x in batch[0:4]]
            cq_inputs, cq_attention_mask = trim_batch(cq_inputs, config.pad_token_id, cq_attention_mask)
            ans_inputs, ans_attention_mask = trim_batch(ans_inputs, config.pad_token_id, ans_attention_mask)
            lb = batch[4] if len(batch) > 4 else None
            yield cq_inputs, cq_attention_mask, ans_inputs, ans_attention_mask


def store_adapter_weights(args, adapter_weights, task_name):
    filename = '{}_adapter.pkl'.format(task_name)
    if args.load_adapter_postfix:
        filename = '{}_adapter_{}.pkl'.format(task_name, args.load_adapter_postfix)
    with open(os.path.join(args.output_dir, filename),'wb') as wf:
        pickle.dump([_.cpu().detach() for _ in adapter_weights], wf)

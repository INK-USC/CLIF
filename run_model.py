import os
import numpy as np
import torch

from transformers import BartTokenizer, BartConfig
from transformers import AdamW, get_linear_schedule_with_warmup, BartForConditionalGeneration
from nets.adapter_bart import BartForConditionalGenerationWithAdapter

from nets.adapter_bart import BartWithAdapterConfig
from nets.cl_model import ConditionedHyperNetForCL, ConditionalHyperNetL2Reg

from nets.regularizers import Weight_Regularized_AdamW

from data_utils.cl_dataloder import TaskSequence, DataLoader
from data_utils.datasets import get_main_metrics

from utils.misc import add_special_tokens, trim_batch, save_predictions, convert_to_single_gpu, \
    load_best_checkpoint, load_state, save_best_checkpoint, save_state, adjust_learning_rate, \
    freeze_layer_norm, get_trainable_params, count_optimized_params, count_params, \
    lazy_save_best_checkpoint, exec_save_best_checkpoint, get_batch_infinite, store_adapter_weights
from configs import get_args, merge_args_into_config
from metrics.em import exact_match_acc
from metrics.squad_f1 import f1_score_tokens_simple
import traceback

import random
import logging
import copy

from tqdm import tqdm

TRIM_FLG = 0


def get_optimizer(args, model, optimizer_grouped_parameters):
    if args.cl_method in ['ewc']:
        logger.info('using weighted regularized adamw optimizer because cl method is {}'.format(args.cl_method))
        optimizer = Weight_Regularized_AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        optimizer.set_model(model)
    else:
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.total_steps)
    return optimizer, scheduler


def run(args, logger):
    tokenizer = BartTokenizer.from_pretrained(args.model)
    main_task_sequence = TaskSequence(args, args.tasks, tokenizer, few_shot=args.few_shot_training)
    config = BartWithAdapterConfig.from_pretrained(args.model)
    merge_args_into_config(args, config)

    bart_model = BartForConditionalGenerationWithAdapter(config)

    # if args.do_train and args.checkpoint is None:
    bart_model, debug_info = bart_model.from_pretrained(args.model, config=config, output_loading_info=True)
    add_special_tokens(bart_model, tokenizer, args)
    bart_model.reinit_generated_lm_head_weight()

    if args.cl_method in ['naive', 'ewc']:
        model = ConditionedHyperNetForCL(args, bart_model, config)
    elif args.cl_method == 'hnet':
        model = ConditionalHyperNetL2Reg(args, bart_model, config)
    else:
        raise NotImplementedError

    optimizer_grouped_parameters = get_trainable_params(args, model)
    optimizer, scheduler = get_optimizer(args, model, optimizer_grouped_parameters)
    model.set_label_vocab_space(main_task_sequence.get_label_space_map())

    opt_param_count = count_optimized_params(optimizer_grouped_parameters)
    param_count = count_params(model)
    logger.info('Optimized parameters: {}; total params: {}'.format(opt_param_count, param_count))

    if args.do_train:
        if args.checkpoint is not None:
            model.load_state_dict(convert_to_single_gpu(
                torch.load(args.checkpoint)))  # will be overriden by "load_best_checkpoint anyway"
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        if torch.cuda.is_available():
            model.to(torch.device("cuda"))
        mdl = model.module if args.n_gpu > 1 else model
        # iterate over task sequence here

        # continual learning (not mtl)
        if not args.mtl:
            task_iterator = main_task_sequence.get_dataloader_sequence_iterator()
            all_tasks = args.tasks
            for task_id, (task_name, (train_loader, dev_loader, test_loader)) in enumerate(task_iterator):
                if args.load_task != -1 and task_id < args.load_task:
                    logger.info('skipping task {}'.format(task_id))
                    continue
                if args.reset_optimizer_per_task:
                    optimizer, scheduler = get_optimizer(args, model, optimizer_grouped_parameters)

                # load the best model checkpoint after each task
                if task_id != 0:
                    load_best_checkpoint(args, model, optimizer, scheduler, postfix='_task_{}'.format(task_id - 1),
                                         use_tmp=args.ssd)

                if args.few_shot_validation:
                    # save the model checkpoint for now
                    logger.info('Started few shot validation')
                    few_shot_train(args, config, model, mdl, optimizer, scheduler, tokenizer, current_task_id=task_id)

                    logger.info('Finished few shot validation')
                mdl.update_current_task_id(task_id)
                train(
                    args, config, logger, model, tokenizer, train_loader, task_id, optimizer, scheduler,
                    main_task_sequence, fewshot=False, task_name=task_name,
                    eval_this_task_only=args.eval_every_k_tasks > 1,
                )
        else:
            # multi-task learning
            mtl_train_dataloader = main_task_sequence.get_mtl_dataloader(split='train', task_num=args.mtl_task_num)
            train(args, config, logger, model, tokenizer, mtl_train_dataloader, 0, optimizer, scheduler,
                  main_task_sequence, fewshot=False, task_name='mtl', mtl_max_task=args.mtl_task_num,
                  mtl=True)
        if args.ssd:
            save_best_checkpoint(args, model, optimizer, scheduler)

    if args.do_predict or args.do_few_shot_predict:
        if not args.fresh_checkpoint:
            checkpoint = os.path.join(args.output_dir, args.predict_checkpoint)
            model.load_state_dict(convert_to_single_gpu(torch.load(checkpoint)), strict=False)
            logger.info("Loading checkpoint from {}".format(checkpoint))
        else:
            logger.info('Not loading any checkpoint')
        if torch.cuda.is_available():
            model.to(torch.device("cuda"))
        model.eval()
        if args.do_predict:
            task_iterator = main_task_sequence.get_dataloader_sequence_iterator()

            for task_id, (task_name, (train_loader, dev_loader, test_loader)) in enumerate(task_iterator):
                if args.load_task != -1 and task_id < args.load_task:
                    logger.info('skipping task {}'.format(task_id))
                    continue
                ems = inference(args, config, model, tokenizer, main_task_sequence, test_loader, task_id=task_id,
                                task_name=task_name)
                logger.info("Task id {}, Task name {}, metric score: {}".format(task_id, task_name, ems))
        else:
            model.resize_stored_task_embs(n=len(args.tasks) * args.max_split_id + model.seen_full_tasks.item())
            optimizer_grouped_parameters = get_trainable_params(args, model)
            optimizer, scheduler = get_optimizer(args, model, optimizer_grouped_parameters)
            if args.gen_adapter_weight_only:
                score_dicts = inference_over_seen_tasks(args, config, model if args.n_gpu == 1 else model.module,
                                                        tokenizer, main_task_sequence, logger,
                                                        current_task_id=len(main_task_sequence),
                                                        split='dev' if not args.test else 'test',
                                                        eval_this_task_only=False,
                                                        fewshot=True)

            else:
                few_shot_train(args, config, model, model, optimizer, scheduler, tokenizer, current_task_id=-1)

            # few_shot_task_sequence = TaskSequence(args, args.tasks, tokenizer, few_shot=True)
            # few_shot_task_iterator = few_shot_task_sequence.get_dataloader_sequence_iterator()
            # optimizer_grouped_parameters = get_trainable_params(args, model)
            # optimizer, scheduler = get_optimizer(args, model, optimizer_grouped_parameters)

            # for task_id, (task_name, (train_loader, dev_loader, test_loader)) in few_shot_task_iterator:
            #    inference_few_shot(args,config, model, tokenizer, train_loader, dev_loader, few_shot_task_sequence,
            #                       task_id, adapt=args.do_few_shot_adapt, optimizer=optimizer, scheduler=scheduler,
            #                       task_name=task_name)


def get_regularizer(args, config, model, current_task_id, train_dataloader):
    if args.cl_method == 'ewc':
        from nets.regularizers import EWC
        regularizer = EWC(config, model, None, [train_dataloader], [current_task_id], args.output_dir)
    else:
        regularizer = None
    return regularizer


def train(args, config, logger, model, tokenizer, train_dataloader, task_id, optimizer, scheduler,
          main_task_sequence, eval_at_epoch_end=None, max_train_step=None, eval_period=None, postfix='',
          eval_this_task_only=False,
          fewshot=False, task_name=None, mtl_max_task=None, mtl=False):
    global TRIM_FLG
    model.train()
    eval_period, eval_at_epoch_end = args.eval_period if eval_period is None else eval_period, \
                                     args.eval_at_epoch_end if eval_at_epoch_end is None else eval_at_epoch_end
    max_train_step = args.max_train_step if max_train_step is None else max_train_step

    mdl = model.module if args.n_gpu > 1 else model

    global_step = 0
    train_losses = []
    best_accuracy, best_loss = -1.0, 1e10
    wait_step = 0
    stop_training = False

    logger.info("Starting training!")

    # reset task embedding stms. does not affect stored task embs
    mdl.reset_long_short_term_state()

    if args.hard_long_term:  # strictly fixed long term task emb
        logger.info('computing hard long term task emb')
        if not mtl:
            mdl.hard_update_task_emb(config, train_dataloader, task_id)
        else:
            for tmp_task_id in range(mtl_max_task):
                tmp_train_dataloader = main_task_sequence.get_data_loader(args.tasks[tmp_task_id], split='train')
                mdl.hard_update_task_emb(config, tmp_train_dataloader, tmp_task_id)

    # register regularizer, note: happens for each task
    regularizer = get_regularizer(args, config, mdl, task_id, train_dataloader)
    mdl.register_regularizer(regularizer)

    mdl.do_task_start(current_task_id=task_id)
    model.train()
    if args.freeze_layer_norm:
        logger.info('freeze layer norm')
        freeze_layer_norm(model)
    save_args0, save_args1 = None, None

    # for random sample batches for st rep
    if args.sample_batch:
        batch_sample_dataloader = DataLoader(train_dataloader.dataset, shuffle=True, batch_size=args.train_batch_size)
        batch_sample_iterator = get_batch_infinite(config, batch_sample_dataloader)

    for epoch in range(int(args.num_train_epochs)):
        for batch_idx, batch in tqdm(enumerate(train_dataloader), desc="Epoch {}".format(epoch)):
            if max_train_step > 0 and global_step >= max_train_step:
                break
            global_step += 1

            model_task_id, model_task_name = task_id, task_name
            if args.mtl:
                (model_task_id, model_task_name), batch = batch

            cq_inputs, cq_attention_mask, ans_inputs, ans_attention_mask = [torch.stack(x, 0).transpose(0, 1).cuda() for
                                                                            x in batch[0:4]]

            if args.try_max_len and not TRIM_FLG:
                logger.info('try max len: {}, {}'.format(cq_inputs.size(), ans_inputs.size()))
                TRIM_FLG = 1
            else:
                cq_inputs, cq_attention_mask = trim_batch(cq_inputs, config.pad_token_id, cq_attention_mask)
                ans_inputs, ans_attention_mask = trim_batch(ans_inputs, config.pad_token_id, ans_attention_mask)
            lb = batch[4] if len(batch) > 4 else None

            if args.sample_batch:
                if args.mtl:
                    batch_sample = train_dataloader.sample_batch_from_task(model_task_id)
                    cq_inputs_sample, cq_attention_mask_sample, ans_inputs_sample, ans_attention_mask_sample = [
                        torch.stack(x, 0).transpose(0, 1).cuda() for x in batch_sample[0:4]]
                else:
                    cq_inputs_sample, cq_attention_mask_sample, ans_inputs_sample, ans_attention_mask_sample = next(
                        batch_sample_iterator)
                mdl.update_task_emb(cq_inputs_sample, cq_attention_mask_sample, ans_inputs_sample,
                                    ans_attention_mask_sample, task_id=model_task_id,
                                    ignore_long=args.hard_long_term)
            else:
                mdl.update_task_emb(cq_inputs, cq_attention_mask, ans_inputs, ans_attention_mask, task_id=model_task_id,
                                    ignore_long=args.hard_long_term)
            loss, _, ret_dict = model(cq_inputs, cq_attention_mask, ans_inputs, ans_attention_mask, lb,
                                      is_training=True,
                                      task_id=model_task_id, task_name=model_task_name)
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if torch.isnan(loss).data:
                logger.info("Stop training because loss=%s" % (loss.data))
                stop_training = True
                break
            train_losses.append(loss.detach().cpu())

            if global_step % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()  # We have accumulated enough gradients
                scheduler.step()
                model.zero_grad()

            if global_step % eval_period == 0 or (batch_idx == len(train_dataloader) - 1 and eval_at_epoch_end):
                model.eval()
                score_dicts = inference_over_seen_tasks(args, config, model if args.n_gpu == 1 else model.module,
                                                        tokenizer, main_task_sequence, logger,
                                                        current_task_id=task_id if not args.mtl else (mtl_max_task - 1),
                                                        split='dev' if not args.test else 'test',
                                                        global_step=global_step, postfix=postfix,
                                                        eval_this_task_only=eval_this_task_only, fewshot=fewshot)

                # get a main metric score for this task
                seen_task_scores = [score_dict[get_main_metrics(args.tasks[task_id])] for task_id, score_dict in
                                    enumerate(score_dicts)]

                curr_score = seen_task_scores[-1] if not args.mtl else np.mean(seen_task_scores)
                curr_loss = np.mean(train_losses)
                logger.info("Step %d Train loss %.2f %s on epoch=%d, postfix %s" % (
                    global_step,
                    curr_loss,
                    curr_score,
                    epoch, postfix))
                avg_task_score = np.mean(seen_task_scores)
                logger.info('Step {}, task scores: {}, avg task score: {}, postfix {}'.format(
                    global_step,
                    ', '.join(['%.2f' % x for x in seen_task_scores]),
                    avg_task_score,
                    postfix
                ))

                train_losses = []
                if best_accuracy < curr_score:
                    if not fewshot:
                        save_args0 = lazy_save_best_checkpoint(args, model, optimizer, scheduler, use_tmp=args.ssd)
                        save_args1 = lazy_save_best_checkpoint(args, model, optimizer, scheduler,
                                                               postfix='_task_{}'.format(task_id), use_tmp=args.ssd,
                                                               current_task_id=task_id)
                    logger.info("Saving model with best %s -> %s on epoch=%d, global_step=%d" % \
                                (best_accuracy, curr_score, epoch, global_step))
                    best_accuracy = curr_score
                    wait_step = 0
                    stop_training = False

                else:
                    wait_step += 1
                    if wait_step >= args.wait_step:
                        stop_training = True
                        break
                model.train()
                if args.freeze_layer_norm:
                    freeze_layer_norm(model)
        if stop_training:
            break

    if not fewshot and eval_this_task_only and args.eval_every_k_tasks > 1 and task_id % args.eval_every_k_tasks == 0:
        score_dicts = inference_over_seen_tasks(args, config, model if args.n_gpu == 1 else model.module,
                                                tokenizer, main_task_sequence, logger,
                                                current_task_id=task_id if not args.mtl else (mtl_max_task - 1),
                                                split='dev', global_step=global_step, postfix=postfix,
                                                eval_this_task_only=False, fewshot=fewshot)
        seen_task_scores = [score_dict[get_main_metrics(args.tasks[task_id])] for task_id, score_dict in
                            enumerate(score_dicts)]

        curr_score = seen_task_scores[-1] if not args.mtl else np.mean(seen_task_scores)
        curr_loss = np.mean(train_losses)
        logger.info("Step %d Train loss %.2f %s on epoch=%d, postfix %s" % (
            global_step,
            curr_loss,
            curr_score,
            -1, postfix))
        avg_task_score = np.mean(seen_task_scores)
        logger.info('Step {}, task scores: {}, avg task score: {}, postfix {}'.format(
            global_step,
            ', '.join(['%.2f' % x for x in seen_task_scores]),
            avg_task_score,
            postfix
        ))
    if save_args0 is not None and not fewshot:
        exec_save_best_checkpoint(save_args0)
        exec_save_best_checkpoint(save_args1)

    mdl.do_task_end(current_task_id=task_id)


def few_shot_train(args, config, model, mdl, optimizer, scheduler, tokenizer, current_task_id=-1):
    few_shot_task_sequence = TaskSequence(args, args.tasks, tokenizer, few_shot=True)
    task_iterator = few_shot_task_sequence.get_dataloader_sequence_iterator()
    for fs_task_id, (fs_task_name, (few_shot_train_loader, few_shot_dev_loader, few_shot_test_loader)) in enumerate(
            task_iterator):
        if args.start_task >= 0 and fs_task_id < args.start_task:
            continue
        if args.stop_task > 0 and fs_task_id == args.stop_task:
            break
        if args.skip_tasks and fs_task_id in args.skip_tasks:
            continue
        logger.info('Few shot validation for {}({}) at task {}'.format(fs_task_id, fs_task_name, current_task_id))

        if args.load_adapter:
            file_name = '{}_adapter.pkl'.format(
                fs_task_name) if not args.load_adapter_postfix else '{}_adapter_{}.pkl'.format(fs_task_name,
                                                                                               args.load_adapter_postfix)

            model.bart_model.load_adapter_weights_from_path(os.path.join(args.load_adapter_path, file_name))

        all_states = save_state(args, mdl, optimizer, scheduler)
        few_shot_args = copy.copy(args)
        few_shot_args.train_batch_size = args.few_shot_train_batch_size
        few_shot_args.num_train_epochs = args.few_shot_num_train_epochs
        few_shot_args.wait_step = args.few_shot_wait_step
        few_shot_args.max_train_step = args.few_shot_max_train_step

        train(few_shot_args, config, logger, model, tokenizer, few_shot_train_loader, fs_task_id, optimizer, scheduler,
              few_shot_task_sequence, postfix='fewshot_at_{}'.format(current_task_id), eval_this_task_only=True,
              fewshot=True, task_name=fs_task_name, eval_period=args.few_shot_eval_period, eval_at_epoch_end=False)
        load_state(args, mdl, optimizer, scheduler, all_states)


def inference(args, config, model, tokenizer, main_task_sequence, dev_dataloader, task_id=None, task_name=None,
              global_step=-1,
              current_task_id=-1, task_emb=None, postfix='', limit_examples=False):
    with torch.no_grad():
        predictions, labels, questions = [], [], []
        bos_token_id = config.bos_token_id
        for i, batch in enumerate(dev_dataloader):
            if limit_examples and i == args.few_shot_test_batch_num:
                break
            pad_token_id = config.pad_token_id
            cq_inputs, cq_attention_mask, ans_inputs, ans_attention_mask = [torch.stack(x, 0).transpose(0, 1).cuda() for
                                                                            x
                                                                            in batch[0:4]]
            cq_inputs, cq_attention_mask = trim_batch(cq_inputs, config.pad_token_id, cq_attention_mask)
            ans_inputs, ans_attention_mask = trim_batch(ans_inputs, config.pad_token_id, ans_attention_mask)
            lb = batch[4] if len(batch) > 4 else None
            _, outputs, _ = model(cq_inputs, cq_attention_mask, ans_inputs, ans_attention_mask, lb, is_training=False,
                                  task_id=task_id, task_emb=task_emb, use_task_emb=task_emb is not None,
                                  task_name=task_name)
            outputs = outputs[:, 1:]
            for cq, gt, pred in zip(cq_inputs, ans_inputs, outputs):
                pred = tokenizer.convert_ids_to_tokens(pred)
                cq = tokenizer.convert_ids_to_tokens(cq)
                gt = tokenizer.convert_ids_to_tokens(gt)
                predictions.append(pred)
                labels.append(gt)
                questions.append(cq)

        raw_filename = 'results_task_{}_{}'.format(task_id, task_name)
        if task_id != -1 and task_id is not None:
            raw_filename += '_task_{}'.format(current_task_id)
        if global_step != -1:
            raw_filename += '_step_{}'.format(global_step)
        if postfix:
            raw_filename += '_{}'.format(postfix)
        if args.postfix:
            raw_filename += '_{}'.format(args.postfix)
        raw_filename += '.csv'
        save_predictions(config, labels, predictions, questions, tokenizer, raw_filename)
        em_score = exact_match_acc(predictions, labels)
        f1_score = f1_score_tokens_simple(predictions, labels, tokenizer)
    return {'em': em_score, 'f1': f1_score}


def inference_few_shot(args, config, model, tokenizer, train_dataloader, dev_dataloader, task_sequence, task_id,
                       task_name=None,
                       global_step=-1, current_task_id=-1, adapt=False, optimizer=None, scheduler=None, postfix=''):
    if not args.train_task_embs:
        task_emb = []
        with torch.no_grad():
            for i, batch in enumerate(train_dataloader):
                cq_inputs, cq_attention_mask, ans_inputs, ans_attention_mask = [torch.stack(x, 0).transpose(0, 1).cuda()
                                                                                for x
                                                                                in batch[0:4]]
                cq_inputs, cq_attention_mask = trim_batch(cq_inputs, config.pad_token_id, cq_attention_mask)
                ans_inputs, ans_attention_mask = trim_batch(ans_inputs, config.pad_token_id, ans_attention_mask)
                task_emb_instance = model.basic_task_encoder(cq_inputs, cq_attention_mask, ans_inputs,
                                                             ans_attention_mask)
                task_emb.extend(task_emb_instance.split(1))
            task_emb = torch.cat(task_emb).mean(0)
    else:
        task_emb = model.stored_task_embs[task_id]

    if args.gen_adapter_weight_only or (
            args.save_adapter_weight and (args.save_adapter_step <= 0 or global_step == args.save_adapter_step)):
        generated_weights = model.weight_generator(task_emb.unsqueeze(0))
        logger.info('storing adapter for {}: {}'.format(task_id, task_name))
        store_adapter_weights(args, generated_weights, task_name)
        if args.gen_adapter_weight_only:
            return None

    # if adapt:
    #    all_states = save_state(args, model, optimizer, scheduler)
    #    train(args, config, logger, model, tokenizer, train_dataloader, task_id, optimizer, scheduler,
    #          task_sequence, eval_at_epoch_end=False, eval_period=5 * len(train_dataloader), eval_this_task_only=True)
    #    load_state(args, model, optimizer, scheduler, all_states)

    with torch.no_grad():
        score_dict = inference(args, config, model, tokenizer, task_sequence, dev_dataloader, task_id, task_name,
                               global_step,
                               current_task_id, task_emb=task_emb, limit_examples=True, postfix=postfix)
    return score_dict


def inference_over_seen_tasks(args, config, model, tokenizer, task_sequence, logger, current_task_id, split='test',
                              global_step=-1, postfix='', eval_this_task_only=False, fewshot=False):
    task_iterator = task_sequence.get_dataloader_sequence_iterator()
    scores = []
    for task_id, (task_name, (train_loader, dev_loader, test_loader)) in enumerate(task_iterator):
        if task_id > current_task_id:
            break
        if eval_this_task_only and task_id != current_task_id:
            continue
        if split == 'test':
            loader = test_loader
        else:
            loader = dev_loader
        if not fewshot:
            score_dict = inference(args, config, model, tokenizer, task_sequence, loader, task_id=task_id,
                                   task_name=task_name,
                                   global_step=global_step, current_task_id=current_task_id, postfix=postfix)
        else:
            score_dict = inference_few_shot(args, config, model, tokenizer, train_loader, loader, task_sequence,
                                            task_id=task_id,
                                            task_name=task_name, global_step=global_step,
                                            current_task_id=current_task_id, postfix=postfix)
        logger.info("Task id {} over {} set, metric score: {}".format(task_id, split, score_dict))
        scores.append(score_dict)
    return scores


def main(args, logger):
    run(args, logger)


if __name__ == '__main__':
    args = get_args()

    ##### Start writing logs

    log_filename = "{}log.txt".format("" if args.do_train else "eval_")

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        handlers=[logging.FileHandler(os.path.join(args.output_dir, log_filename)),
                                  logging.StreamHandler()])
    logger = logging.getLogger(__name__)
    logger.info(args)
    logger.info(args.output_dir)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.n_gpu = torch.cuda.device_count()

    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # if not args.do_train and not args.do_predict:
    #    raise ValueError("At least one of `do_train` or `do_predict` must be True.")

    logger.info("Using {} gpus".format(args.n_gpu))
    if not args.debug:
        try:
            main(args, logger)
        except Exception as err:
            logger.error(repr(err))
            traceback.print_tb(err.__traceback__)
    else:
        main(args, logger)
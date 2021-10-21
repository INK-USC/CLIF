import argparse
import os
from data_utils.datasets import task_collection_to_tasks

def get_args(special=None):
    parser = argparse.ArgumentParser()

    ## Basic parameters
    #parser.add_argument("--train_file", default="data/structured_zeroshot-train-kilt.jsonl")
    #parser.add_argument("--predict_file", default="data/structured_zeroshot-dev-kilt.jsonl")
    #parser.add_argument("--dataset", default="zsre", required=True)
    parser.add_argument("--tasks", nargs='*')
    parser.add_argument("--task_collection", nargs='?')
    parser.add_argument('--split_id', type=int, default=-1)
    parser.add_argument('--merge_split', action='store_true')
    parser.add_argument("--no_cache", action='store_true')
    parser.add_argument("--model", default="facebook/bart-base", required=False)

    parser.add_argument("--output_dir", default=None, type=str, required=True)
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_predict", action='store_true')
    parser.add_argument("--do_few_shot_predict", action='store_true')
    parser.add_argument("--do_few_shot_adapt", action='store_true')
    parser.add_argument("--predict_checkpoint", type=str, default="best-model.pt")

    parser.add_argument('--add_space', action='store_true')

    parser.add_argument("--fp16", action='store_true')

    # some fine-grained options of training
    parser.add_argument("--train_limit", type=int, default=-1)
    parser.add_argument("--enforce_train_shuffle", action='store_true')
    parser.add_argument("--no_short_term", action='store_true')
    parser.add_argument("--hard_long_term", action='store_true', help='long term mem is absolutely fixed')
    parser.add_argument("--use_task_emb_mem", action='store_true', help='use task emb mem')
    parser.add_argument("--hard_long_term_limit", default=-1, type=int)
    parser.add_argument("--train_task_embs", action='store_true')
    parser.add_argument("--sample_batch",action='store_true')
    parser.add_argument('--zero_long_term', action='store_true')
    parser.add_argument("--limit_label_vocab_space", action='store_true')
    parser.add_argument("--few_shot_stat_label_space", action='store_true')
    parser.add_argument("--long_term_task_emb_num", type=int, nargs='?', default=-1)
    parser.add_argument('--freeze_layer_norm', action='store_true')

    # mtl arguments
    parser.add_argument("--mtl", action='store_true')
    parser.add_argument("--mtl_task_num", type=int)
    parser.add_argument("--sqrt", action='store_true')

    ## mainly for debugging and ablation studies
    parser.add_argument('--train_all', action='store_true')
    parser.add_argument('--train_flex', action='store_true')
    parser.add_argument('--base_model_lr', type=float, default=3e-5)
    parser.add_argument('--reset_optimizer_per_task', action='store_true')
    parser.add_argument('--eval_at_epoch_end', action='store_true')

    ## Model parameters
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--do_lowercase", action='store_true', default=False)
    parser.add_argument("--freeze_embeds", action='store_true', default=False)

    # few shot learning params
    parser.add_argument("--few_shot_training", action='store_true')
    parser.add_argument("--few_shot_validation", action='store_true')
    parser.add_argument("--few_shot_test_batch_num", default=100, type=int)
    parser.add_argument("--start_task", default=0, type=int)
    parser.add_argument("--stop_task", default=int(1e10), type=int)
    parser.add_argument("--skip_tasks", nargs='*', type=int)
    parser.add_argument("--example_limit", type=int, default=0, help='example limit for *few shot* learning')
    parser.add_argument("--k_shot", type=int, default=16, help='k shot num for datasets that are already splitted')
    parser.add_argument("--max_split_id", type=int, default=5)
    parser.add_argument("--fresh_checkpoint", action='store_true')
    parser.add_argument("--no_load", action='store_true')
    parser.add_argument("--test", action='store_true')



    # for params belwo, -1 means use the same as the regular training
    parser.add_argument("--few_shot_num_train_epochs", type=int, default=800)
    parser.add_argument("--few_shot_train_batch_size", type=int, default=64)
    parser.add_argument("--few_shot_wait_step", type=int, default=100)
    parser.add_argument("--few_shot_max_train_step", type=int, default=-1)


    # Preprocessing/decoding-related parameters
    parser.add_argument('--max_input_length', type=int, default=256)
    parser.add_argument('--max_output_length', type=int, default=8)
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument("--append_another_bos", action='store_true', default=False)

    # Training-related parameters
    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--predict_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=3e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--warmup_proportion", default=0.01, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=0.1, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="grad accum steps")
    parser.add_argument("--scale_by_accumulation", action='store_true')
    parser.add_argument("--try_max_len", action='store_true')
    parser.add_argument("--num_train_epochs", default=200, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_train_step", default=-1, type=int)
    parser.add_argument("--warmup_steps", default=500, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--total_steps", default=100000, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--wait_step', type=int, default=10000000000)
    parser.add_argument('--load_task', type=int, default=-1)
    parser.add_argument('--scale_loss', action='store_true')

    # Other parameters
    parser.add_argument("--verbose", action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")
    parser.add_argument('--eval_period', type=int, default=2000,
                        help="Evaluate & save model")
    parser.add_argument('--skip_intermediate_ckpt', action='store_true')
    parser.add_argument('--eval_every_k_tasks', type=int, default=1)
    parser.add_argument('--few_shot_eval_period', type=int, default=400)
    parser.add_argument('--prefix', type=str, default='',
                        help="Prefix for saving predictions")
    parser.add_argument('--postfix', default='')
    parser.add_argument('--debug', action='store_true',
                        help="Use a subset of data for debugging")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--crossfit_k_shot', default=-1, type=int)
    parser.add_argument('--gen_adapter_weight_only', action='store_true')

    parser.add_argument('--load_adapter', action='store_true')
    parser.add_argument('--load_adapter_path', default='')
    parser.add_argument('--load_adapter_postfix', default='')
    parser.add_argument('--save_adapter_weight', action='store_true')
    parser.add_argument('--save_adapter_step', default=-1, type=int)


    # adapter params
    parser.add_argument('--adapter_dim', default=64, type=int)
    parser.add_argument('--adapter_dim_final', default=32, type=int)
    parser.add_argument('--adapter_layer_norm', action='store_true')
    parser.add_argument('--generator_hdim', default=32,type=int)
    parser.add_argument('--generator_hdim_small', default=1,type=int)
    parser.add_argument('--adapter_final_layer_dim', default=64, type=int)
    parser.add_argument('--no_param_gen', action='store_true')
    parser.add_argument('--l2reg', default=0.0, type=float)
    parser.add_argument('--skip_adapter', action='store_true')
    parser.add_argument('--task_emb_dim', default=768, type=int)
    parser.add_argument('--task_encoder_model', default='bart')

    # cl params
    parser.add_argument('--task_num', type=int)
    parser.add_argument('--stm_size', type=int, default=10)

    parser.add_argument('--cl_method', type=str, default='naive')
    parser.add_argument('--h_l2reg', type=float, default=0.01)

    # ewc, mas params
    parser.add_argument('--reg_lambda', type=float, default=0.01)

    # special tokens
    parser.add_argument('--sep_token', default='<sep>')
    parser.add_argument('--ans_token', default='<ans>')

    if special == 'mbpa++':
        parser.add_argument('--batch_size', type=int, default=32,
                            help='Enter the batch size')
        parser.add_argument('--mode', default='train',
                            help='Enter the mode - train/test')
        parser.add_argument('--order', default=1, type=int,
                            help='Enter the dataset order - 1/2/3/4')
        #parser.add_argument('--epochs', default=2, type=int)
        parser.add_argument('--model_path', type=str,
                            help='Enter the path to the model weights')
        parser.add_argument('--memory_path', type=str,
                            help='Enter the path to the replay memory')
        parser.add_argument('--meta', action='store_true')
        parser.add_argument('--replay_freq', type=int, default=100)
        parser.add_argument('--sample_size', type=int, default=100)
        parser.add_argument('--retrieve_similar', action='store_true')
        parser.add_argument('--local_step', type=int, default=1)


        parser.add_argument('--random_retrieve', action='store_true')
    #if special == 'leopard':
    parser.add_argument('--inner_step', default=1, type=int)
    parser.add_argument('--inner_lr', default=1e-4, type=float)
    parser.add_argument('--te_batch_size', default=64, type=int)
    parser.add_argument("--te_k", default=1, type=int)
    parser.add_argument('--variant', default='leopard')
    parser.add_argument('--ssd', action='store_true')

    args = parser.parse_args()

    if (args.task_collection and args.tasks) or (not args.task_collection and not args.tasks) :
        raise ValueError('conflicting {}, {}'.format(args.task_collection, args.tasks))
    elif args.task_collection:
        args.tasks = task_collection_to_tasks(args.task_collection)

    # dirty fix
    # if args.do_few_shot_predict:
    #     print("Overriding some options in dirty fix")
    #     args.gradient_accumulation_steps = 1
    #     args.postfix = "naive"
    #     args.cl_method = "naive"

    if args.long_term_task_emb_num == -1:
        args.long_term_task_emb_num = len(args.tasks)

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        print("Output directory () already exists and is not empty.")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    return args


def merge_args_into_config(args, config):
    config.adapter_dim = args.adapter_dim
    config.adapt_layer_norm = False
    config.adapter_final_layer_dim = args.adapter_final_layer_dim
    config.task_num = len(args.tasks) if args.task_num is None else args.task_num
    config.long_term_task_emb_num = args.long_term_task_emb_num
    config.stm_size = args.stm_size
    config.max_output_length = args.max_output_length
    config.output_dir = args.output_dir
    config.generator_hdim = args.generator_hdim
    config.generator_hdim_small = args.generator_hdim_small
    config.adapter_dim_final = args.adapter_dim_final
    config.cl_method = args.cl_method
    config.h_l2reg = args.h_l2reg
    config.num_beams = args.num_beams
    config.no_param_gen = args.no_param_gen
    config.limit_label_vocab_space = args.limit_label_vocab_space
    config.skip_adapter = args.skip_adapter
    config.task_emb_dim = args.task_emb_dim
    config.train_task_embs = args.train_task_embs
    config.task_encoder_model = args.task_encoder_model
    config.train_flex = args.train_flex

def merge_args(src, tgt):
    for k, v in src.__dict__.items():
        setattr(tgt,k,v)

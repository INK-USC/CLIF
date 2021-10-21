# Learn Continually, Generalize Rapidly: Lifelong Knowledge Accumulation for Few-shot Learning

This repo is for Findings at EMNLP 2021 paper: Learn Continually, Generalize Rapidly: Lifelong Knowledge Accumulation for Few-shot Learning. Code clean-up is still in progress.

## Data
Please extract the downloaded data and place it under `PROJECT_DIR/datasets`. Our training data stream and few-shot datasets are curated from
https://github.com/iesl/leopard and https://github.com/INK-USC/CrossFit.


## Environment
Our code uses PyTorch 1.7.1. To allow fp16 training, you should also install [apex](https://github.com/NVIDIA/apex).


## Running Experiments

**Training on CLIF-26**

```
reg=0.01
lr=1e-4
seed=0
python run_model.py --tasks cola sst2 mrpc stsb qqp mnli qnli rte wnli \
--output_dir runs/glue_cfew_10k_choice_hnet_hardlong_sample_reg${reg}_s64_d256_limit/${lr}/${seed} \
--do_train --eval_period 100000 --eval_at_epoch_end  --wait_step 3 --num_train_epochs 100 --seed ${seed} \
--train_batch_size 64 --gradient_accumulation_steps 2 --learning_rate ${lr} --max_output_length 8 \
--generator_hdim 32 --example_limit 100 --train_limit 10000 --cl_method hnet --h_l2reg ${reg} \
--adapter_dim 256 --adapter_dim_final 64  --hard_long_term  --limit_label_vocab_space \
--sample_batch --scale_loss --stm_size 64
```

**Few-shot evaluation on CLIF-26**

```
python run_model.py --task_collection leopard --k_shot 16 --max_input_length 100  \
--output_dir /runs/glue_cfew_10k_choice_hnet_hardlong_sample_reg${reg}_s64_d256_limit/${lr}/${seed} \
--do_few_shot_predict --eval_period 100000 --eval_at_epoch_end  --wait_step 3 --num_train_epochs 100 \
--seed ${seed} --train_batch_size 64 --predict_batch_size 16 --few_shot_train_batch_size 16 \
--few_shot_wait_step 100000 --few_shot_num_train_epochs 800 --wait_step 3 --gradient_accumulation_steps 4 \
--scale_by_accumulation --learning_rate ${lr} --max_output_length 8  --generator_hdim 32 \
--example_limit 100 --train_limit 10000 --cl_method naive --h_l2reg ${reg} --adapter_dim 256 \
--adapter_dim_final 64 --hard_long_term --limit_label_vocab_space --no_short_term --long_term_task_emb_num 9 \
--postfix "naive_16shot"  --sample_batch --stm_size 64 --few_shot_eval_period 200
```

**Training and evaluation on CLIF-55**

```
reg=0.01
lr=1e-4
seed=0
python run_model.py  --task_collection crossfit_cls_train --crossfit_k_shot 16 --ssd --output_dir runs/crossfit_hnet_merge_space_${reg}/${lr}/${seed} --skip_intermediate_ckpt --add_space --merge_split --split_id ${seed} --seed ${seed} --do_train --eval_every_k_tasks 5 --eval_period 100 --skip_intermediate_ckpt --train_batch_size 64 --wait_step 3 --num_train_epochs 10000000  --learning_rate ${lr} --max_output_length 64 --example_limit 100 --train_limit 10000 --cl_method hnet --h_l2reg ${reg} --adapter_dim 256 --generator_hdim 32 --adapter_dim_final 64 --sample_batch --hard_long_term --stm_size 64
python run_model.py --task_collection crossfit_cls_train --crossfit_k_shot 16 --ssd --output_dir runs/crossfit_hnet_merge_space${reg}/${lr}/${seed} --skip_intermediate_ckpt --add_space --merge_split --split_id ${seed} --seed ${seed} --do_predict --eval_every_k_tasks 5 --eval_period 100 --skip_intermediate_ckpt --train_batch_size 64 --wait_step 3 --num_train_epochs 10000000  --learning_rate ${lr} --max_output_length 64 --example_limit 100 --train_limit 10000 --cl_method hnet --h_l2reg ${reg} --adapter_dim 256 --generator_hdim 32 --adapter_dim_final 64 --sample_batch --hard_long_term --stm_size 64
for split_id in 0 1 2 3 4
do
  python run_model.py --task_collection crossfit_cls_test --crossfit_k_shot 16 --ssd --postfix "split${split_id}"  --long_term_task_emb_num 45 --do_few_shot_predict --few_shot_eval_period 200 --few_shot_num_train_epochs 800 --few_shot_train_batch_size 64 --few_shot_wait_step 100 --mtl_task_num 45 --output_dir runs/crossfit_hnet_merge_space_${reg}/${lr}/${seed} --add_space  --limit_label_vocab_space --split_id ${split_id} --seed ${seed} --eval_period 100 --train_batch_size 64 --gradient_accumulation_steps 1 --wait_step 6 --num_train_epochs 10000  --learning_rate ${lr} --max_output_length 64 --example_limit 100 --train_limit 10000 --cl_method naive --adapter_dim 256 --generator_hdim 32 --adapter_dim_final 64 --sample_batch --hard_long_term
done
```


Here are mapping between command line arguments and implemented methods.
- BART-Single without adapter: `--cl_method naive --no_param_gen --skip_adapter --train_all`
- BART-Single-MTL: `--cl_method naive --no_param_gen --skip_mtl --mtl --train_all`
- BiHNET-Vanilla: `--cl_method naive --hard_long_term`
- BiHNET with trained task embeddings: `--cl_method hnet --no_short_term --train_task_embs --hard_long_term`
- BART-Adapter-Single: `--cl_method naive --no_param_gen --lr 3e-4`



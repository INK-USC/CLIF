import torch
from torch import nn
from .adapter_bart import BartForConditionalGenerationWithAdapter
from .weight_generator import ParameterGenerator
from .task_inference import SingleTaskEmbedding, LongTermState, ShortTermState
from utils.misc import trim_batch
from .utils import TaskEmbMemory


scaler = torch.cuda.amp.GradScaler()

class ConditionedHyperNetForCL(nn.Module):
    def __init__(self, args, bart_model, config):
        super().__init__()
        self.args = args
        self.config = config
        self.mtl = self.args.mtl
        self.bart_model = bart_model
        if self.config.no_param_gen:
            self.weight_generator = None
        else:
            self.weight_generator = ParameterGenerator(config, output_dims=self.bart_model.get_adapter_dims())

        self.basic_task_encoder = SingleTaskEmbedding(config)
        self.train_task_embs = args.train_task_embs

        self.long_term_task_emb_num = self.config.long_term_task_emb_num
        self.task_emb_dim = config.task_emb_dim
        if self.train_task_embs or self.args.train_flex:
            self.stored_task_embs = nn.Parameter(torch.zeros(self.config.long_term_task_emb_num, self.task_emb_dim))
        else:
            self.register_buffer('stored_task_embs', torch.zeros(self.config.long_term_task_emb_num, self.task_emb_dim))
        self.register_buffer('seen_full_tasks', torch.zeros(1).long())
        self.num_beams = config.num_beams
        self.cl_method = config.cl_method

        self.long_term_state = None
        self.short_term_state = None
        self.reset_long_short_term_state()
        self.regularizer = None
        self.no_short_term = args.no_short_term
        self.task_name_to_vocab_space = None

        self.l2reg = args.l2reg
        self.task_emb_debug = []


    def resize_stored_task_embs(self, n):
        stored_emb = self.stored_task_embs.data
        extended_stored_emb = torch.zeros(n, self.task_emb_dim).to(stored_emb.device)
        extended_stored_emb[:len(stored_emb)] = stored_emb
        if self.train_task_embs or self.args.train_flex:
            self.stored_task_embs = nn.Parameter(extended_stored_emb)
        else:
            self.register_buffer('stored_task_embs', extended_stored_emb)

    def forward(self, cq_input, cq_attention_masks, ans_input, ans_attention_masks, labels, is_training, task_id=None,
                 task_name=None, task_emb=None, use_task_emb=False):
        #if self.mtl:
        #    task_id = 0
        #    task_name = 'mtl'
        return self._forward(cq_input, cq_attention_masks, ans_input, ans_attention_masks, labels, is_training, task_id=task_id,
                 task_name=task_name, task_emb=task_emb, use_task_emb=use_task_emb)

    def _forward(self, cq_input, cq_attention_masks, ans_input, ans_attention_masks, labels, is_training, task_id=None,
                 task_name=None, task_emb=None, use_task_emb=False):
        ret_dict = {}
        if is_training:
            # need to update the long short term mem
            total_loss, long_term_ret, short_term_ret = self._forward_train(cq_input, cq_attention_masks, ans_input,
                                                                           ans_attention_masks, labels, is_training,
                                                                            task_id, task_name)
            return total_loss, (long_term_ret, short_term_ret), ret_dict
        elif task_id is not None and not use_task_emb:
            task_emb = self.stored_task_embs[task_id]
            ret = self._forward_with_task_emb(cq_input, cq_attention_masks, ans_input, ans_attention_masks, labels,
                                              task_emb, task_name, generate=True)
            return None, ret, ret_dict
        elif use_task_emb and task_emb is not None: # testing given task emb
            ret = self._forward_with_task_emb(cq_input, cq_attention_masks, ans_input, ans_attention_masks, labels,
                                              task_emb, task_name, generate=True)
            return None, ret, ret_dict
        else:
            raise ValueError(task_id, task_emb, use_task_emb)

    def compute_l2reg(self, generated_weights):
        generated_weights = torch.cat(generated_weights)
        loss = self.l2reg * (generated_weights ** 2).sum(-1)
        #loss.backward()
        return loss

    def _forward_train(self, cq_input, cq_attention_masks, ans_input, ans_attention_masks, labels, is_training, task_id, task_name):
        #with torch.no_grad():
        long_term_emb = self.stored_task_embs[task_id] #self.long_term_state.get_task_emb().to(cq_input.device)
        short_term_emb = self.short_term_state.get_task_emb().to(cq_input.device)
        long_term_ret = self._forward_with_task_emb(cq_input, cq_attention_masks, ans_input, ans_attention_masks, labels,
                                                   long_term_emb, task_name, reg_weights=True)


        if self.args.scale_loss:
            long_term_ret.loss = long_term_ret.loss / 2
        if self.args.scale_by_accumulation:
            long_term_ret.loss = long_term_ret.loss / self.args.gradient_accumulation_steps
        long_term_ret.loss.backward()
        short_term_ret = None
        if not self.no_short_term:
            short_term_ret = self._forward_with_task_emb(cq_input, cq_attention_masks, ans_input, ans_attention_masks, labels,
                                                        short_term_emb, task_name, reg_weights=True)
            if self.args.scale_loss:
                short_term_ret.loss = short_term_ret.loss / 2
            if self.args.scale_by_accumulation:
                short_term_ret.loss = short_term_ret.loss / self.args.gradient_accumulation_steps
            short_term_ret.loss.backward()
            total_loss = long_term_ret.loss + short_term_ret.loss
        else:
            total_loss = long_term_ret.loss

        return total_loss, long_term_ret, short_term_ret

    @torch.no_grad()
    def update_task_emb(self, cq_input, cq_attention_masks, ans_input, ans_attention_masks, task_id, ignore_long=False):
        instance_task_emb = self.basic_task_encoder(cq_input, cq_attention_masks, ans_input, ans_attention_masks)
        instance_task_emb = instance_task_emb.detach()
        self.short_term_state.update_emb_batch(instance_task_emb)

        if not ignore_long:
            self.long_term_state.update_emb_batch(instance_task_emb)
            long_term_emb = self.long_term_state.get_task_emb()
            # short_term_emb = self.short_term_state.get_task_emb()
            if task_id is not None:
                self.stored_task_embs[task_id] = long_term_emb

    @torch.no_grad()
    def hard_update_task_emb(self, config, train_dataloder, task_id):

        if self.train_task_embs: # train task embs, skip update
            self.long_term_state = LongTermState(self.config, trainable=True, weight=self.stored_task_embs, idx=task_id)
            long_term_emb = torch.zeros_like(self.stored_task_embs[task_id])
            long_term_emb = torch.nn.init.normal_(long_term_emb,mean=0, std=0.02)
            self.stored_task_embs[task_id].data = long_term_emb.data
        else:
            self.long_term_state = LongTermState(self.config)
            for batch_idx, batch in enumerate(train_dataloder):
                if self.args.hard_long_term_limit != -1 and batch_idx >= self.args.hard_long_term_limit:
                    break
                cq_inputs, cq_attention_mask, ans_inputs, ans_attention_mask = [torch.stack(x, 0).transpose(0, 1).cuda() for
                                                                                x in batch[0:4]]
                cq_inputs, cq_attention_mask = trim_batch(cq_inputs, config.pad_token_id, cq_attention_mask)
                ans_inputs, ans_attention_mask = trim_batch(ans_inputs, config.pad_token_id, ans_attention_mask)

                instance_task_emb = self.basic_task_encoder(cq_inputs, cq_attention_mask, ans_inputs, ans_attention_mask)
                instance_task_emb = instance_task_emb.detach()

                self.long_term_state.update_emb_batch(instance_task_emb)
            long_term_emb = self.long_term_state.get_task_emb()

            if self.args.zero_long_term:
                long_term_emb = torch.mean(self.stored_task_embs[:self.seen_full_tasks.item()], 0)
            if task_id is not None:
                self.stored_task_embs[task_id] = long_term_emb

    def update_current_task_id(self, task_id):
        self.seen_full_tasks[0] = task_id

    def _forward_with_task_emb(self, cq_input, cq_attention_masks, ans_input, ans_attention_mask, labels, task_emb,
                               task_name, generate=False, reg_weights=False):
        assert task_emb.dim() == 1
        generated_weights = None
        if not self.config.no_param_gen:
            generated_weights = self.weight_generator(task_emb.unsqueeze(0))
            self.bart_model.set_adapter_weights(generated_weights)
        if generate:
            gen_output = self.bart_model.generate(input_ids=cq_input, attention_mask=cq_attention_masks,
                                                  max_length=cq_input.size(1) + self.config.max_output_length,
                                                  num_beams=self.num_beams, num_beam_groups=1, task_name=task_name)
            ret = gen_output
        else:
            output = self.bart_model(cq_input, cq_attention_masks, labels=ans_input, task_name=task_name)
            ret = output

        if reg_weights and self.l2reg != 0:
            reg_loss = self.compute_l2reg(generated_weights)
            ret.loss += reg_loss

        return ret

    def get_task_emb(self, task_id=None, long_short_term=None):
        if task_id is None:
            if long_short_term == 'long':
                task_emb = self.long_term_state.get_task_emb()
            elif long_short_term == 'short':
                task_emb = self.short_term_state.get_task_emb()
            else:
                raise ValueError
        else:
            task_emb = self.stored_task_embs[task_id]
        return task_emb

    def do_task_start(self, current_task_id):
        if self.cl_method == 'ewc':
            self.regularizer.task_start_do()

    def do_task_end(self, current_task_id):
        if self.cl_method == 'ewc':
            self.regularizer.task_end_do()

    def register_regularizer(self, a):
        self.regularizer = a

    def reset_long_short_term_state(self):
        self.long_term_state = LongTermState(self.config)
        self.short_term_state = ShortTermState(self.config)

    def set_label_vocab_space(self, task_name_to_vocab_space):
        #self.bart_model.set_label_vocab_space(valid_token_ids)
        self.task_name_to_vocab_space = task_name_to_vocab_space
        self.bart_model.task_name_to_vocab_space = task_name_to_vocab_space

class ConditionalHyperNetL2Reg(ConditionedHyperNetForCL):
    def __init__(self, args, bart_model, config):
        super().__init__(args, bart_model, config)
        self.past_generated_weights = torch.zeros(self.long_term_task_emb_num, self.weight_generator.get_output_dim())
        self.counter = 0

        self.use_task_emb_mem = args.use_task_emb_mem
        # add a replay memory for past stm

        assert not args.no_param_gen
        self.old_weight_generator = ParameterGenerator(config, output_dims=self.bart_model.get_adapter_dims())

        if args.use_task_emb_mem:
            self.task_emb_mem = TaskEmbMemory(args)

    def resize_stored_task_embs(self, n):
        stored_emb = self.stored_task_embs
        extended_stored_emb = torch.zeros(n, self.task_emb_dim).to(stored_emb.device)
        extended_stored_emb[:len(stored_emb)] = stored_emb
        self.register_buffer('stored_task_embs', extended_stored_emb)

        extended_past_generated_weights = torch.zeros(n, self.weight_generator.get_output_dim())
        extended_past_generated_weights[:len(self.past_generated_weights)] = self.past_generated_weights
        self.past_generated_weights = extended_past_generated_weights

    def do_task_start(self, current_task_id):
        # for each visited task
        for prev_task_id in range(current_task_id):
            task_emb = self.stored_task_embs[prev_task_id]
            generated_weights = self.weight_generator(task_emb.unsqueeze(0), concat=True)
            self.past_generated_weights[prev_task_id] = generated_weights.cpu().detach()

    # additional loss for continual learning
    def hypernet_l2_loss(self, device):
        # choose a previous task and perform regularization
        current_task_id = self.seen_full_tasks.item()
        if current_task_id != 0:
            prev_task_id = self.counter % current_task_id
            self.counter += 1
            prev_task_emb = self.stored_task_embs[prev_task_id].to(device).detach()
            generated_weights = self.weight_generator(prev_task_emb.unsqueeze(0), concat=True)
            stored_generated_weights = self.past_generated_weights[prev_task_id].to(device).detach()
            # long term regularization loss
            loss = self.config.h_l2reg * ((generated_weights - stored_generated_weights) ** 2).sum(-1)

            # short term regularization loss
            if self.use_task_emb_mem:
                prev_short_term_emb, prev_task_id = self.task_emb_mem.sample()
                generated_weights = self.weight_generator(prev_short_term_emb.unsqueeze(0), concat=True)
                stored_generated_weights = self.past_generated_weights[prev_task_id].to(device).detach()
                short_term_loss = self.config.h_l2reg * ((generated_weights - stored_generated_weights) ** 2).sum(-1)
                loss += short_term_loss
            if self.args.scale_by_accumulation:
                loss = loss / self.args.gradient_accumulation_steps
            loss.backward()
            return loss
        else:
            return 0

    def _forward(self, cq_input, cq_attention_masks, ans_input, ans_attention_masks, labels, is_training, task_id=None,
                 task_name=None, task_emb=None, use_task_emb=False):
        ret_dict = {}

        if is_training:
            if self.use_task_emb_mem:
                if self.no_short_term:
                    long_term_emb = self.stored_task_embs[task_id]
                    self.task_emb_mem.store(long_term_emb, task_id)
                else:
                    short_term_emb = self.short_term_state.get_task_emb().to(cq_input.device)
                    self.task_emb_mem.store(short_term_emb, task_id)
            pred_loss, long_term_ret, short_term_ret = self._forward_train(cq_input, cq_attention_masks, ans_input,
                                                                           ans_attention_masks, labels, is_training, task_id, task_name)
            h_l2loss = self.hypernet_l2_loss(device=cq_input.device)
            total_loss = pred_loss + h_l2loss
            # there is NO backard outside the loop!
            # finish all backward operations in forward (to save gpu mem!)
            return total_loss, (long_term_ret, short_term_ret), ret_dict
        else:
            task_emb = self.stored_task_embs[task_id]
            ret = self._forward_with_task_emb(cq_input, cq_attention_masks, ans_input, ans_attention_masks, labels,
                                              task_emb, task_name, generate=True)
            return None, ret, ret_dict

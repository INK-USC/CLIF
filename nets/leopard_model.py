import higher
import torch
from torch import nn
from . weight_generator import ParameterGenerator
from .task_inference import SingleTaskEmbedding, LongTermState, ShortTermState
from .leopard_bart import BartForSequenceClassificationWithAdaptiveClassifierHead

class Leopard(nn.Module):
    def __init__(self, args, bart_model: BartForSequenceClassificationWithAdaptiveClassifierHead, config):
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

        self.inner_step = args.inner_step
        self.inner_lr = args.inner_lr
        self.te_k = args.te_k

        self.variant = args.variant
        # only at inference
        self.initialized_tasks = set()

    def forward(self, cq_input, cq_attention_masks, ans_input, ans_attention_masks, labels, dataloader, task_id, task_name,
              is_training, is_few_shot=False):
        if is_few_shot:
            return self._forward_few_shot(cq_input, cq_attention_masks, ans_input, ans_attention_masks, labels, dataloader, task_id, task_name,
                             is_training=is_training)
        else:
            return self._forward(cq_input, cq_attention_masks, ans_input, ans_attention_masks, labels, dataloader, task_id, task_name,
                             is_training=is_training)


    def partition_by_labels(self, labels, *partition_vectors):
        values = set()
        for b in range(labels.size(0)):
            values.add(labels[b].item())
        values = sorted(list(values))
        ret = []
        for value in values:
            indices = labels == value
            partitioned = []
            for vector in partition_vectors:
                partitioned.append(vector[indices])
            ret.append(partitioned)
        return values, ret

    def _forward_few_shot(self, cq_input, cq_attention_masks, ans_input, ans_attention_masks, labels, dataloader,
                          task_id, task_name, is_training):
        if self.variant == 'leopard':
            if task_name not in self.initialized_tasks:
                ans_input, ans_attention_masks = None, None # no use
                te_batch = dataloader.get_full_batch_from_task(task_id, config=self.config)
                te_labels = te_batch[-1]
                te_labels, te_batches_by_label = self.partition_by_labels(te_labels, *te_batch)
                all_generated_weights = []
                for label, te_batch_by_label in zip(te_labels, te_batches_by_label):
                    # training time, use labeled instances to predict params
                    te_cq_input, te_cq_attention_masks, te_ans_input ,te_ans_attention_masks = \
                        te_batch_by_label[0:4]
                    te_task_emb = self.basic_task_encoder(te_cq_input, te_cq_attention_masks, te_ans_input, te_ans_attention_masks)
                    te_task_emb = te_task_emb.mean(0).unsqueeze(0)
                    generated_weights = self.weight_generator(te_task_emb)
                    all_generated_weights.append(generated_weights)
                self.bart_model.set_classifier_head_weight(task_id, task_name, te_labels, all_generated_weights, keep_optimizable=True)
                self.initialized_tasks.add(task_name)

        ret = self.bart_model(cq_input, cq_attention_masks, ans_input, ans_attention_masks, labels=labels, task_name=task_name)
        if is_training:
            ret.loss.backward()
        return ret.loss, ret, None


    def _forward(self, cq_input, cq_attention_masks, ans_input, ans_attention_masks, labels, dataloader,
                       task_id, task_name, is_training):
        ans_input, ans_attention_masks = None, None # no use
        if self.variant == 'leopard':
            te_batch = dataloader.sample_batch_from_task(task_id, cat=True, config=self.config, k=self.args.te_k)
            te_labels = te_batch[-1]
            te_labels, te_batches_by_label = self.partition_by_labels(te_labels, *te_batch)
            all_generated_weights = []
            for label, te_batch_by_label in zip(te_labels, te_batches_by_label):
                # training time, use labeled instances to predict params
                te_cq_input, te_cq_attention_masks, te_ans_input ,te_ans_attention_masks = \
                    te_batch_by_label[0:4]
                te_task_emb = self.basic_task_encoder(te_cq_input, te_cq_attention_masks, te_ans_input, te_ans_attention_masks)
                te_task_emb = te_task_emb.mean(0).unsqueeze(0)
                generated_weights = self.weight_generator(te_task_emb)
                all_generated_weights.append(generated_weights)
            self.bart_model.set_classifier_head_weight(task_id, task_name, te_labels, all_generated_weights)

        inner_optimizer = torch.optim.SGD(self.bart_model.task_specific_parameters(), lr=self.inner_lr)
        with higher.innerloop_ctx(self.bart_model, inner_optimizer, track_higher_grads=False, copy_initial_weights=False) as (fnet, diffopt):
            train_losses = []
            for g in range(self.inner_step):
                supp_batch = dataloader.sample_batch_from_task(task_id, cat=True, config=self.config)
                cq_input_supp, cq_attention_masks_supp, _, _, labels_supp=supp_batch[0:5]
                ret = fnet(cq_input_supp, cq_attention_masks_supp, labels=labels_supp, task_name=task_name)
                #ret.loss.backward(retain_graph=True)
                grad_w, grad_b = torch.autograd.grad(ret.loss, fnet.task2head[task_name].out_proj.weight, retain_graph=True), \
                                 torch.autograd.grad(ret.loss, fnet.task2head[task_name].out_proj.bias, retain_graph=True)
                if type(grad_w) is tuple:
                    grad_w, grad_b = grad_w[0], grad_b[0]
                fnet.optimize_task_specific_param(task_name, self.inner_lr, grad_w, grad_b, variant=self.variant)
                #self.bart_model.zero_grad()
                diffopt.step(ret.loss)
                train_losses.append(ret.loss.item())

            if is_training:
                all_trainable_params = [_ for _ in fnet.parameters()] + [_ for _ in self.weight_generator.parameters()]
                all_trainable_params_origin = [_ for _ in self.bart_model.parameters()] + [_ for _ in self.weight_generator.parameters()]
                #params = [_ for _ in self.bart_model.task_specific_parameters()]
                ret = fnet(cq_input, cq_attention_masks, ans_input, ans_attention_masks, labels=labels, task_name=task_name)
                grads = torch.autograd.grad(ret.loss, all_trainable_params, allow_unused=True)
                for p, g in zip(all_trainable_params_origin, grads):
                    if g is not None:
                        if p.grad is None:
                            p.grad = g.detach()
                        else:
                            p.grad += g.detach()

                #ret.loss.backward()
            else:
                with torch.no_grad():
                    self.train(False)
                    ret = fnet(cq_input, cq_attention_masks, ans_input, ans_attention_masks, labels=labels,
                               task_name=task_name)
                    self.train(True)
        return ret.loss, ret, None

    def set_task_label_space(self, task_name_to_label_space):
        #self.bart_model.set_label_vocab_space(valid_token_ids)
        self.task_name_to_label_space = task_name_to_label_space
        self.bart_model.set_task_name_to_label_space(task_name_to_label_space)



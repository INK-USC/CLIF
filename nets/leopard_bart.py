import torch
from torch import nn
from transformers.models.bart.modeling_bart import BartEncoder, BartDecoder, BartModel, \
    BartForConditionalGeneration, BartEncoderLayer, BartDecoderLayer, BartPretrainedModel, \
    Seq2SeqSequenceClassifierOutput
from transformers.models.bart.modeling_bart import shift_tokens_right, CrossEntropyLoss, Seq2SeqLMOutput, ACT2FN
from transformers.models.bart.configuration_bart import BartConfig
from transformers.configuration_utils import PretrainedConfig
from torch.nn import functional as F

class BartClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        pooler_dropout: float,
    ):
        super().__init__()
        #self.dense = nn.Linear(input_dim, inner_dim)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(input_dim, num_classes)


    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states

class DotDict():
    def __init__(self):
        pass

class BartClassificationHeadWoParams(nn.Module):
    """Head for sentence-level classification tasks."""
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        pooler_dropout: float,
    ):
        super().__init__()
        #self.dense = nn.Linear(input_dim, inner_dim)
        self.dropout = nn.Dropout(p=pooler_dropout)
        out_proj = nn.Linear(input_dim, num_classes)
        self.out_proj = DotDict()
        self.out_proj.weight = out_proj.weight.data.detach()
        self.out_proj.bias = out_proj.bias.data.detach()


    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.dropout(hidden_states)
        hidden_states = F.linear(hidden_states, self.out_proj.weight, self.out_proj.bias) #self.out_proj(hidden_states)
        return hidden_states



class BartForSequenceClassificationWithAdaptiveClassifierHead(BartPretrainedModel):
    def __init__(self, config: BartConfig, **kwargs):
        if 'variant' in kwargs:
            self.variant = kwargs.pop('variant')
        else:
            self.variant = 'fomaml'
        super().__init__(config, **kwargs)
        self.model = BartModel(config)
        self.config = config
        #self.classification_head = BartClassificationHead(
        #    config.d_model,
        #    config.num_labels,
        #    config.classifier_dropout,
        #)
        #self.model._init_weights(self.classification_head.dense)
        #self.model._init_weights(self.classification_head.out_proj)
        self.task2labels = None
        self.task2head = nn.ModuleDict()

    def set_task_name_to_label_space(self, task2labels):
        """
        needs to call this function and confirm optimizer is re-initialized
        """
        self.task2labels = task2labels
        for task_name, labels in task2labels.items():
            if self.variant == 'fomaml':
                self.task2head[task_name] = BartClassificationHead(self.config.d_model, len(labels),
                                                               self.config.classifier_dropout)
            else:
                self.task2head[task_name] = BartClassificationHeadWoParams(self.config.d_model, len(labels),
                                                                   self.config.classifier_dropout)
    def task_specific_parameters(self):
        # params = []
        # for module in self.model.encoder.layers[3:]:
        #     params.extend([_ for _ in module.parameters()])
        # for module in self.model.decoder.layers[3:]:
        #     params.extend([_ for _ in module.parameters()])
        # return params
        return self.parameters()

    def optimize_task_specific_param(self, task_name, sgd_lr, grad_w, grad_b, variant='leopard'):
        classifier_head = self.task2head[task_name]
        if variant == 'leopard':
            new_weight = classifier_head.out_proj.weight - sgd_lr * grad_w
            new_bias = classifier_head.out_proj.bias - sgd_lr * grad_b
            classifier_head.out_proj.weight = new_weight
            classifier_head.out_proj.bias = new_bias
        else:
            classifier_head.out_proj.weight.data -= sgd_lr * grad_w
            classifier_head.out_proj.bias.data -= sgd_lr * grad_b

    def set_classifier_head_weight(self, task_id, task_name, labels, all_generated_weights, keep_optimizable=False):
        classifier_head = self.task2head[task_name]
        new_w, new_b = [], []
        for label in self.task2labels[task_name]:
        #for label, generated_weights in zip(labels, all_generated_weights):
            if label not in labels:
                w = classifier_head.out_proj.weight[label].view(-1).data.detach()
                b = classifier_head.out_proj.bias[label].view(-1).data.detach()
            else:
                idx = labels.index(label)
                generated_weights = all_generated_weights[idx]
                generated_weights = generated_weights[0]
                w = generated_weights[:classifier_head.out_proj.weight.size(-1)]
                b = generated_weights[classifier_head.out_proj.weight.size(-1):]
            new_w.append(w)
            new_b.append(b)
        new_w = torch.stack(new_w) # [label, feat]
        new_b = torch.stack(new_b).view(-1)#[label]
        if not keep_optimizable:
            del classifier_head.out_proj.weight
            del classifier_head.out_proj.bias

            classifier_head.out_proj.weight = new_w
            classifier_head.out_proj.bias = new_b
        else:
            classifier_head.out_proj.weight.data.copy_(new_w.detach())
            classifier_head.out_proj.bias.data.copy_(new_b.detach())

    def get_adapter_dims(self):
        return [self.config.d_model + 1]

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        encoder_outputs=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        task_name=None,
        task_id=None
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            use_cache = False

        if input_ids is None and inputs_embeds is not None:
            raise NotImplementedError(
                f"Passing input embeddings is currently not supported for {self.__class__.__name__}"
            )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]  # last hidden state

        eos_mask = input_ids.eq(self.config.eos_token_id)

        # if len(torch.unique(eos_mask.sum(1))) > 1:
        #     print(input_ids)
        #     raise ValueError("All examples must have the same number of <eos> tokens. Size of input_ids: {}".format(input_ids.size()))
        #
        #
        # sentence_representation = hidden_states[eos_mask, :].view(hidden_states.size(0), -1, hidden_states.size(-1))[
        #     :, -1, :
        # ]

        sentence_representation = hidden_states[:, 0, :].view(hidden_states.size(0), -1, hidden_states.size(-1))[
            :, -1, :
        ]

        classification_head = self.task2head[task_name]
        logits = classification_head(sentence_representation)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, len(self.task2labels[task_name])), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

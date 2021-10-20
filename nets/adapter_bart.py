from transformers.models.bart.modeling_bart import BartEncoder, BartDecoder, BartModel, \
    BartForConditionalGeneration, BartEncoderLayer, BartDecoderLayer
from transformers.models.bart.modeling_bart import shift_tokens_right, CrossEntropyLoss, Seq2SeqLMOutput, ACT2FN
from transformers.models.bart.configuration_bart import BartConfig
from transformers.configuration_utils import PretrainedConfig
import pickle

from .utils import label_smoothed_nll_loss, total_param_dim
from typing import Optional, Tuple
from collections import OrderedDict


import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import numpy as np


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight, gain=0.0000001)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


class BartWithAdapterConfig(BartConfig):
    def __init__(
            self,
            activation_dropout=0.0,
            activation_function="gelu",
            vocab_size=50265,
            d_model=1024,
            encoder_ffn_dim=4096,
            encoder_layers=12,
            encoder_attention_heads=16,
            decoder_ffn_dim=4096,
            decoder_layers=12,
            decoder_attention_heads=16,
            encoder_layerdrop=0.0,
            decoder_layerdrop=0.0,
            attention_dropout=0.0,
            dropout=0.1,
            max_position_embeddings=1024,
            init_std=0.02,
            classifier_dropout=0.0,
            num_labels=3,
            is_encoder_decoder=True,
            pad_token_id=1,
            bos_token_id=0,
            eos_token_id=2,
            normalize_before=False,
            add_final_layer_norm=False,
            scale_embedding=False,
            normalize_embedding=True,
            static_position_embeddings=False,
            add_bias_logits=False,
            adapter_dim=64,
            adapt_layer_norm=False,
            unfreeze_hyper_encoder=False,
            **common_kwargs
    ):
        if "hidden_size" in common_kwargs:
            raise ValueError("hidden size is called d_model")

        super().__init__(
            num_labels=num_labels,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            **common_kwargs,
        )
        self.vocab_size = vocab_size
        self.d_model = d_model  # encoder_embed_dim and decoder_embed_dim
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_layers = self.num_hidden_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_layerdrop = encoder_layerdrop
        self.decoder_layerdrop = decoder_layerdrop
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.init_std = init_std  # Normal(0, this parameter)
        self.activation_function = activation_function

        # Params introduced for Mbart
        self.scale_embedding = scale_embedding  # scale factor will be sqrt(d_model) if True
        self.normalize_embedding = normalize_embedding  # True for mbart, False otherwise
        self.normalize_before = normalize_before  # combo of fairseq's encoder_ and decoder_normalize_before
        self.add_final_layer_norm = add_final_layer_norm

        # Params introduced for Marian
        self.add_bias_logits = add_bias_logits
        self.static_position_embeddings = static_position_embeddings

        # 3 Types of Dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.dropout = dropout

        # Classifier stuff
        self.classif_dropout = classifier_dropout

        # Adapter
        self.adapter_dim = adapter_dim
        self.generator_hdim = 64
        self.generator_hdim_small = 1
        self.adapt_layer_norm = adapt_layer_norm
        self.unfreeze_hyper_encoder = unfreeze_hyper_encoder


class ModelWithAdapter(nn.Module):
    def init_adapter(self, input_dim, mid_dim, output_dim, config):
        self.config = config
        self.adapter_name_to_weight = OrderedDict()
        self.skip_adapter = self.config.skip_adapter
        self.no_param_gen = self.config.no_param_gen
        self.adapter_down_weight, self.adapter_up_weight, self.adapter_down_bias, self.adapter_up_bias = \
            None, None, None, None
        #self.adapter_up, self.adapter_down = None, None  # modules
        self.adapter_id = 0
        if self.no_param_gen:
            self.all_adapters = nn.ModuleList()

        if self.no_param_gen:
            task_num = self.config.task_num
            for i in range(task_num):
                adapter = nn.ModuleList(
                    [nn.Linear(input_dim, mid_dim),
                    nn.Linear(mid_dim, output_dim)]
                ).cuda()
                self.all_adapters.append(adapter)
        #else:
        self.adapter_down_weight = torch.zeros(input_dim, mid_dim).cuda()
        self.adapter_down_bias = torch.zeros(mid_dim).cuda()
        self.adapter_up_weight = torch.zeros(mid_dim, output_dim).cuda()
        self.adapter_up_bias = torch.zeros(output_dim).cuda()
        self.dirty = False

    def set_adapter_down_weight(self, tensor):
        self.adapter_down_weight = tensor
        if self.no_param_gen:
            self.all_adapters[self.adapter_id][0].weight.copy_(tensor)
            # self.adapter_down_weight = None
            # self.adapter_down_func = nn.Linear(tensor.size(0), tensor.size(1)).cuda()

    def set_adapter_down_bias(self, tensor):
        self.adapter_down_bias = tensor
        if self.no_param_gen:
            self.all_adapters[self.adapter_id][0].bias.copy_(tensor)
            #self.adapter_down_bias = None

    def set_adapter_up_weight(self, tensor):
        self.adapter_up_weight = tensor
        if self.no_param_gen:
            self.all_adapters[self.adapter_id][1].weight.copy_(tensor)
            #self.adapter_up_weight = None
            #self.adapter_up_func = nn.Linear(tensor.size(0), tensor.size(1)).cuda()

    def set_adapter_up_bias(self, tensor):
        self.adapter_up_bias = tensor
        if self.no_param_gen:
            self.all_adapters[self.adapter_id][1].bias.copy_(tensor)

    def set_adapter_id(self, adapter_id):
        self.adapter_id = adapter_id

    def register_adapter_name_to_weight(self, names, weights):
        for name, weight in zip(names, weights):
            self.adapter_name_to_weight[name] = weight

    def get_my_module_weight_dims(self):
        return [
            self.adapter_down_weight.size(),
            self.adapter_down_bias.size(),
            self.adapter_up_weight.size(),
            self.adapter_up_bias.size()
        ]

    def get_my_weight_dim(self):
        s = total_param_dim(self.get_my_module_weight_dims())
        return s

    def adapter_down(self, x):
        if self.no_param_gen:
            return self.all_adapters[self.adapter_id][0](x)
        return F.linear(x, self.adapter_down_weight.t(), self.adapter_down_bias)

    def adapter_up(self, x):
        if self.no_param_gen:
            return self.all_adapters[self.adapter_id][1](x)
        return F.linear(x, self.adapter_up_weight.t(), self.adapter_up_bias)

    def set_adapter_weights(self, weight_vector):
        # default behavior
        return self.set_my_adapter_weights(weight_vector)
            
    def set_my_adapter_weights(self, weight_vector):
        sizes = self.get_my_module_weight_dims()
        prev_start = 0
        for size, (name, value) in zip(sizes, self.adapter_name_to_weight.items()):
            flat_size = np.product(size)
            weight_data = weight_vector[prev_start:prev_start + flat_size].cuda()
            # value.copy_(weight_data.view(*value.size()))
            if not self.no_param_gen:
                setattr(self, name, weight_data.view(*value.size()))
            else:
                #getattr(self, name).data.copy_(weight_data.view(*value.size()).detach())
                weight = weight_data.view(*value.size())
                if name == 'adapter_down_weight':
                    self.all_adapters[self.adapter_id][0].weight.data.copy_(weight_data.view(*value.size()).t())
                elif name == 'adapter_down_bias':
                    self.all_adapters[self.adapter_id][0].bias.data.copy_(weight_data.view(*value.size()))
                elif name == 'adapter_up_weight':
                    self.all_adapters[self.adapter_id][1].weight.data.copy_(weight_data.view(*value.size()).t())
                elif name == 'adapter_up_bias':
                    self.all_adapters[self.adapter_id][1].bias.data.copy_(weight_data.view(*value.size()))
                else:
                    raise ValueError(name)
            prev_start += flat_size


class EncoderLayerWithAdapter(BartEncoderLayer, ModelWithAdapter):
    def __init__(self, config: BartConfig):
        super().__init__(config)
        #BartEncoderLayer.__init__(self, config)
        #ModelWithAdapter.__init__(self, config)
        self.adapter_dim = config.adapter_dim
        #self.set_adapter_down_weight(torch.zeros(self.embed_dim, self.adapter_dim).cuda())
        #self.set_adapter_down_bias(torch.zeros(self.adapter_dim).cuda())

        #self.set_adapter_up_weight(torch.zeros(self.adapter_dim, self.embed_dim).cuda())
        #self.set_adapter_up_bias(torch.zeros(self.embed_dim).cuda())
        self.init_adapter(self.embed_dim, self.adapter_dim, self.embed_dim, config)
        
        self.register_adapter_name_to_weight(['adapter_down_weight', 'adapter_down_bias','adapter_up_weight',
                                              'adapter_up_bias'],[self.adapter_down_weight, self.adapter_down_bias,
                                             self.adapter_up_weight, self.adapter_up_bias])

    def forward(self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_head_mask: torch.Tensor,
        output_attentions: bool = False
        ):

        residual = hidden_states
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)

        if not self.skip_adapter:
            residual_adapter = hidden_states
            hidden_states = self.adapter_down(hidden_states)
            hidden_states = self.activation_fn(hidden_states)
            hidden_states = self.adapter_up(hidden_states)
            hidden_states = residual_adapter + hidden_states

        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = F.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        if torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class DecoderLayerWithAdapter(BartDecoderLayer, ModelWithAdapter):
    def __init__(self, config: BartConfig):
        super().__init__(config)
        self.adapter_dim = config.adapter_dim

        #self.set_adapter_down_weight(torch.zeros(self.embed_dim, self.adapter_dim).cuda())
        #self.set_adapter_down_bias(torch.zeros(self.adapter_dim).cuda())

        #self.set_adapter_up_weight(torch.zeros(self.adapter_dim, self.embed_dim).cuda())
        #self.set_adapter_up_bias(torch.zeros(self.embed_dim).cuda()) 
        
        self.init_adapter(self.embed_dim, self.adapter_dim, self.embed_dim, config)

        self.register_adapter_name_to_weight(['adapter_down_weight', 'adapter_down_bias','adapter_up_weight',
                                              'adapter_up_bias'],[self.adapter_down_weight, self.adapter_down_bias,
                                             self.adapter_up_weight, self.adapter_up_bias])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        encoder_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
    ):
        residual = hidden_states

        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # add present self-attn cache to positions 1,2 of present_key_value tuple
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)

        # adapter plugs in here
        if not self.skip_adapter:
            residual_adapter = hidden_states
            hidden_states = self.adapter_down(hidden_states)
            hidden_states = self.activation_fn(hidden_states)
            hidden_states = self.adapter_up(hidden_states)
            hidden_states = residual_adapter + hidden_states

        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Cross-Attention Block
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states

            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=encoder_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
            )
            hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = F.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class BartEncodeWithAdapter(BartEncoder):
    def __init__(self, config: BartConfig, embed_tokens):
        super(BartEncodeWithAdapter, self).__init__(config, embed_tokens)
        self.layers = nn.ModuleList(
            [EncoderLayerWithAdapter(config) for _ in range(config.encoder_layers)]
        )


class BartDecoderWithAdapter(BartDecoder):
    def __init__(self, config: BartConfig, embed_tokens: nn.Embedding):
        super(BartDecoderWithAdapter, self).__init__(config, embed_tokens)
        self.layers = nn.ModuleList(
            [DecoderLayerWithAdapter(config) for _ in range(config.decoder_layers)]
        )


class BartModelWithAdapter(BartModel):
    def __init__(self, config: BartConfig):
        super(BartModelWithAdapter, self).__init__(config)
        self.encoder = BartEncodeWithAdapter(config, self.shared)
        self.decoder = BartDecoderWithAdapter(config, self.shared)


class BartForConditionalGenerationWithAdapter(BartForConditionalGeneration, ModelWithAdapter):
    def __init__(self, config: BartConfig):
        super().__init__(config)
        base_model = BartModelWithAdapter(config)
        self.model = base_model
        self.config = config
        self.embed_dim = config.d_model
        self.adapter_dim = config.adapter_dim_final
        self.activation_fn = ACT2FN[config.activation_function]
        self.init_adapter(self.embed_dim, self.adapter_dim, self.model.shared.num_embeddings, config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.reinit_generated_lm_head_weight()
        self.adapter_down_weight, self.adapter_down_bias, self.adapter_up_weight, self.adapter_up_bias = \
            None, None, None, None
        self.task_name_to_vocab_space = None

    def reinit_generated_lm_head_weight(self):
        # self.adapter_down_weight = torch.zeros(self.embed_dim, self.adapter_dim).cuda()
        # self.adapter_down_bias = torch.zeros(self.adapter_dim).cuda()
        #
        # self.adapter_up_weight = torch.zeros(self.adapter_dim, self.model.shared.num_embeddings).cuda()
        # self.adapter_up_bias = torch.zeros(self.model.shared.num_embeddings).cuda()

        #self.set_adapter_down_weight(torch.zeros(self.embed_dim, self.adapter_dim).cuda())
        #self.set_adapter_down_bias(torch.zeros(self.adapter_dim).cuda())

        #self.set_adapter_up_weight(torch.zeros(self.embed_dim, self.model.shared.num_embeddings).cuda())
        #self.set_adapter_up_bias(torch.zeros(self.model.shared.num_embeddings).cuda())

        self.init_adapter(self.embed_dim, self.adapter_dim, self.model.shared.num_embeddings, self.config)

        self.register_adapter_name_to_weight(['adapter_down_weight', 'adapter_down_bias','adapter_up_weight',
                                              'adapter_up_bias'],[self.adapter_down_weight, self.adapter_down_bias,
                                             self.adapter_up_weight, self.adapter_up_bias])
    
    def get_children_adapter_modules(self):
        return [_ for _ in self.model.encoder.layers] + [_ for _ in self.model.decoder.layers]
    
    def get_adapter_dims(self):
        adapter_modules = self.get_children_adapter_modules()
        required_weight_dims = [module.get_my_weight_dim() for module in adapter_modules] + \
                            [self.get_my_weight_dim()]
        return required_weight_dims

    def set_adapter_weights(self, all_adapter_weights):
        all_children_adapter_modules = self.get_children_adapter_modules()
        for module, weight in zip(all_children_adapter_modules, all_adapter_weights[:len(all_children_adapter_modules)]):
            module.set_adapter_weights(weight)
        wb_weights = torch.cat(all_adapter_weights[len(all_children_adapter_modules):], 0)
        self.set_my_adapter_weights(wb_weights)
        
    def set_adapter_id(self, adapter_id):
        adapter_modules = self.get_children_adapter_modules()
        for module in adapter_modules:
            module.set_adapter_id(adapter_id)
        self.adapter_id = adapter_id

    def load_adapter_weights_from_path(self, path):
        with open(path,'rb') as f:
            weights = pickle.load(f)
        self.set_adapter_weights(weights)

    def set_label_vocab_space(self, valid_token_ids):
        self.valid_token_ids = valid_token_ids

    def mask_logits_by_label_vocab_space(self, task_name, lm_logits):
        mask = torch.ones(lm_logits.size(-1)).bool().to(lm_logits.device)
        mask[list(self.task_name_to_vocab_space[task_name])] = 0
        mask = mask.expand(*lm_logits.size()[:-1], -1)
        masked_lm_logits = lm_logits.masked_fill(mask, -np.inf)
        return masked_lm_logits

    # class BartWithAdapter(BartForConditionalGenerationWithAdapter):
    def forward(self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        task_name=None):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        lm_logits_residual = self.lm_head(outputs[0]) + self.final_logits_bias

        #lm_logits = F.linear(outputs[0], self.generated_lm_head_weight, self.generated_lm_head_bias)
        if not self.skip_adapter :
            hidden_states = self.adapter_down(outputs[0])
            hidden_states = self.activation_fn(hidden_states)
            hidden_states = self.adapter_up(hidden_states)
            lm_logits = lm_logits_residual + hidden_states
        else:
            lm_logits = lm_logits_residual

        if self.config.limit_label_vocab_space and task_name is not None:
            lm_logits = self.mask_logits_by_label_vocab_space(task_name, lm_logits)
        elif self.config.limit_label_vocab_space:
            raise ValueError('specified limit label vocab space but task name missing')

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def encoders(self):
        return self.model.encoder.layers

    def decoders(self):
        return self.model.decoder.layers

    def backup_layer_norm_parameters(self):
        for encoder in self.encoders():
            encoder.self_attn_layer_norm_bc = copy.deepcopy(encoder.self_attn_layer_norm)
        for decoder in self.decoders():
            decoder.self_attn_layer_norm_bc = copy.deepcopy(decoder.self_attn_layer_norm)

    def restore_layer_norm_parameters(self):
        for encoder in self.encoders():
            encoder.self_attn_layer_norm = copy.deepcopy(encoder.self_attn_layer_norm_bc)
        for decoder in self.decoders():
            decoder.self_attn_layer_norm = copy.deepcopy(decoder.self_attn_layer_norm_bc)

    def _prepare_encoder_decoder_kwargs_for_generation(
        self, input_ids: torch.LongTensor, model_kwargs
    ):
        # retrieve encoder hidden states
        encoder = self.get_encoder()
        encoder_kwargs = {
            argument: value for argument, value in model_kwargs.items() if not argument.startswith("decoder_")
                                                                           and not argument.startswith('task_name')
        }
        model_kwargs["encoder_outputs"] = encoder(input_ids, return_dict=True, **encoder_kwargs)
        return model_kwargs

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        task_name=None,
        **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
            "task_name": task_name
        }

def _make_linear_from_emb(emb):
    vocab_size, emb_size = emb.weight.shape
    lin_layer = nn.Linear(vocab_size, emb_size, bias=False)
    lin_layer.weight.data = emb.weight.data
    return lin_layer

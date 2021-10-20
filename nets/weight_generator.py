import torch
from torch import nn
from transformers.activations import ACT2FN
from typing import List

def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight, gain=0.001)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m

class SingleMLPWeightGenerator(nn.Module):
    def __init__(self, config, input_dim, output_dim, idx=None, prev_f_input_dim=None):
        super().__init__()
        self.config = config
        self.input_dim = input_dim  # config.d_model
        self.hidden_dim = config.generator_hdim
        self.output_dim = output_dim
        if config.adapt_layer_norm:
            self.output_dim += 2 * config.d_model

        if self.output_dim > 10000 * config.d_model:
            self.hidden_dim = config.generator_hdim_small

        self.f_input_dim = None

        lineary_factory = Linear
        self.linear1 = lineary_factory(self.input_dim, self.hidden_dim)
        self.activation_fn = ACT2FN[config.activation_function]
        self.linear2 = lineary_factory(self.hidden_dim, self.output_dim)


    def get_output_dim(self):
        return self.output_dim

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation_fn(x)
        x = self.linear2(x)
        return x.view(-1)


class ParameterGenerator(nn.Module):
    def __init__(self, config, output_dims: List[int]):
        super().__init__()

        self.config = config
        self.input_dim = config.hidden_size
        self.output_dims = output_dims

        l = []

        prev_f_input_dim = None
        for idx, output_dim in enumerate(output_dims):
            m = SingleMLPWeightGenerator(config, self.input_dim, output_dim, idx, prev_f_input_dim)
            prev_f_input_dim = m.f_input_dim
            l.append(m)
        self.decoders = nn.ModuleList(l)

    def decode(self, sr):
        return [one_decoder(sr) for one_decoder in self.decoders]

    def forward(self, task_embedding, concat=False):
        params = self.decode(task_embedding)
        if concat:
            params = torch.cat(params)
        return params

    def get_output_dim(self):
        return sum(self.output_dims)


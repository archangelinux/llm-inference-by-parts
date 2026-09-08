import torch
import torch.nn as nn
import torch.nn.functional as F

#per channel quantization
def quantize_weight(w): #->(q, scale)
    row_maxes = w.abs().max(dim=1, keepdim=True).values #(w.shape[0], 1)
    scale = (row_maxes/127).clamp(min=torch.finfo(w.dtype).tiny) # .tiny is the smallest positive normal value, vs .min is negative
    q = (w/scale).round().to(torch.int8)
    return (q, scale)


class QuantLinear(nn.Module):
    def __init__(self, linear): #built from an existing Linear
        super().__init__()
        q, s = quantize_weight(linear.weight)
        self.bias = linear.bias # nn.Parameter autoregistered
        #need to register tensors with pytorch
        self.register_buffer("q", q)
        self.register_buffer("scale", s)
        
    def forward(self, x):
        w = self.q.to(x.dtype)* self.scale
        return F.linear(x, w, self.bias) #x @ w.T + bias


def quantize_model(model):
    for block in model.transformer.h: #for each hiddne block
        block.attn.c_attn = QuantLinear(block.attn.c_attn)
        block.attn.c_proj = QuantLinear(block.attn.c_proj)
        block.mlp.c_fc = QuantLinear(block.mlp.c_fc)
        block.mlp.c_proj = QuantLinear(block.mlp.c_proj)
    return model

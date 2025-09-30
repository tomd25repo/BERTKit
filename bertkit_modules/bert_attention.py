import torch
from typing import Optional
from torch import nn

from bertkit_modules.bert_self_attention import BertSelfAttention
from bertkit_modules.bert_self_output import BertSelfOutput

class BertAttention(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.self = BertSelfAttention(
            config,
            layer_idx=layer_idx,
        )
        self.output = BertSelfOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
    ) -> tuple[torch.Tensor]:
        self_outputs = self.self(
            hidden_states,
            attention_mask=attention_mask,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs

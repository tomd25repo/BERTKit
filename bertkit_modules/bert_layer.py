import torch
from torch import nn
from typing import Optional

from bertkit_modules.bert_attention import BertAttention
from bertkit_modules.bert_intermediate import BertIntermediate
from bertkit_modules.bert_output import BertOutput
from bertkit_modules.bert_utils import apply_chunking_to_forward

class BertLayer(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1  # index where seq len is encoded (2nd dimension)
        
        #
        # building blocks of a BERT layer: Attention, FFN, output layer
        #
        self.attention = BertAttention(config, layer_idx=layer_idx)  # defined in bertkit_modules
        self.intermediate = BertIntermediate(config)   # layer after attention
        self.output = BertOutput(config)   # layer after attention

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> tuple[torch.Tensor]:
        #
        # compute self attention from input representation.
        # self attention module returns context as first argument (attention weights * values)
        # and attention probs as second argument 
        #
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask=attention_mask,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions to output tuple if desired
        
        # the apply_chunking_to_forward function splits long inputs into chunks and feeds them
        # sequentially. This function originally stems from the pytorch_utils module that is imported.
        # We carve out this function and provide it directly.
        
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

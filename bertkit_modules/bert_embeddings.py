import torch
from typing import Optional
from torch import nn

class BertEmbeddings(nn.Module):
    

    def __init__(self, config):
        #
        # define the needed building blocks for the BERT entrance block
        #
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        # self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        # data in the registered buffers are *no* parameters which means they are not updated by
        # the optimizer but are numbers being part of the model itself. Position IDs logically are
        # important model data but can not be updated, same for token_type_ids!
        # pytorch documentation:
        # This is typically used to register a buffer that should not be considered a model parameter. 
        # For example, BatchNorm’s running_mean is not a parameter, but is part of the module’s state. 
        # Buffers, by default, are persistent and will be saved alongside parameters. This behavior can 
        # be changed by setting persistent to False. The only difference between a persistent buffer and 
        # a non-persistent buffer is that the latter will not be a part of this module’s state_dict.
        
        # this buffer here becomes accessible as self.position_ids and is prefilled with the
        # numbers 0,1,...,max_seq_length.
        # Hint:  torch.arange(config.max_position_embeddings) yields a 1D torch tensor tensor([  0,   1, ... ,511])
        # torch.arange(config.max_position_embeddings).expand((1, -1)) yields a 2D tensor
        # tensor([[  0,   1,   2, ... 510, 511]]) . Shape therefore is torch.Size([1, 512])
        
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )   # shape of the tensor will be torch.Size([1, 512]), a 2D tensor!
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )   # same shape as position_ids ([batch_size, seq_length]), a 2D tensor!
        
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """
        The forward function. Currently we want the tokenizer be used and do not support
        feeding BERT with embeddings right from the start. This may be an option later.
        """
        #
        # currently we assume the shape of the input IDs as follows:
        # (batch_size, seq_length, embedding_dimension?)
        #
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            raise ValueError("BERT input error. You must provide input ids as generated from the tokenizer")

        seq_length = input_shape[1]
        
        position_ids = self.position_ids[:, 0 : seq_length]
        print("Position ids:", position_ids)

            
        # Setting the token_type_ids to the registered buffer in constructor 
        # where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users 
        # when tracing the model without passing token_type_ids
        
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                print("hasattr")
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
                print(token_type_ids)
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)
        
        inputs_embeds = self.word_embeddings(input_ids)
        
        # add token embeddings for the two possible types of tokens
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        print("shape toty embeddings:", token_type_embeddings.shape)
        print("shape embeddings:", embeddings.shape)
        print("shape word embeddings:", inputs_embeds.shape)
        
        # add positional embeddings
        #
        # attention: positional embeddings have shape [1, seq_len, emb_dim] and therefore
        # a broadcasting has to be done to the shape [bs, seq_len, emb_dim]. Since the latter
        # two dimensions are compatible, broadcasting is possible without any further means
        # and done by pytorch automatically.
        #
        position_embeddings = self.position_embeddings(position_ids)
        print("shape position embeddings:", position_embeddings.shape)
        embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        print("embeddings shape:", embeddings.size())
        return embeddings 

import torch
from bertkit_tokenizer import tokenization # the amended Google tokenizer
from typing import Dict, Tuple, Union

# for a proper mypy check do 
# pip install types-six, otherwise you may get a mypy error fron the tokenization module
# where six is used.


class BertKitTokenizer():
    
    def __init__(self, vocab_file:Union[str, None]=None):
        
        if vocab_file is None:
            raise ValueError("Please provide path to vocab.txt!")
        
        self.cls_token = "[CLS]"
        self.sep_token = "[SEP]"
        self.pad_token = "[PAD]"
        self.unk_token = "[UNK]"
        self.mask_token = "[MASK]"
        self.max_seq_length = 512
        self.tokenizer = tokenization.FullTokenizer(vocab_file)
        self.vocab = self.tokenizer.vocab   # we need access to the vocab dict from FullT.
    
    
    def convert_tokenlist(self, tokenlist:list[list[str]]) -> list[list[int]]:
        id_list = []
        for sentence_tokens in tokenlist:
            sentence_list = [self.vocab.get(token, self.vocab["[UNK]"]) for token in sentence_tokens]
            id_list.append(sentence_list)
        return id_list
    
    
    def pad_sequences(self, sequences:list[list[int]], pad_value=0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Pad sequences to the length of the longest sequence.
        Generate corresponding token_type_id lists (all zero).
        Generate attention lists where entries are 1 for non pad tokens.
        Returns:
            List of lists where all sublists have the same length
        """
        # sanity checks:
        assert sequences is not None
        assert sequences[0] is not None

        # Find the maximum length
        max_length = max(len(seq) for seq in sequences)

        # Pad each sequence to max_length
        padded_sequences = []
        token_type_ids = []
        attention_mask = []
        
        for seq in sequences:
            padded_seq = seq + [pad_value] * (max_length - len(seq))
            padded_sequences.append(padded_seq)
            
            token_type_ids.append([0] * len(padded_seq))
            
            attention_seq = [1]*len(seq) + [0]*(max_length - len(seq))
            attention_mask.append(attention_seq)
            
        p = torch.tensor(padded_sequences, dtype=torch.long)
        t = torch.tensor(token_type_ids, dtype=torch.long)
        a = torch.tensor(attention_mask, dtype=torch.long)

        return p, t, a
                             
                             
    def tokenize(self, texts:Union[str, list[str]]) -> Dict[str, torch.Tensor]:
        
        tokenlist = []
        if isinstance(texts, str):
            tokens = self.tokenizer.tokenize(texts)
            assert isinstance(tokens, list)
            if len(tokens) > self.max_seq_length - 2:   # consider CLS + SEP token!
                tokens = tokens[:self.max_seq_length - 2]
            tokens = [self.cls_token] + tokens + [self.sep_token]
            tokenlist.append(tokens)
        elif isinstance(texts, list):
            for text in texts:
                tokens = self.tokenizer.tokenize(text)
                assert isinstance(tokens, list)
                if len(tokens) > self.max_seq_length - 2:   # consider CLS + SEP token!
                    tokens = tokens[:self.max_seq_length - 2]
                tokens = [self.cls_token] + tokens + [self.sep_token]
                tokenlist.append(tokens)
        else:
            raise ValueError("Type of input text must be str or list[str]!")

        #
        # Now we have a list of lists of tokens. 
        # They are now converted to ids.
        # In case of an empty string the tokenlist at least contains
        # the CLS and SEP token.
        #

        input_ids = self.convert_tokenlist(tokenlist)
        assert  input_ids is not None
        
        padded_sequences, token_type_ids, attention_mask = self.pad_sequences(
            input_ids,
            pad_value=self.vocab[self.pad_token]
        )
        
        result = {"input_ids": padded_sequences, 
                  "token_type_ids": token_type_ids, 
                  "attention_mask": attention_mask}
        return result
    
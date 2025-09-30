# BERTKit

A standalone version of BERT completely independent of the HF ecosystem and reproducing sentence embeddings from the [CLS] token or pooling layer as the original version. 

The repo contains an independent Tokenizer, the BERT model in a reduced code version derived from the original HF code and a weight loader for instantiating the kit model with the original pretrained weigths. 

This effort was done mainly for
- having a very lightweight system analogous to BERT allowing for quick experiments and embedding generation 
- educational purposes such as detailed architecture analysis
- extensible code base completely independent of a large ecosystem

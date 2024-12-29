import numpy as np

from tokenizer import BPE_RLE_Tokenizer as Tokenizer

with open('quantized_coeffs.txt','r') as f:
    text = f.read()

data = text
# print(data)

tokenizer = Tokenizer()
tokenizer.load_merges("neo_tokenizer/merges.json")
tokenizer.load_vocab("neo_tokenizer/vocab.json")

encoded = tokenizer.encode(data.strip().split(),as_ids=True)
# print(tokenizer.decode(encoded))

#532

import torch

buf = torch.tensor(np.array(encoded)[:10640+1])
x = (buf[:-1]).view(20, 532) # inputs
y = (buf[1:]).view(20, 532) # targets

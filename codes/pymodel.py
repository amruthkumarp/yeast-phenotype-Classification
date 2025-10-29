# Load model and tokenizer
from transformers import AutoTokenizer, AutoModel
import torch
import os,sys
import numpy as np
import pandas as pd
import gc

proxy = "http://245hsbb003@ibab.ac.in:Amruth517222@proxy.ibab.ac.in:3128"

# Set both upper and lowercase versions
os.environ["HTTP_PROXY"]  = proxy
os.environ["HTTPS_PROXY"] = proxy
os.environ["http_proxy"]  = proxy
os.environ["https_proxy"] = proxy

#read csv data
df=pd.read_csv('output_sequences.csv')

sequences = [df['sequence'].iloc[0]]
print(f"Batch size: {len(sequences)}")
print(f"Batch size: {len(sequences)}")
# Load tokenizer from local folder

tokenizer = AutoTokenizer.from_pretrained("models/segment_nt_tokenizer", trust_remote_code=True)

# Load model from local folder
model = AutoModel.from_pretrained("models/segment_nt_model", trust_remote_code=True)

print("Model loaded susscessfully")

# Choose the length to which the input sequences are padded. By default, the 
# model max length is chosen, but feel free to decrease it as the time taken to 
# obtain the embeddings increases significantly with it.
# The number of DNA tokens (excluding the CLS token prepended) needs to be dividible by
# 2 to the power of the number of downsampling block, i.e 4.
# The number of DNA tokens (excluding the CLS token prepended) needs to be dividible by
# the square of the number of downsampling block in SegmentNT UNet head, i.e 4.
# In the paper and in the jax colab, the length is set at 8333 tokens, which corresponds
# to 49992 nucleotides. On Google Colab, the inference with this length fits on the
# JAX model but does not fit in the Torch model. Therefore, we select here a slightly
# smaller length.
max_num_dna_tokens = 8200+1

assert (max_num_dna_tokens - 1) % 4 == 0, (
    "The number of DNA tokens (excluding the CLS token prepended) needs to be divisible by "
    "2 to the power of the number of downsampling blocks, i.e. 4."
)

# If max_num_tokens is larger than what was used to train Segment-NT, the rescaling
# factor needs to be adapted.
if max_num_dna_tokens + 1 > 5001:
    inference_rescaling_factor = (max_num_dna_tokens + 1) / 2048

    # Apply the new rescaling factor to all Rotary Embeddings layer.
    num_layers = len(model.esm.encoder.layer)

    for layer in range(num_layers):
      model.esm.encoder.layer[layer].attention.self.rotary_embeddings.rescaling_factor = inference_rescaling_factor
else:
    inference_rescaling_factor = None

#  dna sequence and tokenize it

tokens = tokenizer.batch_encode_plus(sequences, return_tensors="pt", padding="max_length", truncation=True,max_length = max_num_dna_tokens)["input_ids"]
# Print the shape
print(tokens.shape)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

gc.collect()
torch.cuda.empty_cache()

# Infer
tokens = tokens.to(device)
# If using attention_mask


attention_mask = tokens != tokenizer.pad_token_id
attention_mask = attention_mask.to(device)

with torch.no_grad():
	outs = model(
    			tokens,
    			attention_mask=attention_mask,
    			output_hidden_states=True
			)

# Obtain the logits over the genomic features
logits = outs.logits.detach()
# Transform them in probabilities
# Transform them on probabilities
probabilities = np.asarray(torch.nn.functional.softmax(logits, dim=-1).cpu())[...,-1]

print(f"Probabilities shape: {probabilities.shape}")

np.save("output_np0.npy", probabilities)

# Get probabilities associated with intron
idx_intron = model.config.features.index("intron")
probabilities_intron = probabilities[:,:,idx_intron]
print(f"Intron probabilities shape: {probabilities_intron.shape}")


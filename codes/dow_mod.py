from transformers import AutoTokenizer, AutoModel
import torch

# Download and save locally
tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/segment_nt", trust_remote_code=True)
tokenizer.save_pretrained("/home/ibab/models/segment_nt_tokenizer")

model = AutoModel.from_pretrained("InstaDeepAI/segment_nt", trust_remote_code=True)
model.save_pretrained("/home/ibab/models/segment_nt_model")
# Load model and tokenizer





# Choose the length to which the input sequences are padded. By default, the 
# model max length is chosen, but feel free to decrease it as the time taken to 
# obtain the embeddings increases significantly with it.
# The number of DNA tokens (excluding the CLS token prepended) needs to be dividible by
# 2 to the power of the number of downsampling block, i.e 4.
max_length = 12 + 1

assert (max_length - 1) % 4 == 0, (
    "The number of DNA tokens (excluding the CLS token prepended) needs to be dividible by"
     "2 to the power of the number of downsampling block, i.e 4.")

# Create a dummy dna sequence and tokenize it
sequences = ["ATTCCGATTCCGATTCCG", "ATTTCTCTCTCTCTCTGAGATCGATCGATCGAT"]
tokens = tokenizer.batch_encode_plus(sequences, return_tensors="pt", padding="max_length", max_length = max_length)["input_ids"]

# Infer
attention_mask = tokens != tokenizer.pad_token_id
outs = model(
    tokens,
    attention_mask=attention_mask,
    output_hidden_states=True
)

# Obtain the logits over the genomic features
logits = outs.logits.detach()
# Transform them in probabilities
probabilities = torch.nn.functional.softmax(logits, dim=-1)
print(f"Probabilities shape: {probabilities.shape}")

# Get probabilities associated with intron
idx_intron = model.config.features.index("intron")
probabilities_intron = probabilities[:,:,idx_intron]
print(f"Intron probabilities shape: {probabilities_intron.shape}")




import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
load=np.load("stacked_array.npy")

print(load.shape)

#download the pretrained model
tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/segment_nt", trust_remote_code=True)
model = AutoModel.from_pretrained("InstaDeepAI/segment_nt", trust_remote_code=True)

#using config function to get the index of features(what order)
print(f"Features inferred: {model.config.features}")

#get index of particular  index
protein_coding_gene = model.config.features.index("protein_coding_gene")

probabilities_protein_coding_gene = load[...,protein_coding_gene]
print(f"Intron probabilities shape: {probabilities_protein_coding_gene.shape}")

df=pd.DataFrame(probabilities_protein_coding_gene)

df

df.to_csv("protein_coding_gene_probabilities.csv")

lncRNA = model.config.features.index('lncRNA')

probabilities_lncRNA = load[...,lncRNA]
print(f"Intron probabilities shape: {probabilities_lncRNA.shape}")

df1=pd.DataFrame(probabilities_lncRNA)

df1.to_csv("lncRNA_probabilities.csv")

exon = model.config.features.index('exon')

probabilities_exon = load[...,exon]
print(f"Intron probabilities shape: {probabilities_exon.shape}")

df2=pd.DataFrame(probabilities_exon)

df2.to_csv("exon_probabilities.csv")

Intron = model.config.features.index('intron')

probabilities_intron = load[...,Intron]
print(f"Intron probabilities shape: {probabilities_intron.shape}")

df3=pd.DataFrame(probabilities_intron)

df3.to_csv("intron_probabilities.csv")

splice_donor = model.config.features.index('splice_donor')

probabilities_splice_donor = load[...,splice_donor]
print(f"splice_donor probabilities shape: {probabilities_splice_donor.shape}")

df4=pd.DataFrame(probabilities_splice_donor)

df4.to_csv("splice_donor_probabilities.csv")

splice_acceptor=  model.config.features.index('splice_donor')

probabilities_splice_acceptor = load[...,splice_acceptor]
print(f"splice_donor probabilities shape: {probabilities_splice_acceptor.shape}")

df5=pd.DataFrame(probabilities_splice_acceptor)

df5.to_csv("splice_acceptor_probabilities.csv")

UTR5= model.config.features.index('5UTR')

probabilities_5UTR = load[...,UTR5]
print(f"splice_donor probabilities shape: {probabilities_5UTR.shape}")

df6=pd.DataFrame(probabilities_5UTR)

df6.to_csv("5UTR_probabilities.csv")

UTR3= model.config.features.index('3UTR')

probabilities_3UTR = load[...,UTR3]
print(f"splice_donor probabilities shape: {probabilities_3UTR.shape}")

df7=pd.DataFrame(probabilities_3UTR)

df7.to_csv("3UTR_probabilities.csv")

CTCF_bound= model.config.features.index('CTCF-bound')

probabilities_CTCF = load[...,CTCF_bound]
print(f"splice_donor probabilities shape: {probabilities_CTCF.shape}")

df8=pd.DataFrame(probabilities_CTCF)

df8.to_csv("CTCF_bound_probabilities.csv")

polyA_signal= model.config.features.index('polyA_signal')

probabilities_polyA_signal = load[...,polyA_signal]
print(f"polyA_signal probabilities shape: {probabilities_polyA_signal.shape}")

df9=pd.DataFrame(probabilities_polyA_signal)

df9.to_csv("polyA_signal_probabilities.csv")

enhancer_Tissue_specific= model.config.features.index('enhancer_Tissue_specific')

probabilities_penhancer_Tissue_specific = load[...,enhancer_Tissue_specific]
print(f"polyA_signal probabilities shape: {probabilities_penhancer_Tissue_specific.shape}")

df10=pd.DataFrame(probabilities_penhancer_Tissue_specific)

df10.to_csv("enhancer_Tissue_specific_probabilities.csv")

enhancer_Tissue_invariant= model.config.features.index('enhancer_Tissue_invariant')

probabilities_enhancer_Tissue_invariant = load[...,enhancer_Tissue_invariant]
print(f"polyA_signal probabilities shape: {probabilities_enhancer_Tissue_invariant.shape}")

df11=pd.DataFrame(probabilities_enhancer_Tissue_invariant)

df11.to_csv("enhancer_Tissue_invariant_probabilities.csv")

promoter_Tissue_specific= model.config.features.index('promoter_Tissue_specific')

promoter_Tissue_specific = load[...,promoter_Tissue_specific]
print(f"polyA_signal probabilities shape: {promoter_Tissue_specific.shape}")

df12=pd.DataFrame(promoter_Tissue_specific)

df12.to_csv("promoter_Tissue_specific_probabilities.csv")

promoter_Tissue_invariant= model.config.features.index('promoter_Tissue_invariant')

promoter_Tissue_invariant = load[...,promoter_Tissue_invariant]
print(f"polyA_signal probabilities shape: {promoter_Tissue_invariant.shape}")

df132=pd.DataFrame(promoter_Tissue_invariant)

df.to_csv("promoter_Tissue_invariant_probabilities.csv")
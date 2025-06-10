import torch
import esm
import pandas as pd
import pickle

# Load ESM-2 model
# model, alphabet = esm.pretrained.load_model_and_alphabet_local(model_path)
model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval() 
df = pd.read_excel(r'data.xlsx',)

protain = df['sequence'].tolist()
smiles = df['ID'].tolist()

datas = []
for i in range(len(protain)):
    datas.append((smiles[i], protain[i])) 

sequence_representations = []

for data in datas:
    try :
        batch_labels, batch_strs, batch_tokens = batch_converter([data])
    except Exception as e:
        print(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[30], return_contacts=True)

    token_representations = results["representations"][30]

# Generate per-sequence representations via averaging
    for i, tokens_len in enumerate(batch_lens):
        sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))
esm_embedding = {}
for i in range(len(sequence_representations)):
    esm_embedding[datas[i][0]] = sequence_representations[i]
print(sequence_representations[0].shape)
print(len(esm_embedding))

file_path = 'Peptide.pkl'
with open(file_path, 'wb') as file:
    pickle.dump(esm_embedding, file)

with open(file_path, 'rb') as file:
    loaded_dict = pickle.load(file)
    

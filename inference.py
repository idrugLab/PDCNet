import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from dataset import Inference_Dataset
import pandas as pd
from model import PredictModel
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_curve, auc
import pickle
import math
from encoding import *


def cover_dict(path):
    file_path = path
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    tensor_dict = {key: tf.constant(value) for key, value in data.items()}
    new_data = {i: value for i, (key, value) in enumerate(tensor_dict.items())}
    return new_data


def score(y_test, y_pred):
    auc_roc_score = roc_auc_score(y_test, y_pred)
    prec, recall, _ = precision_recall_curve(y_test, y_pred)
    prauc = auc(recall, prec)
    y_pred_print = [round(y, 0) for y in y_pred]
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_print).ravel()
    se = tp / (tp + fn)
    sp = tn / (tn + fp)
    acc = (tp + tn) / (tp + fn + tn + fp)
    mcc = (tp * tn - fn * fp) / math.sqrt((tp + fn) * (tp + fp) * (tn + fn) * (tn + fp))
    P = tp / (tp + fp)
    F1 = (P * se * 2) / (P + se)
    BA = (se + sp) / 2
    PPV = tp / (tp + fp)
    NPV = tn / (fn + tn)
    return tp, tn, fn, fp, se, sp, mcc, acc, auc_roc_score, F1, BA, prauc, PPV, NPV



def process_list(input_list):
    input_list.append(np.mean(input_list))
    mean_value = np.mean(input_list[:-1])
    std_value = np.std(input_list[:-1], ddof=0)
    mean_range = f'{mean_value:.4f} ± {std_value:.4f}'
    input_list[-1] = mean_range
    print(input_list)
    return input_list


def extract_tensors(index, Peptide_dict):
    return Peptide_dict[index]


Peptide_dict = cover_dict('Peptide_test_random2.pkl')


def extract_t2(file_path, index, padding=71):
    df = pd.read_excel(file_path)
    sequences = df['sequence'].tolist()


    selected_sequences = [sequences[index]]

    one_hot_encoded = one_hot_padding(selected_sequences, padding)
    pos_encoded = position_encoding(padding)
    blosum_encoded = blosum62_padding(selected_sequences, padding)
    zscale_encoded = zscale_padding(selected_sequences, padding)

    pos_encoded_expanded = np.expand_dims(pos_encoded, axis=0).repeat(len(selected_sequences), axis=0)

    combined_encoding = np.concatenate(
        (one_hot_encoded, pos_encoded_expanded, blosum_encoded, zscale_encoded), axis=2
    )


    return tf.constant(combined_encoding, dtype=tf.float32)



medium = {'name': 'Medium', 'num_layers': 6, 'num_heads': 8, 'd_model': 256, 'path': 'medium_weights', 'addH': True}
arch = medium
num_layers = arch['num_layers']
num_heads = arch['num_heads']
d_model = arch['d_model']
addH = arch['addH']
dff = d_model * 2
vocab_size = 18
dense_dropout = 0.4
seed = 1
df = pd.read_excel('data_test.xlsx')
np.random.seed(seed=seed)
tf.random.set_seed(seed=seed)
sml_list1 = df['Payload SMILES'].tolist()
sml_list2 = df['Linker SMILES'].tolist()

ans = []
y_preds = []
res = []
n = len(sml_list1)
for i in range(n):
    x1 = [sml_list1[i]]
    x2 = [sml_list2[i]]
    t1 = extract_tensors(i, Peptide_dict)
    t2 = extract_t2('data_test.xlsx', i)
    t1 = tf.expand_dims(t1, axis=0)

    inference_dataset1 = Inference_Dataset(x1, addH=addH).get_data()
    inference_dataset2 = Inference_Dataset(x2, addH=addH).get_data()

    x1, adjoin_matrix1, smiles1, atom_list1 = next(iter(inference_dataset1.take(1)))
    x2, adjoin_matrix2, smiles2, atom_list2 = next(iter(inference_dataset2.take(1)))

    seq1 = tf.cast(tf.math.equal(x1, 0), tf.float32)
    seq2 = tf.cast(tf.math.equal(x2, 0), tf.float32)

    mask1 = seq1[:, tf.newaxis, tf.newaxis, :]
    mask2 = seq2[:, tf.newaxis, tf.newaxis, :]

    model = PredictModel(num_layers=num_layers,
                         d_model=d_model,
                         dff=dff,
                         num_heads=num_heads,
                         vocab_size=vocab_size,
                         a=1,
                         dense_dropout=dense_dropout)

    pred = model(x1=x1, mask1=mask1, training=False, adjoin_matrix1=adjoin_matrix1, x2=x2, mask2=mask2,
                 adjoin_matrix2=adjoin_matrix2, t1=t1, t2=t2)
    model.load_weights('classification_weights/PDC_2024.h5')

    x = model(x1=x1, mask1=mask1, training=False, adjoin_matrix1=adjoin_matrix1, x2=x2, mask2=mask2,
              adjoin_matrix2=adjoin_matrix2, t1=t1, t2=t2)
    y_preds.append(x)

y_preds = tf.sigmoid(y_preds)
y_preds = tf.reshape(y_preds, (-1,))
y_hat = tf.where(y_preds < 0.5, 0, 1)
for i in y_preds.numpy():
    ans.append(i)
for i in y_hat.numpy():
    res.append(i)
print(ans)
print(res)

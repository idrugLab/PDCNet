# PDCNet
semi-supervised learning for PDC property prediction.
![image](https://github.com/idrugLab/PDCNet/blob/main/PDCNet.png)
# Dataset
The dataset used to build the model is `data.xlsx`, split into a training set, validation set and test set in a ratio of 8:1:1. The `sequence` and `ID` columns in the data are used to generate the ESM-2 embedding `peptide.pkl`.
# Requried package:
## Example of ESM-2 environment installation
```
conda create -n esm-2 python==3.9
pip install fair-esm  # latest release, OR:
pip install git+https://github.com/facebookresearch/esm.git  # bleeding edge, current repo main branch
 ```
## Example of PDCNet environment installation
```
conda create -n PDCNet python==3.9
pip install -r requirements.txt
```
# Train and Predict
## Example of obtaining embeddings for peptides
```
conda activate esm-2
python ESM-2.py
```
After completion of the run, you will find a .pkl file in the current directory. It is a dictionary where the keys are PDC IDs (if there is no PDC ID, you can add a column with numerical values to the original data and name it PDC ID), and the values are tensors of 640 dimensions.

## Example of training PDCNet
Create a folder named "medium3_weights" and place the file "bert_weightsMedium_20.h5" from this repository into that folder.
```
conda activate PDCNet
python class.py
```

## Examples of using PDCNet to inference.
Prepare the data to be predicted, including the `sequence` and `ID` columns for generating the embeddings. Then, make sure you have the `SMILES` columns for the linker and payload to run the `inference.py` file to make predictions. An example run is as follows
```
conda activate PDCNet
python inference.py
```
## Using PDCNet for predictions
`You can visit the (https://PDCNet.idruglab.cn) website to make predictions.`

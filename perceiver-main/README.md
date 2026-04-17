# perceiver
Network model to predict gene expression and cell properties from masked-out scRNA data

Network model based off of "A single-cell gene expression language model"
https://arxiv.org/abs/2210.14330

which in turn is based on the Perceiver IO architecture
https://arxiv.org/abs/2107.14795

## Creating the train and test set
Run create_dataset.py - must modify source_path and target_path in file
Source data must be taken from /sc/arion/projects/psychAD/NPS-AD/.../*.h5ad
Creates a numpy memmap data structure for the epxression data, and a pkl file for the cell properties and other metadata

## Training the model
Run train.py
Network and training configs are found in config.py

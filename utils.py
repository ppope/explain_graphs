import os
import time
import pickle
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

from keras import metrics
from keras import backend as K
from keras.models import Model
from keras.layers import (Input, Dense, Softmax, Lambda)
from keras.optimizers import Adagrad
from keras.initializers import RandomNormal

from rdkit.Chem import MolFromSmiles
from deepchem.feat import WeaveFeaturizer, ConvMolFeaturizer
from deepchem.splits import RandomSplitter, ScaffoldSplitter
from chainer_chemistry.dataset.parsers.csv_file_parser import CSVFileParser
from chainer_chemistry.dataset.preprocessors.nfp_preprocessor import NFPPreprocessor

from sklearn.metrics import (accuracy_score, precision_score, roc_auc_score,
                             recall_score, auc, average_precision_score,
                             roc_curve, precision_recall_curve)


def load_data(csv_fp, labels_col="p_np", smiles_col="smiles"):
    """
    Load BBBP data
    """
    csvparser = CSVFileParser(NFPPreprocessor(), labels = labels_col, smiles_col = smiles_col)
    data_ = csvparser.parse(csv_fp,return_smiles = True)
    atoms, adjs, labels = data_['dataset'].get_datasets()
    smiles = data_['smiles']
    return {"atoms": atoms,
            "adjs": adjs,
            "labels":labels,
            "smiles": smiles}


def normalize_adj(adj, symmetric=True):
    if symmetric:
        d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0)
        a_norm = adj.dot(d).transpose().dot(d).tocsr()
    else:
        d = sp.diags(np.power(np.array(adj.sum(1)), -1).flatten(), 0)
        a_norm = d.dot(adj).tocsr()
    return a_norm


def preprocess_adj(adj, symmetric=True):
    adj = sp.csr_matrix(adj)
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize_adj(adj, symmetric)
    return adj.toarray()


def preprocess(raw_data, feats="convmol"):
    """
    Preprocess molecule data
    """
    labels = raw_data['labels']
    smiles = raw_data['smiles']
    adjs = raw_data['adjs']
    num_classes = np.unique(labels).shape[0]

    #One hot labels
    labels_one_hot = np.eye(num_classes)[labels.reshape(-1)]

    if feats == "weave":
        featurizer = WeaveFeaturizer()
    elif feats == "convmol":
        featurizer =  ConvMolFeaturizer()

    mol_objs = featurizer.featurize([MolFromSmiles(smile) for smile in smiles])

    #Sort feature matrices by node degree
    node_features = []
    for i,feat in enumerate(mol_objs):
        sortind = np.argsort(adjs[i].sum(axis = 1) - 1)
        N = len(sortind)
        sortMatrix = np.eye(N)[sortind,:]
        node_features.append(np.matmul(sortMatrix.T, feat.get_atom_features()))

    #Normalize Adjacency Mats
    norm_adjs = [preprocess_adj(A) for A in adjs]

    return {'labels_one_hot': labels_one_hot,
            'node_features': node_features,
            'norm_adjs': norm_adjs}


def dense(n_hidden, activation='relu',
          init_stddev=0.1, init_mean=0.0,
          seed=None):
    """
    Helper function for configuring `keras.layers.Dense`
    """
    kernel_initializer = RandomNormal(mean=init_mean, stddev=init_stddev, seed=seed)
    bias_initializer = RandomNormal(mean=init_mean, stddev=init_stddev, seed=seed)
    return Dense(n_hidden, activation=activation,
                 kernel_initializer=kernel_initializer,
                 bias_initializer=bias_initializer,
                 use_bias=True)

def matmul(XY):
    """
    Matrix multiplication for use with `keras.layers.Lambda`
    Compatible with `keras.models.Model`
    """
    X,Y = XY
    return K.tf.matmul(X,Y)


def GAP(X):
    return K.tf.reduce_mean(X, axis=1, keepdims=True)


def keras_gcn(config):
    """
    Keras GCN for graph classification

    We must "fool" `keras.model.Model` into accepting
    inputs tensors (adjacency matrix, node features)
    of different shape in the first axis than target tensors.

    Otherwise, e.g.
    ```
    ValueError: Input arrays should have the same number of samples
    as target arrays. Found 20 input samples and 1 target samples.
    ```

    Additionally, to write a matrix multiplication layer compatible with
    `keras.model.Models` we must use `keras.layers.Lambda`
    """
    d = config['d']
    init_stddev = config['init_stddev']
    L1 = config['L1']
    L2 = config['L2']
    L3 = config['L3']
    N = config['N']
    num_classes = config['num_classes']
    batch_size = config['batch_size']
    assert batch_size == 1, "Batch size != 1 Not Implemented!"

    A_batch = Input(shape=(batch_size,N,N), batch_shape=(batch_size,N,N))
    X_batch = Input(shape=(batch_size,N,d), batch_shape=(batch_size,N,d))
    Y = Input(shape=(batch_size, num_classes), batch_shape=(batch_size, num_classes))

    h1 = dense(L1)(Lambda(matmul)([A_batch, X_batch]))
    h2 = dense(L2)(Lambda(matmul)([A_batch, h1]))
    h3 = dense(L3)(Lambda(matmul)([A_batch, h2]))
    gap = Lambda(GAP)(h3)
    gap=  Lambda(lambda y: K.squeeze(y, 1))(gap)
    logits = dense(num_classes, activation='linear')(gap)
    Y_hat = Softmax()(logits)

    model = Model(inputs=[A_batch, X_batch], outputs=Y_hat)
    model.compile(optimizer='adam',
                  loss=lambda y_true,y_pred: K.mean(K.binary_crossentropy(
                      y_true, logits, from_logits=True), axis=-1))
                  #loss='binary_crossentropy')
    return model


def gcn_train(model, data, num_epochs, train_inds):
    norm_adjs = data['norm_adjs']
    labels_one_hot = data['labels_one_hot']
    labels=np.argmax(labels_one_hot,axis=1)

    negind=np.argwhere(labels==0).squeeze()
    posind=np.argwhere(labels==1).squeeze()
    Nepoch=min(len(negind),len(posind))

    node_features = data['node_features']
    total_loss = []
    for epoch in range(num_epochs):
        epoch_loss = []
        epoch_correct = []
        #Train
        rand_inds = np.random.permutation(np.concatenate((np.random.permutation(negind)[:Nepoch],np.random.permutation(posind)[:Nepoch])))
        for ri in rand_inds:
            A_arr = norm_adjs[ri][np.newaxis, :, :]
            X_arr = node_features[ri][np.newaxis, :, :]
            Y_arr = labels_one_hot[ri][np.newaxis, :]
            sample_loss = model.train_on_batch(x=[A_arr, X_arr], y=Y_arr, )
            epoch_loss.append(sample_loss)
        #Eval
        for ri in rand_inds:
            A_arr = norm_adjs[ri][np.newaxis, :, :]
            X_arr = node_features[ri][np.newaxis, :, :]
            Y_arr = labels_one_hot[ri][np.newaxis, :]
            sample_pred = model.predict([A_arr, X_arr])
            sample_correct = np.argmax(sample_pred) == np.argmax(Y_arr)
            epoch_correct.append(sample_correct)
        mean_epoch_loss = sum(epoch_loss) / len(epoch_loss)
        epoch_acc = sum(epoch_correct) / len(epoch_correct)
        print("Epoch: {}, Mean Loss: {:.3f}, Accuracy: {:.3f}".format(epoch, mean_epoch_loss, epoch_acc))
        total_loss.extend(epoch_loss)
    last_epoch_acc = epoch_acc
    return total_loss, last_epoch_acc


class MockDataset:
    """Mock Dataset class for a DeepChem Dataset"""
    def __init__(self, smiles):
        self.ids = smiles

    def __len__(self):
        return len(self.ids)


def partition_train_val_test(smiles, dataset):
    """
    Split a molecule dataset (SMILES) with deepchem built-ins
    """

    ds = MockDataset(smiles)

    if dataset == "BBBP":
        splitter = ScaffoldSplitter()
    elif dataset == "BACE":
        splitter = ScaffoldSplitter()
    elif dataset == "TOX21":
        splitter = RandomSplitter()

    train_inds, val_inds, test_inds = splitter.split(ds)

    return {"train_inds": train_inds,
            "val_inds": val_inds,
            "test_inds": test_inds}


def run_train(config, data, inds):
    """
    Sets splitter. Partitions train/val/test.
    Loads model from config. Trains and evals.
    Returns model and eval metrics.
    """
    train_inds = inds["train_inds"]
    val_inds = inds["val_inds"]
    test_inds = inds["test_inds"]

    model = keras_gcn(config)
    loss, accuracy = gcn_train(model, data, config['num_epochs'], train_inds)

    train_eval = evaluate(model, data, train_inds)
    test_eval = evaluate(model, data, test_inds)
    val_eval = evaluate(model, data, val_inds)

    return model, {"train": train_eval,
                   "test": test_eval,
                   "val": val_eval}



def print_evals(eval_dict):
    print("Accuracy: {0:.3f}".format(eval_dict["accuracy"]))
    print("Precision: {0:.3f}".format(eval_dict["precision"]))
    print("AUC ROC: {0:.3f}".format(eval_dict["roc_auc"]))
    print("AUC PR: {0:.3f}".format(eval_dict["avg_precision"]))
    print("eval time (s): {0:.3f}".format(eval_dict["eval_time"]))


def evaluate(model, data, inds, thresh=0.5):
    t_test = time.time()
    preds = np.concatenate([model.predict([data["norm_adjs"][i][np.newaxis, :, :],
                              data["node_features"][i][np.newaxis, :, :]])
                              for i in inds], axis=0)

    preds = preds[:,1]
    labels = np.array([np.argmax(data["labels_one_hot"][i]) for i in inds])
    roc_auc = roc_auc_score(labels, preds)
    roc_curve_ = roc_curve(labels, preds)
    precision = precision_score(labels, (preds > thresh).astype('int'))
    acc = accuracy_score(labels, (preds > thresh).astype('int'))
    ap = average_precision_score(labels, preds)
    pr_curve_ = precision_recall_curve(labels, preds)


    return {"accuracy": acc,
            "roc_auc": roc_auc,
            "precision": precision,
            "avg_precision": precision,
            "eval_time": (time.time() - t_test),
            "roc_curve": roc_curve_,
            "pr_curve": pr_curve_}


def print_eval_avg(eval_dict, split, metric):
    N = len(eval_dict.keys())
    vals = [eval_dict[i][split][metric] for i in range(N)]
    return "{0:.3f} +/- {1:.3f}".format(np.mean(vals), np.std(vals))


def occlude_and_predict(X_arr, A_arr, masks, thresh, model):
    """
    COPIES and mutates input data

    Returns predicted CLASS (not prob.) of occluded data
    """
    #Copy node features. We need to edit it.
    X_arr_occ = X_arr.copy()

    #Occlude activated nodes for each explain method
    #NB: array shape is (batch, N, D)
    # and batches are always of size 1
    X_arr_occ[0, masks > thresh, :] = 0

    #Predict on occluded image. Save prediction
    prob_occ = model.predict_on_batch(x=[A_arr, X_arr_occ])

    y_hat_occ = prob_occ.argmax()
    return y_hat_occ

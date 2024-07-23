import tensorflow
from keras import layers,regularizers,Sequential
from tensorflow import keras
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import roc_auc_score
from keras.layers import Lambda
from scipy.stats import pearsonr
import math
import tensorflow as tf
from keras.layers import Layer
from tensorflow.keras.optimizers import Adam
import keras.backend as K
from keras.layers import Layer, Softmax
import keras.backend as K
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import LearningRateScheduler
class PCCMonitor(Callback):
    def __init__(self, validation_data, filepath, verbose=1):
        super(PCCMonitor, self).__init__()
        self.validation_data = validation_data
        self.filepath = filepath
        self.best_pcc = 0.5 
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
        x_val, y_val = self.validation_data
        preds = self.model.predict(x_val)
        pcc, _ = pearsonr(preds.flatten(), y_val)
        if pcc > self.best_pcc:
            self.best_pcc = pcc
            self.model.save(self.filepath, overwrite=True)
            if self.verbose:
                print(f'\nEpoch {epoch+1}: PCC improved to {self.best_pcc:.4f}, saving model to {self.filepath}')
        else:
            if self.verbose:
                print(f'\nEpoch {epoch+1}: PCC did not improve from {self.best_pcc:.4f}')

class MyModel:

    def __init__(self, dim_pep=128, dim_mhc=128):
        self.dim_pep = dim_pep
        self.dim_mhc = dim_mhc
        self.model = self.build_model()

    def build_model(self):
        pep_input = keras.Input(shape=(11, 10, 1), name="pinp")
        mhc_input = keras.Input(shape=(34, 10, 1), name="mhcinp")
        padding1_layer1 = layers.ZeroPadding2D(padding=((0, 0), (0, 0)))
        padding1_layer2 = layers.ZeroPadding2D(padding=((0, 0), (0, 0)))
        padding2_layer1 = layers.ZeroPadding2D(padding=((0, 2), (0, 0)))
        padding2_layer2 = layers.ZeroPadding2D(padding=((1, 1), (0, 0)))
        padding3_layer1 = layers.ZeroPadding2D(padding=((0, 4), (0, 0)))
        padding3_layer2 = layers.ZeroPadding2D(padding=((1, 3), (0, 0)))
        padding4_layer1 = layers.ZeroPadding2D(padding=((1, 5), (0, 0)))
        padding4_layer2 = layers.ZeroPadding2D(padding=((3, 3), (0, 0)))

        Conv1 = layers.Conv2D(self.dim_pep, (1, 10), strides=(1, 1), activation='relu')
        Conv2 = layers.Conv2D(self.dim_pep, (3, 10), strides=(1, 1), activation='relu')
        Conv3 = layers.Conv2D(self.dim_pep, (5, 10), strides=(1, 1), activation='relu')
        Conv4 = layers.Conv2D(self.dim_pep, (7, 10), strides=(1, 1), activation='relu')
        reshape = layers.Reshape((-1, 128, 1))
        def process(input_tensor,padding,Conv):
            x = padding(input_tensor)
            x = pep_Conv(x)
            x = reshape(x)
            return x

        pep_re1_1 = process(pep_input, padding1_layer1, Conv1)
        pep_re1_2 = process(pep_input, padding1_layer2, Conv1)
        pep_re2_1 = process(pep_input, padding2_layer1, Conv2)
        pep_re2_2 = process(pep_input, padding2_layer2, Conv2)
        pep_re3_1 = process(pep_input, padding3_layer1, Conv3)
        pep_re3_2 = process(pep_input, padding3_layer2, Conv3)
        pep_re4_1 = process(pep_input, padding4_layer1, Conv4)
        pep_re4_2 = process(pep_input, padding4_layer2, Conv4)

        mhc_re1_1 = process(mhc_input, padding1_layer1, Conv1)
        mhc_re1_2 = process(mhc_input, padding1_layer2, Conv1)
        mhc_re2_1 = process(mhc_input, padding2_layer1, Conv2)
        mhc_re2_2 = process(mhc_input, padding2_layer2, Conv2)
        mhc_re3_1 = process(mhc_input, padding3_layer1, Conv3)
        mhc_re3_2 = process(mhc_input, padding3_layer2, Conv3)
        mhc_re4_1 = process(mhc_input, padding4_layer1, Conv4)
        mhc_re4_2 = process(mhc_input, padding4_layer2, Conv4)

        cont1_1 = layers.concatenate([pep_re1_1, mhc_re1_1], axis=1)
        cont1_2 = layers.concatenate([pep_re1_2, mhc_re1_2], axis=1)
        cont2_1 = layers.concatenate([pep_re2_1, mhc_re2_1], axis=1)
        cont2_2 = layers.concatenate([pep_re2_2, mhc_re2_2], axis=1)
        cont3_1 = layers.concatenate([pep_re3_1, mhc_re3_1], axis=1)
        cont3_2 = layers.concatenate([pep_re3_2, mhc_re3_2], axis=1)
        cont4_1 = layers.concatenate([pep_re4_1, mhc_re4_1], axis=1)
        cont4_2 = layers.concatenate([pep_re4_2, mhc_re4_2], axis=1)

        cont_re = layers.concatenate([cont1_1, cont1_2, cont2_1, cont2_2, cont3_1, cont3_2, cont4_1, cont4_2], axis=1)
        cont_Maxpooling = layers.MaxPooling2D(pool_size=(360, 1))
        LSTM_layer1 = layers.Bidirectional(layers.LSTM(128, return_sequences=True, input_shape=(None, 128)))
        LSTM_layer2 = Sequential([layers.Bidirectional(layers.LSTM(128, return_sequences=True, input_shape=(None, 128))),
                                  layers.Bidirectional(layers.LSTM(128, return_sequences=True))
                                  ])
        def cont_process(cont):
            cont = tf.squeeze(cont, axis=-1)
            cont1 = LSTM_layer1(cont)
            cont2 = LSTM_layer2(cont)
            return cont1, cont2
        cont_re1, cont_re2 = cont_process(cont_re)
        cont_re = layers.concatenate([cont_re1, cont_re2], axis=-1)
        cont_re = tf.expand_dims(cont_re, axis=-1)
        cont_re = cont_Maxpooling(cont_re)
        cont_re = tf.squeeze(cont_re, axis=1)
        cont_re = tf.squeeze(cont_re, axis=-1)
        dropout_05 = layers.Dropout(0.5)
        dropout_045 = layers.Dropout(0.45)
        dense1 = layers.Dense(1024)
        dense2 = layers.Dense(768)
        dense3 = layers.Dense(640)
        dense4 = layers.Dense(384)
        dense6 = layers.Dense(1, activation='sigmoid')

        cont_re = dense1(cont_re)
        cont_re = relu(cont_re)
        cont_re = dropout_05(cont_re)

        cont_re = dense2(cont_re) 
        cont_re = relu(cont_re)

        cont_re = dense3(cont_re)
        cont_re = relu(cont_re)

        cont_re = dense4(cont_re)
        cont_re = relu(cont_re)

        cont_re = dense6(cont_re)
        output = cont_re
        model = keras.Model(inputs=[pep_input, mhc_input], outputs=output)
        return model
CUTOFF = 1.0 - math.log(500, 50000)
X_pep = np.load('dataset/.npy')
X_mhc = np.load('dataset/.npy')
labels = np.load('dataset/.npy')

with open('dataset/ba_cv_id.txt', 'r') as f:
    fold_ids = np.array([int(line.strip()) for line in f.readlines()])

all_preds = []
all_labels = []

unique_folds = np.unique(fold_ids)
for fold_number in unique_folds:
    print(f"Training on fold {fold_number}")
    train_idx = np.where(fold_ids != fold_number)[0]
    val_idx = np.where(fold_ids == fold_number)[0]

    X_pep_train, X_pep_val = X_pep[train_idx], X_pep[val_idx]
    X_mhc_train, X_mhc_val = X_mhc[train_idx], X_mhc[val_idx]
    labels_train, labels_val = labels[train_idx], labels[val_idx]

    mymodel = MyModel()
    model = mymodel.model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

    filepath = f'model/.h5'
    pcc_monitor = PCCMonitor(validation_data=([X_pep_val, X_mhc_val], labels_val), filepath=filepath)

    def scheduler(epoch, lr):
        if (epoch + 1) % 10 == 0:  
            return lr * 0.5        
        return lr                  

    lr_scheduler = LearningRateScheduler(scheduler)
    history = model.fit(
        [X_pep_train, X_mhc_train], labels_train,
        batch_size=128,
        epochs=20,
        validation_data=([X_pep_val, X_mhc_val], labels_val),
        callbacks=[pcc_monitor, lr_scheduler]
    )
    model.load_weights(filepath)
    preds = model.predict([X_pep_val, X_mhc_val]).flatten()
    all_preds.extend(preds)
    all_labels.extend(labels_val)

    eval_loss, eval_mae = model.evaluate([X_pep_val, X_mhc_val], labels_val)
    print(f"Fold {fold_number} - Evaluation Loss: {eval_loss}")
    print(f"Fold {fold_number} - Mean Absolute Error: {eval_mae}")
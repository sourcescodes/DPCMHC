import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

X_pep = np.load('dataset/.npy')
X_mhc = np.load('dataset/.npy')

all_preds = []

num_models = 5
for fold_number in range(0, num_models):
    filepath = f'model/.h5'
    model = load_model(filepath)
    
    preds = model.predict([X_pep, X_mhc]).flatten()
    
    if fold_number == 0:
        all_preds = np.zeros_like(preds)
    
    all_preds += preds

all_preds /= num_models
import numpy as np
from sklearn.model_selection import KFold
from .data_loader import load_adhd_data, load_asd_data
from .utils import pad_and_reshape_data
from .model import create_lstm_model

def train_and_evaluate(n_epochs=60):
    # Assuming masker is initialized elsewhere
    masker = NiftiMapsMasker()  # Example placeholder
    lst_adhd, lst_label_adhd = load_adhd_data(masker)
    lst_autsm, lst_label_autsm = load_asd_data(masker)
    
    lst_label_adhd.extend(lst_label_autsm)
    lst_adhd.extend(lst_autsm)
    
    x_data, y_data = pad_and_reshape_data(lst_adhd, lst_label_adhd)
    
    kfold = KFold(10, random_state=0, shuffle=True)
    for train, test in kfold.split(x_data, y_data):
        X_train, X_test = x_data[train], x_data[test]
        y_train, y_test = y_data[train], y_data[test]

        model = create_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        
        model.fit(X_train, y_train, epochs=n_epochs, validation_data=(X_test, y_test))
        # Add evaluation and metrics here

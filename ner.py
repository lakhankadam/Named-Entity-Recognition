import pandas as pd
import numpy as np
from train_model import get_lstm_model, train_model
from tensorflow.keras.utils import plot_model
from split_data import split_train_test_val

def run():
    results = pd.DataFrame()
    train_tokens, val_tokens, test_tokens, train_tags, val_tags, test_tags = split_train_test_val()
    model_bilstm_lstm = get_lstm_model()
    # plot_model(model_bilstm_lstm)
    results['with_add_lstm'] = train_model(train_tokens, np.array(train_tags), model_bilstm_lstm)

if __name__ == '__main__':
    run()
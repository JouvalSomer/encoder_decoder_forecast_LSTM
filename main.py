



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_loader import TimeSeriesDataset
from encoder import Encoder
from decoder import Decoder
from model import EncoderDecoderModel

from gen_synt_data import make_dataset, plot_data, create_input_output_sequences, train_val_test_split
from gen_synt_data import train_model, eval_model, inverse_transform, plot_pred_vs_actuals

def run_synt_data_test():
    input_length = 100 # sequence length or lookback period
    output_length = 20
    batch_size = 64
    train_val_test_ratio = (0.7, 0.1, 0.2)

    synt_data = make_dataset(time_steps = 3000, frequency = 0.01, num_features=20)

    plot_data(synt_data)

    X, y = create_input_output_sequences(synt_data, input_length, output_length)
    split_data, stat_values = train_val_test_split(X, y, ratio=train_val_test_ratio, normalize=True)

    X_train, y_train, X_val, y_val, X_test, y_test = split_data
    target_mean, target_std = stat_values


    train_dataset = TimeSeriesDataset(X_train, y_train)
    val_dataset = TimeSeriesDataset(X_val, y_val)
    test_dataset = TimeSeriesDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)


    input_size = X_train.shape[2]  # Number of features
    output_size = 1  # Predicting one value at each time step
    hidden_size = 64
    num_layers = 1


    encoder = Encoder(
        input_size=input_size, 
        hidden_size=hidden_size, 
        num_layers=num_layers
    )

    decoder = Decoder(
        output_size=output_size, 
        hidden_size=hidden_size, 
        num_layers=num_layers
    )

    model = EncoderDecoderModel(encoder, decoder, target_len=output_length)

    metric = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 20
    teacher_forcing_ratio = 0.5

    train_model(model, num_epochs, teacher_forcing_ratio, metric, optimizer, train_loader)

    predictions, actuals = eval_model(model, metric, test_loader)

    predictions_inv = inverse_transform(np.concatenate(predictions, axis=0), target_mean, target_std)
    actuals_inv = inverse_transform(np.concatenate(actuals, axis=0), target_mean, target_std)

    plot_pred_vs_actuals(predictions_inv, actuals_inv)



if __name__=='__main__':
    run_synt_data_test()
    
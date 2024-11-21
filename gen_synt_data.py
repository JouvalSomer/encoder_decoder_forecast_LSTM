
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



def make_target_values(t, frequency):
    target = np.sin(2 * np.pi * frequency * t)
    return target


def make_fearures_values(target, time_steps, num_features=5):
    assert isinstance(num_features, int) and num_features >= 1, "num_features must be an int!"
    # One feature (feature1) is a shifted version of the target, and the others are just randome noice
    shift_steps = 10 
    feature = np.roll(target, shift=shift_steps)
    # Introduce NaNs at the beginning due to the shift
    feature[:shift_steps] = np.nan

    features = [feature]

    for i in range(2, num_features+1):
        np.random.seed(i)
        feature = np.random.normal(0, 0.5, size=time_steps)
        features.append(feature)

    return features

def plot_data(data):
    num_columns_in_df = len(data.columns)
    num_columns = 5
    num_rows = (num_columns_in_df//num_columns) + (num_columns_in_df%num_columns)

    fig, axs = plt.subplots(num_rows, num_columns, figsize=(15, 10), facecolor='w', edgecolor='k')

    axs = axs.ravel()
    for i, column in enumerate(data):
        if i == 0:
            axs[i].set_title(f'Target values')
            axs[i].plot(data['target'])

        elif i == 1:
            axs[i].set_title(f'Feature {i} (shifted target)')
            axs[i].plot(data[f'feature{i}'])

        else:
            axs[i].set_title(f'Feature {i} (Random noice)')
            axs[i].plot(data[f'feature{i}'])
    plt.tight_layout()
    plt.show()



def make_dataset(time_steps = 1000, frequency = 0.005, num_features=5):
    assert isinstance(time_steps, int) and time_steps > 0, "time_steps must be apositive int!"

    t = np.arange(0, time_steps)

    target = make_target_values(t, frequency)
    features = make_fearures_values(target, time_steps, num_features=num_features)

    data = pd.DataFrame({'target': target})

    for i, feature in enumerate(features, 1):
        data[f'feature{i}'] = feature


    # Drop the first shift_steps number of rows as they contain nan values
    data.dropna(inplace=True)

    # Reset the index such that it starts on 0 and not shift_steps
    data.reset_index(drop=True, inplace=True)

    return data

def create_input_output_sequences(data, input_length, output_length):
    X, y = [], []
    feature_columns = data.columns[1:]  # Dynamically select all columns except the first (target)
    target_column = data.columns[0]     # The first column is assumed to be the target
    
    for i in range(len(data) - input_length - output_length + 1):
        X_seq = data.iloc[i:i+input_length][feature_columns].values
        y_seq = data.iloc[i+input_length:i+input_length+output_length][target_column].values
        X.append(X_seq)
        y.append(y_seq)
    
    return np.array(X), np.array(y)



def train_val_test_split(X, y, ratio=(0.7, 0.15, 0.15), normalize=True):
    train_ratio, val_ratio, test_ratio = ratio

    num_samples = X.shape[0]
    train_end = int(train_ratio * num_samples)
    val_end = int((train_ratio + val_ratio) * num_samples)

    X_train = X[:train_end]
    y_train = y[:train_end]

    X_val = X[train_end:val_end]
    y_val = y[train_end:val_end]

    X_test = X[val_end:]
    y_test = y[val_end:]

    # print(f'Training samples: {X_train.shape[0]}')
    # print(f'Validation samples: {X_val.shape[0]}')
    # print(f'Test samples: {X_test.shape[0]}')

    if normalize:
        # Normalize features and targets with the mean and std of the Training set
        feature_means = X_train.mean(axis=(0, 1))
        feature_stds = X_train.std(axis=(0, 1))

        target_mean = y_train.mean()
        target_std = y_train.std()

        X_train = normalize_features(X_train, feature_means, feature_stds)
        X_val = normalize_features(X_val, feature_means, feature_stds)
        X_test = normalize_features(X_test, feature_means, feature_stds)

        y_train = normalize_target(y_train, target_mean, target_std)
        y_val = normalize_target(y_val, target_mean, target_std)
        y_test = normalize_target(y_test, target_mean, target_std)

    split_data = X_train, y_train, X_val, y_val, X_test, y_test
    stat_values = target_mean, target_std # For the invers transformation of the data for visualisation after training

    return split_data, stat_values


def normalize_features(X, means, stds):
    return (X - means) / stds

def normalize_target(y, mean, std):
    return (y - mean) / std

# Inverse transform
def inverse_transform(y_norm, mean, std):
    return y_norm * std + mean


def train_model(model, num_epochs, teacher_forcing_ratio, metric, optimizer, train_loader):
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            
            # For teacher forcing, trg should have shape (batch_size, output_length, 1)
            trg = y_batch.unsqueeze(-1)  # Shape: (batch_size, output_length, 1)
            
            # Forward pass
            outputs = model(X_batch, trg=trg, teacher_forcing_ratio=teacher_forcing_ratio)
            outputs = outputs.squeeze(-1)  # Shape: (batch_size, output_length)
            
            # Compute loss
            loss = metric(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.16f}')

    return model



def eval_model(model, metric, test_loader):
    model.eval()
    with torch.no_grad():
        test_loss = 0
        predictions = []
        actuals = []
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch, trg=None, teacher_forcing_ratio=0)
            outputs = outputs.squeeze(-1)  # Shape: (batch_size, target_len)
            
            # Compute loss
            loss = metric(outputs, y_batch)
            test_loss += loss.item()

            predictions.append(outputs.numpy())
            actuals.append(y_batch.numpy())
        
        avg_test_loss = test_loss / len(test_loader)
        print(f'Test Loss: {avg_test_loss:.10f}')

    return predictions, actuals


def plot_pred_vs_actuals(pred, actuals):
    num_samples = pred.shape[1]

    num_columns = 5
    num_rows = (num_samples//num_columns) + (num_samples%num_columns)

    fig, axs = plt.subplots(num_rows, num_columns, figsize=(15, 10), facecolor='w', edgecolor='k')

    fig.suptitle("Actuals vs. Predicted for all idx's")

    axs = axs.ravel()

    for i in range(num_samples):

        # Get the predicted and actual sequences for the selected sample
        pred_sample = pred[:,i]
        actual_sample = actuals[:,i]

        axs[i].plot(range(actual_sample.shape[0]), actual_sample, label='Actual')
        axs[i].plot(range(pred_sample.shape[0]), pred_sample, label='Predictions', linestyle='--')
        axs[i].set_title(f'Actual vs. Predicted for idx: {i+1}')

    plt.legend(bbox_to_anchor=(0.1, -0.05))
    plt.tight_layout()
    plt.show()


if __name__=='__main__':
    input_length = 50 # sequence length or lookback period
    output_length = 11
    batch_size = 32
    train_val_test_ratio = (0.7, 0.15, 0.15)

    synt_data = make_dataset(time_steps = 1000, frequency = 0.01, num_features=20)

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
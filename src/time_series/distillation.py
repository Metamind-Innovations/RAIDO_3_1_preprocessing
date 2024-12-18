import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn, optim


# # # RCIG distillation implementation # # #

# --------- START --------- #

def prepare_data(csv_file):
    """
    Prepare the data from a CSV file.

    This function reads the CSV file and converts the first column to a timestamp
    in seconds. It then returns the prepared data.

    :param csv_file: The path to the CSV file.
    :type csv_file: str
    :return: A tuple containing the reshaped timestamp values and the corresponding column values.
    :rtype: tuple
    """
    df = pd.read_csv(csv_file, sep=';', parse_dates=[0], dayfirst=True, low_memory=False)
    df['timestamp'] = df[df.columns[0]].astype(int) / 10 ** 9
    prepared_data = df[['timestamp', df.columns[1]]].values.astype(float)
    return prepared_data[:, 0].reshape(-1, 1), prepared_data[:, 1]


def rcig_distillation(X, y, distilled_size=100, steps=100):
    """
    Perform RCIG distillation on the given data.

    This function initializes a synthetic dataset, trains a model using CrossEntropyLoss,
    and returns the distilled data.

    :param X: Input features.
    :type X: torch.Tensor
    :param y: Target values.
    :type y: numpy.ndarray
    :param distilled_size: The size of the distilled dataset, defaults to 100.
    :type distilled_size: int, optional
    :param steps: The number of training steps, defaults to 100.
    :type steps: int, optional
    :return: A tuple containing the distilled input features and target values.
    :rtype: tuple
    """
    X, y = prepare_data('power_small.csv')
    input_dim = X.shape[1]
    unique_classes = torch.unique(torch.tensor(y))
    output_dim = len(unique_classes)

    # Map original target values to class indices
    class_to_index = {cls.item(): idx for idx, cls in enumerate(unique_classes)}
    y_mapped = torch.tensor([class_to_index[val] for val in y], dtype=torch.long)

    # Initialize synthetic dataset.
    distilled_X = torch.randn(distilled_size, input_dim, requires_grad=True)
    distilled_y = y_mapped[:distilled_size]  # Ensure using mapped target values

    model = torch.nn.Linear(input_dim, output_dim)
    optimizer = torch.optim.Adam([distilled_X], lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()  # Use CrossEntropyLoss for classification

    for step in range(steps):
        optimizer.zero_grad()
        outputs = model(distilled_X)
        loss = criterion(outputs, distilled_y)  # No need to squeeze or change dtype
        loss.backward()
        optimizer.step()

    return distilled_X.detach().numpy(), distilled_y.detach().numpy()


# --------- END --------- #


# # # RaT-BPTT distillation implementation # # #

# --------- START --------- #

# Define an RNN model for time series forecasting
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])  # Use the last time step's output
        return out


def load_and_preprocess_data(file_path, column='value'):
    """
    Load and preprocess the data from a CSV file.

    This function reads the CSV file, parses the dates, normalizes the specified column,
    and returns the DataFrame along with the scaler used for normalization.

    :param file_path: The path to the CSV file.
    :type file_path: str
    :param column: The column to normalize, defaults to 'value'.
    :type column: str, optional
    :return: A tuple containing the DataFrame and the scaler used for normalization.
    :rtype: tuple
    """
    df = pd.read_csv(file_path, sep=';', parse_dates=[0], dayfirst=True, low_memory=False)
    df[df.columns[0]] = pd.to_datetime(df[df.columns[0]], format='%d/%m/%Y %H:%M')
    df = df.sort_values(df.columns[0])
    scaler = MinMaxScaler()
    df[f'{column}_normalized'] = scaler.fit_transform(df[[column]])
    return df, scaler


def create_sequences(df, sequence_length=10, colum='value'):
    """
    Create sequences and targets from the DataFrame for time series forecasting.

    This function generates sequences of the specified length from the normalized column
    and their corresponding targets.

    :param df: The DataFrame containing the data.
    :type df: pandas.DataFrame
    :param sequence_length: The length of the sequences to create.
    :type sequence_length: int
    :param colum: The column to use for creating sequences, defaults to 'value'.
    :type colum: str, optional
    :return: A tuple containing the sequences and targets.
    :rtype: tuple
    """
    sequences = []
    targets = []
    for i in range(len(df) - sequence_length):
        seq = df[f'{colum}_normalized'].iloc[i:i + sequence_length].values
        target = df[f'{colum}_normalized'].iloc[i + sequence_length]
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)


def train_rat_bptt(model, train_data, train_targets, num_epochs=100):
    """
    Train the RaT-BPTT model.

    This function trains the given model using the provided training data and targets
    for a specified number of epochs.

    :param model: The RNN model to train.
    :type model: torch.nn.Module
    :param train_data: The training data.
    :type train_data: torch.Tensor
    :param train_targets: The training targets.
    :type train_targets: torch.Tensor
    :param num_epochs: The number of epochs to train the model, defaults to 100.
    :type num_epochs: int, optional
    :return: The trained model.
    :rtype: torch.nn.Module
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(train_data)
        loss = criterion(outputs, train_targets)
        loss.backward()
        optimizer.step()
    return model


def generate_new_data(model, initial_sequence, scaler, num_predictions=10):
    """
    Generate new data using the trained model.

    This function generates new data points based on an initial sequence and a
    trained model. The generated data is scaled back to the original range using
    the provided scaler.

    :param model: The trained RNN model.
    :type model: torch.nn.Module
    :param initial_sequence: The initial sequence to start the generation.
    :type initial_sequence: numpy.ndarray
    :param scaler: The scaler used for normalizing the data.
    :type scaler: sklearn.preprocessing.MinMaxScaler
    :param num_predictions: The number of predictions to generate, defaults to 10.
    :type num_predictions: int, optional
    :return: A list containing the generated data points.
    :rtype: list
    """
    model.eval()
    generated_data = []
    current_seq = initial_sequence
    for _ in range(num_predictions):
        current_seq_tensor = torch.tensor(current_seq[np.newaxis, :, np.newaxis], dtype=torch.float32)
        next_value = model(current_seq_tensor).item()
        generated_data.append(next_value)
        # Update the sequence with the new value
        current_seq = np.append(current_seq[1:], next_value)
    return scaler.inverse_transform(np.array(generated_data).reshape(-1, 1))


def add_predictions_to_df(df, predictions, freq='min'):
    """
    Add predicted values to the DataFrame.

    This function adds the predicted values to the original DataFrame with
    corresponding timestamps.

    :param df: The original DataFrame.
    :type df: pandas.DataFrame
    :param predictions: The predicted values.
    :type predictions: list
    :param freq: The frequency for the timestamps, defaults to 'M' (monthly).
    :type freq: str, optional
    :return: The DataFrame with added predictions.
    :rtype: pandas.DataFrame
    """
    last_timestamp = df['time'].iloc[-1]
    prediction_timestamps = pd.date_range(start=last_timestamp, periods=len(predictions) + 1, freq=freq)[1:]
    predicted_df = pd.DataFrame({'time': prediction_timestamps, 'predicted_value': predictions.flatten()})
    return pd.concat([df, predicted_df], ignore_index=True)


def rat_bptt_predictions(column='value', sequence_length=10, epochs=50, num_predictions=10):
    """
    Generate predictions using the RaT-BPTT model.

    This function loads and preprocesses the data, creates sequences, trains the
    RaT-BPTT model, generates new data, and adds the predicted values to the original
    DataFrame.

    :param column: The column to use for predictions, defaults to 'value'.
    :type column: str, optional
    :param sequence_length: The length of the sequences to create, defaults to 10.
    :type sequence_length: int, optional
    :return: The DataFrame with added predictions.
    :rtype: pandas.DataFrame
    """
    # Load and preprocess the data
    df, scaler = load_and_preprocess_data('power_small.csv', column)

    # Create sequences
    sequences, targets = create_sequences(df, sequence_length)

    # Prepare data for PyTorch (convert to tensors)
    train_data = torch.tensor(sequences[:, :, np.newaxis], dtype=torch.float32)  # Add feature dimension
    train_targets = torch.tensor(targets[:, np.newaxis], dtype=torch.float32)

    # Initialize model
    model = RNNModel(1, 64, 1)

    # Train the model
    trained_model = train_rat_bptt(model, train_data, train_targets, num_epochs=epochs)

    # Generate new data (forecast future values)
    initial_sequence = sequences[-1]
    new_data_rat_bptt = generate_new_data(trained_model, initial_sequence, scaler, num_predictions)
    # Add the predicted values to the original DataFrame
    return add_predictions_to_df(df, new_data_rat_bptt)


# Add below lines to the endpoint
# X, y = prepare_data('power_small.csv')
# distilled_rcig_X, distilled_rcig_y = rcig_distillation(torch.tensor(X, dtype=torch.float32), y)
# print(f'Distilled Data (RCIG): {distilled_rcig_X}, {distilled_rcig_y}')

# x = rat_bptt_predictions(column='value', sequence_length=50)
# print(f'Distilled Data (RAT): {x.tail(30)}')

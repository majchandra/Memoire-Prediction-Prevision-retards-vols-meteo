# Importing Libraries

import subprocess

# Ensure all necessary libraries are installed
libraries = ["pandas", "torch", "scikit-learn", "numpy"]
for lib in libraries:
    subprocess.run(["pip", "install", lib], check=True)
    
import time
import itertools
import pandas as pd
import numpy as np
import torch
import sklearn
import torch.nn 
import torch.optim 

# Fixed random seed for reproducibility of results
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print(RANDOM_SEED)


# Implementation of Deep Learning Model
# Long Short-Term Memory (LSTM)

# The timing of the weather data can be changed (0h, 2h, 4h, 8h, 16h, 24h or 48h before flight departure)
timing = 0
# timing = 2
# timing = 4
# timing = 8 
# timing = 16
# timing = 24
# timing = 48

# Features selected for training and evaluation of models
features = ['DEP_DELAY', 'WHEELS_OFF', 'TAXI_OUT', 'FL_DATE', 'CRS_ARR_TIME', 'DEP_DEL15', 
            'CRS_ELAPSED_TIME', 'Pressure (hPa)','CRS_DEP_TIME', 'Humidity (%)','Temperature (°C)'
            ,'Wind Speed (km/h)','Wind','Condition']

target = 'STATUS'

# Load training, validation and test data
df_train = pd.read_csv(f"Training_Dataset_{timing}h.csv")
df_val = pd.read_csv(f"Validation_Dataset_{timing}h.csv")
df_test = pd.read_csv(f"Testing_Dataset_{timing}h.csv")

# Separation of features and target variable
df_train_resampled_X = df_train[features]
df_train_resampled_y = df_train[target]
df_val_resampled_X = df_val[features]
df_val_resampled_y = df_val[target]
df_test_X = df_test[features]
df_test_y = df_test[target]

# Use a GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class LSTM_Model(torch.nn.Module):
    def __init__(self, input_size, LSTM_size, output_size, num_layers, dropout):
        """
        Initialises the LSTM modle.

        Args:
            input_size (int): Size of model inputs.
            LSTM_size (int): Size of the LSTM layers.
            output_size (int): Size of model outputs.
            num_layers (int): Number of stacked LSTM layers.
            dropout (float): Dropout rate used.
        """
        super(LSTM_Model, self).__init__()
        self.LSTM_size = LSTM_size
        self.num_layers = num_layers
        
        # Definition of the LSTM layer
        self.lstm = torch.nn.LSTM(input_size, LSTM_size, num_layers, batch_first=True, dropout=dropout)
        
        # Output layer
        self.fc = torch.nn.Linear(LSTM_size, output_size)
    
    def forward(self, x):
        """
        Defines the forward propagation function for the LNN model.

        Args:
            x (Tensor): Model inputs.

        Returns:
            Tensor: Model output after the forward propagation.
        """
        # Initialise hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.LSTM_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.LSTM_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))  
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

def prepare_lstm_data(df_X, df_y, sequence_length, input_size, device):
    """
    Prepares data for the LSTM model 

    Args:
        df_X (pd.DataFrame): A DataFrame containing the input features.
        df_y (pd.Series): A Serie contenaining target labels.
        sequence_length (int): Length of time sequences used.
        input_size (int): Size of model inputs.
        device (torch.device): The system to which the tensors are transferred.

    Returns:
        Tuple: Two PyTorch tensors for training and evaluating the LSTM model.
    """
    # Calculation of the number of possible sequences
    num_samples = df_X.shape[0]
    new_num_samples = num_samples // sequence_length

    # Preparing input data
    X_LSTM = torch.tensor(df_X.values[:new_num_samples * sequence_length], dtype=torch.float32)
    X_LSTM = X_LSTM.view(new_num_samples, sequence_length, input_size).to(device)
    
    # Preparation of target data
    y_LSTM = torch.tensor(df_y.values[:new_num_samples * sequence_length], dtype=torch.long)
    y_LSTM = y_LSTM.view(new_num_samples, sequence_length).to(device)
    
    return X_LSTM, y_LSTM

input_size = df_train_resampled_X.shape[1]  
sequence_length = 1

# Prepare training, validation and test data
X_train_LSTM, y_train_LSTM = prepare_lstm_data(df_train_resampled_X, df_train_resampled_y, sequence_length, input_size, device)
X_val_LSTM, y_val_LSTM = prepare_lstm_data(df_val_resampled_X, df_val_resampled_y, sequence_length, input_size, device)
X_test_LSTM, y_test_LSTM = prepare_lstm_data(df_test_X, df_test_y, sequence_length, input_size, device)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    """
    Train a model on training data and validate its performance on validation data.

    Args:
        model (nn.Module): Model to be trained.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        criterion (nn.Module): Loss function used.
        optimizer (torch.optim.Optimizer): Optimizer used.
        num_epochs (int): Number of epochs.
        device (torch.device): The system on which the model is trained.

    Returns:
        None.
    """
    for epoch in range(num_epochs):
        # Put the model in training mode
        model.train()

        # Initialise the training loss for each epoch
        train_loss = 0.0

        # Itère on training data
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Reset optimizer gradients
            optimizer.zero_grad()
            
            # Forward propagation of the model
            outputs = model(inputs)
            
            # Calculate the loss
            loss = criterion(outputs, targets.view(-1))
            
            # Update model weights
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Calculates the average training loss for the epoch
        avg_train_loss = train_loss / len(train_loader)

        # Validate the model on validation data
        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, device)
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}')
    

def evaluate_model(model, data_loader, criterion, device, print_metrics=False):
    """
    Evaluate a model on validation/test data.

    Args:
        model (nn.Module): Model to be evaluated..
        data_loader (DataLoader): DataLoader for validation/test data.
        criterion (nn.Module): Loss function used.
        optimizer (torch.optim.Optimizer): Optimizer used.
        device (torch.device): The system on which the model is trained.
        print_metrics (bool): If True, displays the confusion matrix and the classification report.

    Returns:
        Tuple: The average loss and global accuracy of the model.
    """
    
    # Put the model in evaluation mode
    model.eval()

    # Set total loss to zero
    total_loss = 0.0

    # Lists to store all true and predicted values
    all_targets = []
    all_preds = []
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward propagation of the model to obtain predictions
            outputs = model(inputs)

            # Calculate the loss
            loss = criterion(outputs, targets.view(-1))

            total_loss += loss.item()

            # Add targets and predictions to their lists 
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
    
    # Calculate the average loss 
    avg_loss = total_loss / len(data_loader)
    
    # Calculate the global accuracy
    accuracy = sklearn.metrics.accuracy_score(all_targets, all_preds)
    
    if print_metrics:
        # Display the confusion matrix
        print("\nConfusion Matrix:")
        print(sklearn.metrics.confusion_matrix(all_targets, all_preds))
        # Display the classification report
        print("\nClassification Report:")
        print(sklearn.metrics.classification_report(all_targets, all_preds))
        # Display the global accuracy 
        print(f"\nGlobal Accuracy: {accuracy:.4f}")
    
    return avg_loss, accuracy

# Hyperparameters
input_size = df_train_resampled_X.shape[1] # Fixed size of the input layer
output_size = 3  # Fixed size of the output layer
dropout = 0.5 # Fixed dropout rate
batch_size = 1024 # Fixed batch size

# Use the same range of hyperparameter values used for each grid search performed to ensure reproducibility of results
# Grid search for the LSTM model has been split into 2 parts to avoid having a long execution time for this optimisation algorithm

# Uncomment one of the 2 configurations to carry out the grid search

# Grid search 1
'''
num_epochs_list = [20, 50]
num_layers_list = [2, 5, 10]
learning_rate_list = [0.001, 0.0001]
hidden_size_list = [50, 100]
'''
# Grid search 2
num_epochs_list = [20, 50]
num_layers_list = [2, 5, 10]
learning_rate_list = [0.001, 0.0001]
hidden_size_list = [150]



# Generate all possible combinations of hyperparameters tested during the grid search
all_combinations = list(itertools.product(num_epochs_list, num_layers_list, learning_rate_list, hidden_size_list))

best_accuracy = 0.0
best_params = {}

print(f"{timing}h")

# Grid search
for num_epochs, num_layers, learning_rate, hidden_size in all_combinations:   
  print(f"Testing configuration: batch_size={batch_size}, num_epochs={num_epochs}, num_layers={num_layers}, learning_rate={learning_rate}, dropout={dropout}, hidden_size={hidden_size}")

  # Setup
  model = LSTM_Model(input_size, hidden_size, output_size, num_layers, dropout).to(device)
  criterion = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

  # Data loaders
  train_dataset_LSTM = torch.utils.data.TensorDataset(X_train_LSTM, y_train_LSTM)
  train_loader_LSTM = torch.utils.data.DataLoader(train_dataset_LSTM, batch_size=batch_size, shuffle=True)
  val_dataset_LSTM = torch.utils.data.TensorDataset(X_val_LSTM, y_val_LSTM)
  val_loader_LSTM = torch.utils.data.DataLoader(val_dataset_LSTM, batch_size=batch_size)
  test_dataset_LSTM = torch.utils.data.TensorDataset(X_test_LSTM, y_test_LSTM)
  test_loader_LSTM = torch.utils.data.DataLoader(test_dataset_LSTM, batch_size=batch_size)

  # Train the model
  start_time = time.time()
  train_model(model, train_loader_LSTM, val_loader_LSTM, criterion, optimizer, num_epochs, device)
  end_time = time.time()
  total_train_time = end_time - start_time
  print(f'Total training time: {total_train_time:.4f}s')

  # Evaluate on validation set
  start_time = time.time()
  test_loss, test_accuracy = evaluate_model(model, test_loader_LSTM, criterion, device, print_metrics=True)
  end_time = time.time()
  total_evaluation_time = end_time - start_time
  print(f'Total evaluation time: {total_evaluation_time:.4f}s')
  print(f'Test Loss: {test_loss:.4f}')

  # Updating the best results 
  if test_accuracy > best_accuracy:
      best_accuracy = test_accuracy
      best_params = {
          "batch_size": batch_size,
          "num_epochs": num_epochs,
          "num_layers": num_layers,
          "learning_rate": learning_rate,
          "dropout": dropout,
          "hidden_size": hidden_size
      }

print(f"Best configuration: {best_params} with accuracy: {best_accuracy:.4f}")

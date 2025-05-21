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
# MultiLayer Perceptron (MLP)

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

# Convert training, validation and test data into PyTorch tensors and transfer them to the GPU
X_train_MLP = torch.tensor(df_train_resampled_X.values, dtype=torch.float32).to(device)
X_val_MLP = torch.tensor(df_val_resampled_X.values, dtype=torch.float32).to(device)
X_test_MLP = torch.tensor(df_test_X.values, dtype=torch.float32).to(device)
y_train_MLP = torch.tensor(df_train_resampled_y.values, dtype=torch.long).view(-1).to(device)
y_val_MLP = torch.tensor(df_val_resampled_y.values, dtype=torch.long).view(-1).to(device)
y_test_MLP = torch.tensor(df_test_y.values, dtype=torch.long).view(-1).to(device)

class MLP_Model(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout=0.5):
        """
        Initialises the MLP model.

        Args:
            input_size (int): Size of model inputs.
            hidden_sizes (list of int): List of hidden layer sizes.
            output_size (int): Size of model outputs.
            dropout (float, optional): Dropout rate used.
        
        Returns:
            None.
        """
        super(MLP_Model, self).__init__()
        # List for storing layers
        layers = []
        # Size of the previous layer
        # It is initialised to the inputs size
        prev_size = input_size
        
        # Creation of hidden layers with a linear layer, a LeakyRelu activation function and a dropout layer
        for hidden_size in hidden_sizes:
            layers.append(torch.nn.Linear(prev_size, hidden_size))
            layers.append(torch.nn.LeakyReLU())
            layers.append(torch.nn.Dropout(dropout))
            # Update the previous layer size
            prev_size = hidden_size
        
        # Output layer
        layers.append(torch.nn.Linear(prev_size, output_size))
        
        self.mlp = torch.nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Defines the forward propagation function for the MLP model.

        Args:
            x (Tensor): Model inputs.

        Returns:
            Tensor: Model output after the foward propagation.
        """
        return self.mlp(x)

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
num_layers = 3 # Fixed number of layers

# Use the same range of hyperparameter values used for each grid search performed to ensure reproducibility of results
# Grid search for the MLP model has been split into 4 parts to avoid having a long execution time for this optimisation algorithm

# Uncomment one of the 4 configurations to carry out the grid search

# Grid search 1
'''
hidden_sizes_list = [[128, 64, 32], [256, 128, 64], [512, 256, 128]]  
num_epochs_list = [10, 20, 30]
learning_rate_list = [0.001, 0.0001]
batch_size = 1024
'''
# Grid search 2
'''
hidden_sizes_list = [[128, 64, 32], [256, 128, 64], [512, 256, 128]]  
num_epochs_list = [10, 20, 30]
learning_rate_list = [0.001, 0.0001]
batch_size = 512
'''
# Grid search 3
'''
hidden_sizes_list = [[128, 64, 32], [256, 128, 64], [512, 256, 128]]  
num_epochs_list = [50]
learning_rate_list = [0.001, 0.0001]
batch_size = 512
'''
# Grid search 4
hidden_sizes_list = [[128, 64, 32], [256, 128, 64], [512, 256, 128]]  
num_epochs_list = [50]
learning_rate_list = [0.001, 0.0001]
batch_size = 1024


# Generate all possible combinations of hyperparameters tested during the grid search
all_combinations = list(itertools.product(num_epochs_list, learning_rate_list, hidden_sizes_list))

best_accuracy = 0.0
best_params = {}

print(f"{timing}h")

# Grid search
for num_epochs, learning_rate, hidden_sizes in all_combinations:   
  print(f"Testing configuration: batch_size={batch_size}, num_epochs={num_epochs}, learning_rate={learning_rate}, dropout={dropout}, hidden_size={hidden_sizes}")
    
  # Setup
  model = MLP_Model(input_size, hidden_sizes, output_size, dropout).to(device)
  criterion = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
  
  # Data loaders
  train_dataset_MLP = torch.utils.data.TensorDataset(X_train_MLP, y_train_MLP)
  train_loader_MLP = torch.utils.data.DataLoader(train_dataset_MLP, batch_size=batch_size, shuffle=True)
  val_dataset_MLP = torch.utils.data.TensorDataset(X_val_MLP, y_val_MLP)
  val_loader_MLP = torch.utils.data.DataLoader(val_dataset_MLP, batch_size=batch_size)
  test_dataset_MLP = torch.utils.data.TensorDataset(X_test_MLP, y_test_MLP)
  test_loader_MLP = torch.utils.data.DataLoader(test_dataset_MLP, batch_size=batch_size)

  # Train the model
  start_time = time.time()
  train_model(model, train_loader_MLP, val_loader_MLP, criterion, optimizer, num_epochs, device)
  end_time = time.time()
  total_train_time = end_time - start_time
  print(f'Total training time: {total_train_time:.4f}s')

  # Evaluate on validation set
  start_time = time.time()
  test_loss, test_accuracy = evaluate_model(model, test_loader_MLP, criterion, device, print_metrics=True)
  end_time = time.time()
  total_evaluation_time = end_time - start_time
  print(f'Total evaluation time: {total_evaluation_time:.4f}s')
  print(f'Test Loss: {test_loss:.4f}')

  # Updating the best results 
  if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        best_params = {
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "hidden_sizes": hidden_sizes,
            "dropout": dropout,
            "batch_size": batch_size
        }

print(f"Best configuration: {best_params} with accuracy: {best_accuracy:.4f}")

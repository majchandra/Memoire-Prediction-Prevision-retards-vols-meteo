# Importing Libraries

import subprocess

# Ensure all necessary libraries are installed
libraries = ["pandas", "torch", "scikit-learn", "numpy","torchdiffeq"]
for lib in libraries:
    subprocess.run(["pip", "install", lib], check=True)

import time
import pandas as pd
import numpy as np
import torch
import sklearn
import torchdiffeq
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint_adjoint as odeint

# Fixed random seed for reproducibility of results
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print(RANDOM_SEED)


# Implementation of Deep Learning Model
# Liquid Neural Network (LNN)

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
X_train_LNN = torch.tensor(df_train_resampled_X.values, dtype=torch.float32).to(device)
X_val_LNN = torch.tensor(df_val_resampled_X.values, dtype=torch.float32).to(device)
X_test_LNN = torch.tensor(df_test_X.values, dtype=torch.float32).to(device)
y_train_LNN = torch.tensor(df_train_resampled_y.values, dtype=torch.long).view(-1).to(device)
y_val_LNN = torch.tensor(df_val_resampled_y.values, dtype=torch.long).view(-1).to(device)
y_test_LNN = torch.tensor(df_test_y.values, dtype=torch.long).view(-1).to(device)

class LiquidODEFunc(nn.Module):
    def __init__(self, liquid_size):
        """
        Initialises the ODE function for the LNN model.

        Args:
            liquid_size (int): Size of the liquid layer.

        Returns: 
            None.
        """
        super(LiquidODEFunc, self).__init__()

        # Sequential neural network having two linear layers and a Tanh activation function
        self.NN = nn.Sequential(
            nn.Linear(liquid_size, liquid_size),
            nn.Tanh(),
            nn.Linear(liquid_size, liquid_size)
        )
        
    def forward(self, time, input_x):
        """
        Specifies the forward propagation function for computing the ODE.

        Args:
            time (Tensor): Instants of time.
            input_x (Tensor): Inputs to the ODE function.

        Returns:
            Tensor: Model Output after the forward propagation.
        """
        return self.NN(input_x)

class LiquidNeuralNetwork(nn.Module):
    def __init__(self, input_size, liquid_size, output_size, solver='dopri5'):
        """
        Initialises the LNN model.

        Args:
            input_size (int): Size of model inputs.
            liquid_size (int): Size of the liquid layer.
            output_size (int): Size of model outputs.
            solver (str): Solver method for ODEs.
        
        Returns:
            None.
        """
        super(LiquidNeuralNetwork, self).__init__()
        self.liquid_size = liquid_size
        # Input Layer
        self.input_layer = nn.Linear(input_size, liquid_size) 
        # Dropout for regularisation
        self.dropout = nn.Dropout(0.5) # Put it in comments if it is not used
        self.ode_func = LiquidODEFunc(liquid_size)
        # Output Layer
        self.output_layer = nn.Linear(liquid_size, output_size)
        self.solver = solver

    def forward(self, x, time_span):
        """
        Defines the forward propagation function for the LNN model.

        Args:
            x (Tensor): Model inputs.
            time_span (tuple): Time span for solving the ODE.

        Returns:
            Tensor: Network output after propagation of ODE states and the output layer.
        """
        # Input propagation through the input layer
        x = self.input_layer(x)
        # Dropout application
        x = self.dropout(x) # Put it in comments if it is not used
        t = torch.linspace(time_span[0], time_span[1], 10).to(x.device)
        # ODE resolution
        ode_output = torchdiffeq.odeint(self.ode_func, x, t, method=self.solver)
        # Take the last state
        x = ode_output[-1]  
        # Propagation of the last state through the output layer
        return self.output_layer(x)


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
            outputs = model(inputs, (0, 1))
            
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
            outputs = model(inputs, (0, 1))

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
solver = 'dopri5' # Fixed value for the solver

# The values of the hyperparameters below can be changed to test other configurations
hidden_size = 350 # Set the size of the liquid layer
num_epochs = 50 # Set the number of epochs
batch_size = 1024 # Set the batch size
learning_rate = 0.0001 # Set the learning rate value


print(f"{timing}h")
print(f"hidden_size = {hidden_size}")
print(f"epochs = {num_epochs}")
print(f"batch_size = {batch_size}")
print(f"learning_rate = {learning_rate}")
print(f"dropout = 0.5") # Put it in comments if Dropout is not used

# Setup
model = LiquidNeuralNetwork(input_size, hidden_size, output_size, solver).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Data loaders
train_data_LNN = torch.utils.data.TensorDataset(X_train_LNN, y_train_LNN)
train_loader_LNN = torch.utils.data.DataLoader(train_data_LNN, batch_size=batch_size, shuffle=True)
val_data_LNN = torch.utils.data.TensorDataset(X_val_LNN, y_val_LNN)
val_loader_LNN = torch.utils.data.DataLoader(val_data_LNN, batch_size=batch_size)
test_data_LNN = torch.utils.data.TensorDataset(X_test_LNN, y_test_LNN)
test_loader_LNN = torch.utils.data.DataLoader(test_data_LNN, batch_size=batch_size)

# Train the model
start_time = time.time()
train_model(model, train_loader_LNN, val_loader_LNN, criterion, optimizer, num_epochs, device)
end_time = time.time()
total_train_time = end_time - start_time
print(f'Total training time: {total_train_time:.4f}s')

# Evaluate on test set
start_time = time.time()
test_loss, test_acc = evaluate_model(model,  test_loader_LNN, criterion, device, print_metrics=True)
end_time = time.time()
total_evaluation_time = end_time - start_time
print(f'Total evaluation time: {total_evaluation_time:.4f}s')
print(f'Test Loss: {test_loss:.4f}')



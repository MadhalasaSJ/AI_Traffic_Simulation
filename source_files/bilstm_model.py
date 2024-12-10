import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Define BiLSTM Model
class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, light_output_size, volume_output_size):
        super(BiLSTMModel, self).__init__()
        
        # Bi-directional LSTM layer
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, 
                            num_layers=num_layers, batch_first=True, bidirectional=True)
        
        # Fully connected layers for outputs
        self.fc_light = nn.Linear(hidden_size * 2, light_output_size)  # Traffic light prediction (Binary)
        self.fc_volume = nn.Linear(hidden_size * 2, volume_output_size)  # Traffic volume prediction (Regression)

    def forward(self, x):
        lstm_out, (hn, cn) = self.lstm(x)  # Output from LSTM
        
        # Take the last time step's output
        lstm_out = lstm_out[:, -1, :]
        
        # Pass through fully connected layers
        light_output = torch.sigmoid(self.fc_light(lstm_out))  # Sigmoid activation for binary classification
        volume_output = torch.relu(self.fc_volume(lstm_out))  # ReLU for continuous non-negative volume output
        
        return light_output, volume_output

# Generate Dummy Dataset
def create_dummy_dataset(num_samples, sequence_length, input_size):
    X = torch.randn(num_samples, sequence_length, input_size)  # Random input features
    y_light = torch.randint(0, 2, (num_samples, 1)).float()  # Binary output for traffic light (0 or 1)
    y_volume = torch.abs(torch.randn(num_samples, 1))  # Positive continuous output for traffic volume
    return TensorDataset(X, y_light, y_volume)

# Train the Model
def train_model(model, dataloader, criterion_light, criterion_volume, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for inputs, light_labels, volume_labels in dataloader:
            optimizer.zero_grad()
            
            # Forward pass
            light_output, volume_output = model(inputs)
            
            # Calculate losses
            loss_light = criterion_light(light_output, light_labels)
            loss_volume = criterion_volume(volume_output, volume_labels)
            loss = loss_light + loss_volume
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss:.4f}")

# Save the Model
def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

# Main Execution
if __name__ == "__main__":
    # Hyperparameters
    input_size = 256  # Number of input features (adjust as needed)
    hidden_size = 128  # Number of hidden units in LSTM
    num_layers = 2  # Number of layers in the LSTM
    light_output_size = 1  # Output size for traffic light (binary classification)
    volume_output_size = 1  # Output size for traffic volume (regression)
    sequence_length = 10  # Length of input sequences
    num_samples = 1000  # Number of samples in the dataset
    batch_size = 32  # Batch size for training
    num_epochs = 50  # Number of epochs to train
    learning_rate = 0.001  # Learning rate for optimizer

    # Initialize the BiLSTM model
    model = BiLSTMModel(input_size=input_size, hidden_size=hidden_size, 
                        num_layers=num_layers, light_output_size=light_output_size, 
                        volume_output_size=volume_output_size)

    # Create dataset and dataloader
    dataset = create_dummy_dataset(num_samples, sequence_length, input_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Loss functions and optimizer
    criterion_light = nn.BCELoss()  # Binary Cross Entropy for traffic light control
    criterion_volume = nn.MSELoss()  # Mean Squared Error for traffic volume prediction
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train_model(model, dataloader, criterion_light, criterion_volume, optimizer, num_epochs=num_epochs)

    # Save the trained model
    model_save_path = "bilstm_traffic_model.pth"
    save_model(model, model_save_path)

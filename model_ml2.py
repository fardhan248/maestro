import torch
import torch.nn as nn
import torch.optim as optim

class CNNModelBN(nn.Module):
    def __init__(self, steps, features, outputs):
        super(CNNModelBN, self).__init__()
        
        # Adjust in_channels to 72 based on the provided weight shape
        self.conv1 = nn.Conv1d(in_channels=features, out_channels=64, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(64)
        
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=256, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(256)
        
        self.flatten = nn.Flatten()
        
        # Adjust the input dimension of fc1 to 256 based on the provided weight shape
        self.fc1 = nn.Linear(256 * steps, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.2)
        
        self.fc2 = nn.Linear(256, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(256, 64)
        self.bn5 = nn.BatchNorm1d(64)
        
        # Adjust fc4 to match the output dimensions of your provided weight shape
        self.fc4 = nn.Linear(64, outputs)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.flatten(x)
        
        x = self.fc1(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        x = self.bn5(x)
        x = self.relu(x)
        
        x = self.fc4(x)
        return x
    
def create_model_ml2(input_size, output_size):
    steps = 1
    model = CNNModelBN(steps=steps, features=input_size, outputs=output_size)
    return model

def load_model_ml2(input_size, output_size, path):
    model = create_model_ml2(input_size, output_size)
    trained_model_ml2 = path
    # Load the saved weights into the model
    model.load_state_dict(torch.load(trained_model_ml2))
    model.eval()
    print("Successfully  loaded ml2")
    return model
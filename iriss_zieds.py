from google.colab import files
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

"""#2. File Upload"""

# Invoke file upload dialog and select Iris_train.csv and Iris_validate.csv files
uploaded = files.upload()

# Check if the files are in the Google Colab disk (use Bash commands with '!')
!ls

"""#3. Data Preparation"""

# Read data into pandas dataframe objects
# training_data will be used to train the neural network
training_data = pd.read_csv("Iris_train.csv")
# validation_data will be used to check the accuracy of the neural network
validation_data = pd.read_csv("Iris_validate.csv")


# Create a class "IrisDataset" that extracts data from the provided DataFrame
class IrisDataset(Dataset):
    def __init__(self, dataframe):
        # Extract all attributes except the last one as features
        self.features = dataframe.iloc[:, :-1].values
        # Extract labels and transform them using LabelEncoder
        self.labels = LabelEncoder().fit_transform(dataframe.iloc[:, -1].values)

    # Define a method that returns the number of features (dataset length)
    def __len__(self):
        return len(self.features)

    # Define a method that returns the features and label at a specific index
    def __getitem__(self, idx):
        return torch.Tensor(self.features[idx]).float(), torch.tensor(int(self.labels[idx]), dtype=torch.long)

# Create training dataset and data loader using PyTorch DataLoader
train_dataset = IrisDataset(training_data)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Create validation dataset and data loader using PyTorch DataLoader
val_dataset = IrisDataset(validation_data)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

"""#4. Defining the Neural Network"""


class IrisClassifier(nn.Module):
    def __init__(self):
        # Initialize IrisClassifier which is a nn.Module
        super(IrisClassifier, self).__init__()
        # Define two linear layers
        self.fc1 = nn.Linear(4, 16)  # Input size of first layer: 4, output size: 16
        self.fc2 = nn.Linear(16, 3)  # Input size of second layer: 16, output size: 3

    # Forward method describing data flow through the model architecture
    def forward(self, x):
        # Use ReLU activation function on the first layer
        x = F.relu(self.fc1(x))

        # Use linear layer a second time
        x = self.fc2(x)

        # Use log softmax function on the last layer
        return F.log_softmax(x, dim=1)

"""#5. Training the Neural Network"""

# Function to train the model using data from train_loader
def train(model, train_loader, optimizer, criterion, epochs=10):
    for epoch in range(epochs):
        model.train()  # Set model to training mode
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()   # Clear previous gradients
            output = model(batch_x)

            # Convert labels to long tensor
            batch_y = batch_y.long()

            loss = criterion(output, batch_y)
            loss.backward()  # Backpropagate gradients
            optimizer.step() # Perform optimization

# Function to evaluate the model's accuracy using data from val_loader
def evaluate(model, val_loader):
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            output = model(batch_x)
            _, predicted = torch.max(output, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
    accuracy = correct / total * 100
    return accuracy

# Initialize IrisClassifier, optimizer, and criterion
iris_classifier = IrisClassifier()
optimizer = torch.optim.Adam(iris_classifier.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Train the model using the previously defined train function
train(iris_classifier, train_loader, optimizer, criterion, epochs=50)

# Enhanced train function that also evaluates the model on validation data
def train(model, train_loader, val_loader, optimizer, criterion, epochs=10):
    train_losses = []  # To store training losses for visualization
    val_accuracies = []  # To store validation accuracy for visualization

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        val_accuracy = evaluate(model, val_loader)
        val_accuracies.append(val_accuracy)

        # Print training progress
        print(f'Epoch {epoch + 1}/{epochs} -> Avg. Training Loss: {avg_train_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

    return train_losses, val_accuracies

# Import library to plot training losses and accuracy
import matplotlib.pyplot as plt

# Function to visualize training losses and validation accuracy
def plot_results(train_losses, val_accuracies):
    epochs = range(1, len(train_losses) + 1)

    # Plot losses
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot training accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, label='Validation Accuracy', color='orange')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Reinitialize the model, optimizer, and criterion
iris_classifier = IrisClassifier()
optimizer = torch.optim.Adam(iris_classifier.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Results of using the train function
train_losses, val_accuracies = train(iris_classifier, train_loader, val_loader, optimizer, criterion, epochs=50)

# Visualize training loss and accuracy
plot_results(train_losses, val_accuracies)

"""# 6. Testing the Neural Network."""

# Modify function to not print losses and accuracy
def train(model, train_loader, optimizer, criterion, epochs=10):
    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

# Test the neural network accuracy with validation dataset without using training losses and accuracy

def print_test_results(results):
    for result in results:
        print(result)

def test(model, val_loader, class_names):
    model.eval()  # Set model to evaluation mode, do not use gradients

    # Initialize variables for accuracy calculation
    correct = 0
    total = 0

    # Create an empty list to store results
    results = []

    with torch.no_grad():
        for batch_x, batch_y in val_loader:

            # Get model predictions
            output = model(batch_x)

            # Determine the class with the maximum value for each input sample
            _, predicted = torch.max(output, 1)

            for i in range(len(predicted)):
                result_str = f"Classifier result: {class_names[predicted[i]]}; Real species: {class_names[batch_y[i]]}, "

                # Check if the prediction matches the true class
                if predicted[i] == batch_y[i]:
                    result_str += "correct"
                    correct += 1
                else:
                    result_str += "incorrect"

                # Add result to the list
                results.append(result_str)
                total += 1

    # Calculate accuracy and add it to the results list
    accuracy = correct / total * 100
    results.append(f'Accuracy: {accuracy:.2f}% ({correct}/{total})')
    return results

# Training and evaluation
iris_classifier = IrisClassifier()
optimizer = torch.optim.Adam(iris_classifier.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

train(iris_classifier, train_loader, optimizer, criterion, epochs=50)

# Print results in the desired format
class_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
results = test(

iris_classifier, val_loader, class_names)
print_test_results(results)
```
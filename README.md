# Smart-Diving-Suits

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# Sample Sensor Data (Pressure, Oxygen, Heart Rate, Temperature, Depth, Risk)
data = {
    'pressure': [2.3, 5.1, 7.8, 3.4, 6.2, 4.8, 9.0, 1.5, 8.3, 2.9],
    'oxygen': [90, 75, 60, 80, 50, 65, 30, 95, 40, 85],
    'heart_rate': [70, 85, 110, 95, 130, 120, 140, 65, 125, 80],
    'temperature': [36.5, 37.0, 38.2, 36.8, 39.0, 37.5, 38.5, 36.3, 39.2, 37.1],
    'depth': [10, 20, 30, 15, 25, 18, 35, 12, 40, 22],
    'decompression_risk': [0, 0, 1, 0, 1, 1, 1, 0, 1, 0]
}
df = pd.DataFrame(data)

# Data Preprocessing
scaler = StandardScaler()
X = scaler.fit_transform(df.drop(columns=['decompression_risk']))
y = df['decompression_risk'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

# AI-Driven Smart Suit Model
class SmartSuitNN(nn.Module):
    def __init__(self):
        super(SmartSuitNN, self).__init__()
        self.fc1 = nn.Linear(5, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

  def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

model = SmartSuitNN()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop (20-30 Epochs)
num_epochs = 30
loss_history = []
accuracy_history = []
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

  predictions = (outputs > 0.5).float()
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

  avg_loss = running_loss / len(train_loader)
    accuracy = correct / total
    loss_history.append(avg_loss)
    accuracy_history.append(accuracy)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')

# Plot Loss and Accuracy History
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs+1), loss_history, label='Training Loss', color='blue')
plt.plot(range(1, num_epochs+1), accuracy_history, label='Training Accuracy', color='green')
plt.xlabel('Epochs')
plt.ylabel('Loss/Accuracy')
plt.title('Model Training Performance Over Epochs')
plt.legend()
plt.show()

# Model Evaluation
model.eval()
y_pred = []
with torch.no_grad():
    for inputs, _ in test_loader:
        outputs = model(inputs)
        y_pred.append(outputs.squeeze().numpy())

y_pred = np.concatenate(y_pred) > 0.5
accuracy = (y_pred == y_test).mean()
print(f'Accuracy: {accuracy:.4f}')

# Plot Sensor Data with Risk Indicator
plt.figure(figsize=(10, 6))
time_points = range(len(df))
plt.plot(time_points, df['pressure'], label='Pressure', color='blue')
plt.plot(time_points, df['oxygen'], label='Oxygen', color='green')
plt.plot(time_points, df['heart_rate'], label='Heart Rate', color='orange')
plt.plot(time_points, df['temperature'], label='Temperature', color='purple')
plt.fill_between(time_points, 0, 150, where=df['decompression_risk'] == 1, color='red', alpha=0.3, label='Risk Zone')
plt.xlabel('Time Index')
plt.ylabel('Sensor Readings')
plt.title('Sensor Readings with Risk Indicators')
plt.legend()
plt.show()

# Depth vs Risk Graph
plt.figure(figsize=(10, 5))
plt.scatter(df['depth'], df['decompression_risk'], color='red', label='Risk Level')
plt.axvline(x=25, color='blue', linestyle='--', label='Increased Vulnerability Depth')
plt.xlabel('Depth (meters)')
plt.ylabel('Decompression Risk')
plt.title('Depth vs Decompression Risk')
plt.legend()
plt.show()

# Panic Meter based on Heart Rate and Oxygen Levels
def panic_meter(hr, oxygen):
    if hr > 120 and oxygen < 50:
        return 'Panic Mode Activated!'
    return 'Normal'

# Example Usage Cases
example_cases = [
    {'pressure': 3, 'oxygen': 85, 'heart_rate': 75, 'temperature': 36.7, 'depth': 10},  # Safe
    {'pressure': 7, 'oxygen': 45, 'heart_rate': 125, 'temperature': 38.5, 'depth': 30}, # Unsafe
]

for case in example_cases:
    risk_status = panic_meter(case['heart_rate'], case['oxygen'])
    print(f"Dive Status: {risk_status}")

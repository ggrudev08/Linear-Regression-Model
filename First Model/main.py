import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from model import LinearRegressionModel

x = np.array([i for i in range(100)])
x = x.reshape(-1, 1)
y = 46 + 2 * x.flatten()

x_mean, x_std = x.mean(), x.std()
x_normalized = (x - x_mean) /x_std
x_tensor = torch.tensor(x_normalized, dtype=torch.float32)

y_mean, y_std = y.mean(), y.std()
y_normalized = (y - y_mean) / y_std
y_tensor = torch.tensor(y_normalized, dtype=torch.float32)

in_features = 1
out_features = 1
model = LinearRegressionModel(in_features, out_features)

CrossEntropyLoss = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

num_epochs = 30
model.train()

for epoch in range(num_epochs):
    outputs = model(x_tensor)
    loss = CrossEntropyLoss(outputs, y_tensor)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f'Epoch [{epoch +1}/{num_epochs}], Loss: {loss.item():.5f}')

x1 = 121
x1_norm = (x1 - x_mean) / x_std

x1_tensor = torch.tensor(x1_norm, dtype=torch.float32).view(1, -1)

model.eval()

with torch.no_grad():
    prediction_normalized = model(x1_tensor)

prediction = prediction_normalized.item() * y_std + y_mean

print(f"Predicted value for x = {x1}: {prediction}")







import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import json
import numpy as np

# Cargar el dataset desde el CSV (1,000,000 filas)
df = pd.read_csv("datos_monte_carlo_1.csv")

# Definir las features y el target
features = ['OXIDO_COSTO', 'CHATARRA_COSTO', 'SULFURICO_COSTO', 'FERROSO_PRECIO', 'FERRICO_PRECIO', 
            'cantidad_oxido', 'cantidad_sulfurico', 'cantidad_chatarra', 'coeficiente_sulfurico', 
            'coeficiente_chatarra', 'coeficiente_ferrico']
X = df[features].values
y = df['Utilidad'].values.reshape(-1, 1)  # Reshape para target

# Dividir en train/test (80/20) para validación
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalización (fit en train, transform en train y test)
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)  # No .numpy(), ya es array

# Convertir a tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)

# Definir la NN simple (11 inputs -> 64 hidden -> ReLU -> 32 hidden -> ReLU -> 1 output)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(11, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = Net()

# Loss y optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Entrenamiento (e.g., 50 epochs, batch size 1024)
num_epochs = 200
batch_size = 2048
train_loader = torch.utils.data.DataLoader(list(zip(X_train_tensor, y_train_tensor)), batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    model.train()
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

# Evaluación en test (R2 score)
model.eval()
with torch.no_grad():
    y_pred_scaled = model(X_test_tensor).numpy()  # Convert to numpy
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_test_original = scaler_y.inverse_transform(y_test_scaled)  # No .numpy(), ya es array
    r2 = r2_score(y_test_original, y_pred)
    print(f"R2 en test: {r2:.4f}")

# Guardar pesos en JSON (dict de capas, serializable)
state_dict = model.state_dict()
state_dict_serializable = {k: v.tolist() for k, v in state_dict.items()}
with open('nn_weights.json', 'w') as f:
    json.dump(state_dict_serializable, f)
print("Pesos de la NN guardados en 'nn_weights.json'.")

# Guardar scalers para uso posterior (en JSON también)
scaler_X_dict = {'mean': scaler_X.mean_.tolist(), 'scale': scaler_X.scale_.tolist()}
scaler_y_dict = {'mean': scaler_y.mean_.tolist(), 'scale': scaler_y.scale_.tolist()}
with open('scalers.json', 'w') as f:
    json.dump({'scaler_X': scaler_X_dict, 'scaler_y': scaler_y_dict}, f)
print("Scalers guardados en 'scalers.json'.")
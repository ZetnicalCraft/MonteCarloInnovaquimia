import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# Definir la misma estructura de NN que en entrenamiento
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

# Cargar pesos desde JSON y asignar a la NN
with open('nn_weights.json', 'r') as f:
    state_dict_serializable = json.load(f)

state_dict = {k: torch.tensor(v) for k, v in state_dict_serializable.items()}
model = Net()
model.load_state_dict(state_dict)
model.eval()  # Modo inferencia

# Cargar scalers desde JSON
with open('scalers.json', 'r') as f:
    scalers = json.load(f)

scaler_X_mean = np.array(scalers['scaler_X']['mean'])
scaler_X_scale = np.array(scalers['scaler_X']['scale'])
scaler_y_mean = np.array(scalers['scaler_y']['mean'])
scaler_y_scale = np.array(scalers['scaler_y']['scale'])

# Función para normalizar manual (sin sklearn fit)
def scale_X(input_array):
    return (input_array - scaler_X_mean) / scaler_X_scale

def inverse_scale_y(output_scaled):
    return output_scaled * scaler_y_scale + scaler_y_mean

# Ejemplo: Predecir con un nuevo input (11 features)
nuevo_input = np.array([0.555,0.484,0.555,0.542,0.875,7667,9046,2670,0.7324224935115646,0.9711275351504549,0.6208053224909241])  # Tus valores
nuevo_input_scaled = scale_X(nuevo_input)
nuevo_tensor = torch.tensor(nuevo_input_scaled, dtype=torch.float32).unsqueeze(0)  # Batch de 1

with torch.no_grad():
    pred_scaled = model(nuevo_tensor).numpy()
    pred = inverse_scale_y(pred_scaled)[0][0]

print("Utilidad predicha:", pred)
import json
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
from sklearn.preprocessing import StandardScaler
from utilidad_predictor import UtilidadPredictor  # Tu clase

# Cargar dataset para background (usa subset de 200 samples de tu MC para computo)
df = pd.read_csv("datos_monte_carlo_1.csv")
background_df = df.sample(200)  # 200 para SHAP/PDP
background_features = background_df[['OXIDO_COSTO', 'CHATARRA_COSTO', 'SULFURICO_COSTO', 'FERROSO_PRECIO', 'FERRICO_PRECIO', 
                                     'cantidad_oxido', 'cantidad_sulfurico', 'cantidad_chatarra', 'coeficiente_sulfurico', 
                                     'coeficiente_chatarra', 'coeficiente_ferrico']].values

# Ejemplo escenario (exógenas fijas + coefs óptimos de evolutivos, ajusta si quieres)
exogenas = np.array([0.895,0.329,0.462,0.204,0.741,951,2598,373])  # 8 exógenas
opt_coefs = np.array([0.69794728, 0.99998902, 0.97534398])  # De evolutivos
escenario = np.concatenate([exogenas, opt_coefs])  # 11 features

# Cargar predictor (tipo=5 para NN)
predictor = UtilidadPredictor(5)

# Función wrapper para PDP/SHAP (usa predictor.obtener_utilidad como model)
def model_predict(inputs):
    utils = np.zeros(len(inputs))
    for i, input_row in enumerate(inputs):
        utils[i] = predictor.obtener_utilidad(input_row)
    return utils

# Dummy estimator para PDP (fix para sklearn validation)
class DummyEstimator:
    def __init__(self):
        self._estimator_type = 'regressor'  # Indica tipo regressor

    def fit(self, X, y=None):
        self.fitted_ = True  # Para check_is_fitted
        return self

    def predict(self, X):
        return model_predict(X)

# Instancia y "fit" el estimator (dummy)
est = DummyEstimator()
est.fit(background_features)  # Usa background como data dummy

# Wrapper para incluir escalado en el modelo PyTorch (para SHAP en espacio original)
class ScaledNet(nn.Module):
    def __init__(self, model, mean, scale, y_mean, y_scale):
        super(ScaledNet, self).__init__()
        self.model = model
        self.register_buffer('mean', torch.tensor(mean, dtype=torch.float32))
        self.register_buffer('scale', torch.tensor(scale, dtype=torch.float32))
        self.y_mean = y_mean[0]
        self.y_scale = y_scale[0]

    def forward(self, x):
        x_scaled = (x - self.mean) / self.scale
        pred_scaled = self.model(x_scaled)
        return pred_scaled * self.y_scale + self.y_mean

# Instancia el wrapper
scaled_model = ScaledNet(predictor.model, predictor.scaler_X_mean, predictor.scaler_X_scale, 
                         predictor.scaler_y_mean, predictor.scaler_y_scale)
scaled_model.eval()  # Modo evaluación

# Convierte a tensores (raw features)
background_tensor = torch.tensor(background_features, dtype=torch.float32)
escenario_tensor = torch.tensor(escenario.reshape(1, -1), dtype=torch.float32)

# 1. SHAP: DeepExplainer para el modelo wrapped (usa background raw)
explainer = shap.DeepExplainer(scaled_model, background_tensor)
shap_values = explainer.shap_values(escenario_tensor, check_additivity=False)[0]  # Skip check

# Verificación manual (opcional, descomenta si quieres)
# model_output = scaled_model(escenario_tensor).item()
# expected_value = explainer.expected_value
# shap_sum = np.sum(shap_values)
# print(f"Model output: {model_output}")
# print(f"Expected + SHAP sum: {expected_value + shap_sum}")
# print(f"Difference: {abs(model_output - (expected_value + shap_sum))}")

# Gráficas SHAP
features_names = ['OXIDO_COSTO', 'CHATARRA_COSTO', 'SULFURICO_COSTO', 'FERROSO_PRECIO', 'FERRICO_PRECIO', 
                  'cantidad_oxido', 'cantidad_sulfurico', 'cantidad_chatarra', 'coeficiente_sulfurico', 
                  'coeficiente_chatarra', 'coeficiente_ferrico']

# Beeswarm (global)
plt.figure()
shap.summary_plot(shap_values.reshape(1, -1), escenario.reshape(1, -1), feature_names=features_names, show=False)
plt.savefig('shap_beeswarm.png')
plt.close()

# Bar (ranking importancia)
plt.figure()
shap.summary_plot(shap_values.reshape(1, -1), escenario.reshape(1, -1), feature_names=features_names, plot_type="bar", show=False)
plt.savefig('shap_bar.png')
plt.close()

# Forceplot
plt.figure(figsize=(30, 5))  # Wider for more space
shap.force_plot(explainer.expected_value, shap_values.reshape(1, -1), escenario.reshape(1, -1), feature_names=features_names, matplotlib=True, show=False, text_rotation=45, contribution_threshold=0.01)  # Rotate text, show smaller contributions
plt.savefig('shap_force.png')
plt.close()

# Waterfall (static detailed alternative to force plot)
explanation = shap.Explanation(values=shap_values.reshape(1, -1), base_values=explainer.expected_value, data=escenario.reshape(1, -1), feature_names=features_names)
plt.figure(figsize=(30, 16))  # Vertical format for better readability
shap.plots.waterfall(explanation[0], max_display= 12, show=False)
plt.tight_layout() 
plt.savefig('shap_waterfall.png')
plt.close()

# 2. PDP: Usa la instancia "fitted" est
# Para todas 11 features (1D) - Más espacio entre filas
fig, ax = plt.subplots(figsize=(20, 15))  # Aumenta tamaño para más espacio
PartialDependenceDisplay.from_estimator(est, background_features, range(11), feature_names=features_names, ax=ax)
fig.subplots_adjust(hspace=0.5)  # Más espacio vertical entre subplots (ajusta si necesitas más)
plt.tight_layout()  # Ajuste automático de layout
plt.savefig('pdp_1d_all.png')
plt.close()

# 1D para coefs (zoom)
fig, ax = plt.subplots(figsize=(10, 5))
PartialDependenceDisplay.from_estimator(est, background_features, [8,9,10], feature_names=features_names, ax=ax)
plt.tight_layout()
plt.savefig('pdp_1d_coefs.png')
plt.close()

# 2D para pares de coefs
pairs = [(8,9), (8,10), (9,10)]  # Indices coefs
for i, pair in enumerate(pairs):
    fig, ax = plt.subplots(figsize=(8, 6))
    PartialDependenceDisplay.from_estimator(est, background_features, pair, kind='both', feature_names=features_names, ax=ax)
    plt.tight_layout()
    plt.savefig(f'pdp_2d_pair_{i+1}.png')
    plt.close()

print("Gráficas guardadas en PNG: shap_beeswarm, shap_bar, shap_waterfall (estático con detalles), pdp_1d_all, pdp_1d_coefs, pdp_2d_pair_*.")
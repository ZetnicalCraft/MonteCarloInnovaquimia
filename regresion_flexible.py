import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from scipy.optimize import minimize
import json  # Para guardar la fórmula en JSON
import globales

RESUMEN_MODELO = ""
ECUACION_OBTENIDA = ""

# Cargar el dataset desde el CSV (1,000,000 filas)
df = pd.read_csv(globales.nombre_archivo_datos)

# Definir las features y el target
features = ['OXIDO_COSTO', 'CHATARRA_COSTO', 'SULFURICO_COSTO', 'FERROSO_PRECIO', 'FERRICO_PRECIO', 
            'cantidad_oxido', 'cantidad_sulfurico', 'cantidad_chatarra', 'coeficiente_sulfurico', 
            'coeficiente_chatarra', 'coeficiente_ferrico']
X = df[features]
y = df['Utilidad']

# Generar features polinomiales (cambia degree=1 para lineal, degree=2 para polinomial)
poly = PolynomialFeatures(degree=globales.grado, include_bias=False)  # degree=1 para lineal
X_poly = poly.fit_transform(X)

# Normalización de las features polinomiales (StandardScaler)
scaler = StandardScaler()
X_poly_scaled = scaler.fit_transform(X_poly)

# Entrenamiento de la regresión (usando OLS en las features expandidas)
X_poly_scaled_with_const = sm.add_constant(X_poly_scaled)
model = sm.OLS(y, X_poly_scaled_with_const).fit()

# Imprimir el summary del modelo para análisis
print(model.summary())

RESUMEN_MODELO = model.summary()

# Obtener nombres de las features (para degree=1, son las originales)
poly_feature_names = poly.get_feature_names_out(features)

# Para optimización: Selecciona un escenario fijo (e.g., el primer row de las exógenas: primeras 8 columns)
exogenas = X.iloc[0, :8].values  # OXIDO_COSTO a cantidad_chatarra

# Función para predecir utilidad dada los 3 coeficientes (con exógenas fijas)
def predict_utilidad(coefs):
    input_full = np.concatenate([exogenas, coefs])  # Exógenas (8) + coefs (3) = 11 features originales
    input_poly = poly.transform(input_full.reshape(1, -1))  # Genera (para degree=1, no cambia)
    input_poly_scaled = scaler.transform(input_poly)  # Normaliza
    input_poly_scaled_with_const = sm.add_constant(input_poly_scaled, has_constant='add')  # Añade constante
    return model.predict(input_poly_scaled_with_const)[0]

# Función objetivo para minimizar (negativa de la utilidad para maximizar)
def objective(coefs):
    return -predict_utilidad(coefs)

# Optimización de los 3 coeficientes con bounds [0,1]
initial_coefs = np.array([0.5, 0.5, 0.5])
bounds = [(0, 1)] * 3
result = minimize(objective, initial_coefs, bounds=bounds, method='L-BFGS-B')

# Resultados de optimización
opt_coefs = result.x
max_utilidad_predicha = -result.fun

print("\nCoeficientes óptimos encontrados:")
print("coeficiente_sulfurico:", opt_coefs[0])
print("coeficiente_chatarra:", opt_coefs[1])
print("coeficiente_ferrico:", opt_coefs[2])
print("Utilidad máxima predicha:", max_utilidad_predicha)

# Para desnormalizar y obtener la fórmula en escala original
params_scaled = model.params[1:]  # Excluye constante
intercept_scaled = model.params[0]

params_unscaled = params_scaled / scaler.scale_
intercept_unscaled = intercept_scaled - np.sum(params_scaled * scaler.mean_ / scaler.scale_)

# Fórmula desnormalizada aproximada
print("\nFórmula desnormalizada aproximada (para datos originales no escalados):")
formula_terms = [f"{p:.4f} * {poly_feature_names[i]}" for i, p in enumerate(params_unscaled)]
print("Utilidad ≈", intercept_unscaled, "+", " + ".join(formula_terms))

ECUACION_OBTENIDA = f"Utilidad ≈ {intercept_unscaled} + " + " + ".join(formula_terms)

# Guardar la fórmula en JSON para uso en otro script
formula_dict = {
    'intercept': intercept_unscaled,
    'terms': {poly_feature_names[i]: params_unscaled[i] for i in range(len(params_unscaled))}
}
with open(f'formula_flexible_{globales.grado}.json', 'w') as f:
    json.dump(formula_dict, f, indent=4)
print(f"\nFórmula guardada en 'formula_flexible_{globales.grado}.json' para uso en otro script.")

# Ejemplo de prueba con la fórmula desnormalizada en un dato original (e.g., primer row)
input_original = X.iloc[0].values  # Todas 11 features originales
input_poly_original = poly.transform(input_original.reshape(1, -1))[0]  # Para degree=1, igual a input
pred_with_formula = intercept_unscaled + np.dot(params_unscaled, input_poly_original)
print("\nPrueba con fórmula desnormalizada en primer row original:")
print("Utilidad predicha con fórmula:", pred_with_formula)
print("Utilidad real del primer row:", y.iloc[0])

with open(f'resumen_modelo_{globales.grado}.txt', 'w') as f:
    f.write(f"{RESUMEN_MODELO}\n")
    f.write(f"{ECUACION_OBTENIDA}\n")
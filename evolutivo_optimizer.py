from scipy.optimize import differential_evolution
import numpy as np
from utilidad_predictor import UtilidadPredictor  # Importa tu class anterior

# Clase para el optimizador evolutivo
class EvolutivoOptimizer:
    def __init__(self, predictor):
        self.predictor = predictor

    def optimize(self, exogenas):
        # Función objective: -utilidad (para maximizar, minimizamos negativa)
        def objective(coefs):
            params = np.concatenate([exogenas, coefs])
            return -self.predictor.obtener_utilidad(params)

        # Bounds para coefs [0,1]^3
        bounds = [(0, 1)] * 3

        # Diferential evolution (algoritmo evolutivo)
        result = differential_evolution(objective, bounds, strategy='best1bin', popsize=15, maxiter=50, seed=42)

        opt_coefs = result.x
        max_utilidad = -result.fun

        return opt_coefs, max_utilidad

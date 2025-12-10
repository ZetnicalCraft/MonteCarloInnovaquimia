from skopt import gp_minimize
from skopt.space import Real
import numpy as np
from utilidad_predictor import UtilidadPredictor  # Importa tu class

# Clase para el optimizador bayesiano
class BayesianoOptimizer:
    def __init__(self, predictor):
        self.predictor = predictor

    def optimize(self, exogenas):
        # Función objective: -utilidad (para maximizar, minimizamos negativa)
        def objective(coefs):
            params = np.concatenate([exogenas, coefs])
            return -self.predictor.obtener_utilidad(params)

        # Espacio de búsqueda: 3 coefs en [0,1]
        space = [Real(0, 1, name='coef_sulfurico'), 
                 Real(0, 1, name='coef_chatarra'), 
                 Real(0, 1, name='coef_ferrico')]

        # Optimización bayesiana (gp_minimize con 50 calls, 10 initial random)
        result = gp_minimize(objective, space, n_calls=50, n_random_starts=10, random_state=42)

        opt_coefs = result.x
        max_utilidad = -result.fun

        return opt_coefs, max_utilidad

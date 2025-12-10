import numpy as np
from utilidad_predictor import UtilidadPredictor  # Importa tu class

# Clase para el optimizador fuerza bruta con MC
class FuerzaBrutaOptimizer:
    def __init__(self, predictor, n_samples=10000):
        self.predictor = predictor
        self.n_samples = n_samples

    def optimize(self, exogenas):
        # Generar n_samples coefs random en [0,1]^3
        coefs_random = np.random.uniform(0, 1, size=(self.n_samples, 3))

        # Evaluar utilidad para cada
        utils = np.zeros(self.n_samples)
        for i in range(self.n_samples):
            params = np.concatenate([exogenas, coefs_random[i]])
            utils[i] = self.predictor.obtener_utilidad(params)

        # Encontrar el mejor (máxima utilidad)
        best_idx = np.argmax(utils)
        opt_coefs = coefs_random[best_idx]
        max_utilidad = utils[best_idx]

        return opt_coefs, max_utilidad

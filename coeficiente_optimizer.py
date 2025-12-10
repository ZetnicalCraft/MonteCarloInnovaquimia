from evolutivo_optimizer import EvolutivoOptimizer
from bayesiano_optimizer import BayesianoOptimizer
from fuerza_bruta_optimizer import FuerzaBrutaOptimizer
from utilidad_predictor import UtilidadPredictor
import numpy as np

class CoefOptimizer:
    def __init__(self, predictor, metodo):
        self.predictor = predictor
        self.metodo = metodo
        if self.metodo == 1:
            self.optimizer = EvolutivoOptimizer(self.predictor)
        elif self.metodo == 2:
            self.optimizer = BayesianoOptimizer(self.predictor)
        elif self.metodo == 3:
            self.optimizer = FuerzaBrutaOptimizer(self.predictor, n_samples=10000)  # Ajusta n_samples si necesitas
        else:
            raise ValueError("Método inválido: debe ser 1 (evolutivos), 2 (bayesiano) o 3 (fuerza bruta).")

    def optimize(self, exogenas):
        return self.optimizer.optimize(exogenas)

# Ejemplo de uso (comenta si no necesitas)
predictor = UtilidadPredictor(5)  # NN
optimizer = CoefOptimizer(predictor, metodo=1)  # 1 para evolutivos
exogenas = np.array([0.895,0.329,0.462,0.204,0.741,951,2598,373])
opt_coefs, max_utilidad = optimizer.optimize(exogenas)
print("Coeficientes óptimos:", opt_coefs)
print("Utilidad máxima:", max_utilidad)
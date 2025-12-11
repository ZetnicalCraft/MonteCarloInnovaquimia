from coeficiente_optimizer import CoefOptimizer
from utilidad_predictor import UtilidadPredictor
import numpy as np

class GlobalOpt:
    def __init__(self, predictor, datos_monte_carlo, documento_escribir):
        self.predictor = predictor
        self.datos_monte_carlo = datos_monte_carlo
        self.documento_escribir = documento_escribir
        self.coeficiente_optimizer_1 = CoefOptimizer(self.predictor, 1)
        self.coeficiente_optimizer_2 = CoefOptimizer(self.predictor, 2)
        self.coeficiente_optimizer_3 = CoefOptimizer(self.predictor, 3)
        self.counter_1 = 0
        self.counter_2 = 0
        self.counter_3 = 0
        self.diferencial_acumulado = 0

    def optimizar_global(self):
        with open(self.datos_monte_carlo, "r", encoding="utf-8") as reading, \
        open (self.documento_escribir, "w", encoding="utf-8") as writing:
            next(reading)
            count = 0
            for linea in reading:
                count = count + 1
                print(f"Línea: {count}/100")
                valores = linea.strip().split(",")
                valores = [float(v) for v in valores]
                exogenas = np.array(valores[:8])
                utilidad_original = valores[-1]
                opt_coefs_1, max_utilidad_1 = self.coeficiente_optimizer_1.optimize(exogenas)
                opt_coefs_2, max_utilidad_2 = self.coeficiente_optimizer_2.optimize(exogenas)
                opt_coefs_3, max_utilidad_3 = self.coeficiente_optimizer_3.optimize(exogenas)
                utilidades = np.array([max_utilidad_1,max_utilidad_2,max_utilidad_3])
                utilidad_maxima = np.max(utilidades)
                posicion_utilidad_maxima = np.argmax(utilidades)
                diferencial = utilidad_maxima - utilidad_original
                self.diferencial_acumulado = self.diferencial_acumulado + diferencial
                if posicion_utilidad_maxima == 0:
                    self.counter_1 = self.counter_1 + 1
                elif posicion_utilidad_maxima == 1:
                    self.counter_2 = self.counter_2 + 1
                elif posicion_utilidad_maxima == 2:
                    self.counter_3 = self.counter_3 + 1
                writing.write(f"{diferencial}\n")
        print(f"END. Counter_1: {self.counter_1}, Counter_2: {self.counter_2}, Counter_3: {self.counter_3}")
        print(f"Diferencial promedio: {self.diferencial_acumulado/5}")

    def test (self):
        with open(self.datos_monte_carlo, "r", encoding="utf-8") as reading, \
        open (self.documento_escribir, "w", encoding="utf-8") as writing:
            next(reading)
            count = 0
            utilidad_original_acumulada = 0
            for linea in reading:
                count = count + 1
                print(f"Línea: {count}/100")
                valores = linea.strip().split(",")
                valores = [float(v) for v in valores]
                exogenas = np.array(valores[:8])
                utilidad_original = valores[-1]
                utilidad_original_acumulada = utilidad_original_acumulada + utilidad_original
            print(f"Utilidad original acumulada: {utilidad_original_acumulada}")

optimizador_global = GlobalOpt(UtilidadPredictor(5),"datos_monte_carlo_3.csv","datos_optimizado.csv")
# optimizador_global.optimizar_global()

# Luego de probar para cien casos, se obtuvo:
# END. Counter_1: 48, Counter_2: 40, Counter_3: 12
# Diferencial promedio: 1567.38731105
# Utilidad original promedio: -4156.16962258
# Diferencia promedio: 37.712303716%
import pandas as pd
from utilidad_predictor import UtilidadPredictor

def crear_utilidad_surrogado_csv(modelo_surrogado):
    nombre_archivo = f"utilidad_surrogado_{modelo_surrogado}.csv"
    predictor = UtilidadPredictor(modelo_surrogado)
    columnas = ['OXIDO_COSTO', 'CHATARRA_COSTO', 'SULFURICO_COSTO', 'FERROSO_PRECIO', 
                'FERRICO_PRECIO', 'cantidad_oxido', 'cantidad_sulfurico', 'cantidad_chatarra', 
                'coeficiente_sulfurico', 'coeficiente_chatarra', 'coeficiente_ferrico']
    df = pd.read_csv('datos_monte_carlo_2.csv', usecols=columnas)

    with open(nombre_archivo, "a") as archivo:
        archivo.write("Utilidad_surrogado\n")

    for index, fila in df.iterrows():
        Utilidad_surrogado = predictor.obtener_utilidad(fila)
        with open(nombre_archivo, "a") as archivo:
            archivo.write(f"{Utilidad_surrogado}\n")
        if index % 1000 == 0:
            print(f"{index}/1000000")

def obtener_error_promedio(modelo_surrogado):
    nombre_archivo = f"utilidad_surrogado_{modelo_surrogado}.csv"
    print("Cargando Utilidad...")
    datos_reales = pd.read_csv('datos_monte_carlo_2.csv', usecols=["Utilidad"])
    print("Cargando Utilidad surrogado...")
    datos_estimados = pd.read_csv(nombre_archivo, usecols=["Utilidad_surrogado"])
    print("Ambas utilidades cargadas correctamente.")

    if len(datos_reales) != len(datos_estimados):
        raise ValueError("Los DataFrames tienen diferentes longitudes.")
    
    error_fraccional = []
    ssres_total = []
    sstot_total = []

    promedio_datos_reales = datos_reales['Utilidad'].mean()

    for i in range(len(datos_reales)):
        valor_real = datos_reales['Utilidad'].iloc[i]
        valor_estimado = datos_estimados['Utilidad_surrogado'].iloc[i]

        error = abs(valor_real -valor_estimado) / abs(valor_real)
        error_fraccional.append(error)

        ssres = (valor_real - valor_estimado) * (valor_real - valor_estimado)
        ssres_total.append(ssres)

        sstot = (valor_real - promedio_datos_reales) * (valor_real - promedio_datos_reales)
        sstot_total.append(sstot)

        if i % 1000 == 0:
            print(f"{i}/1000000")

    error_fraccional_promedio = sum(error_fraccional) / len(error_fraccional)
    ssres_sigma = sum(ssres_total) / len(ssres_total)
    sstot_sigma = sum(sstot_total) / len(sstot_total)
    r2 = 1 - (ssres_sigma / sstot_sigma)

    print(f"error_fraccional_promedio: {error_fraccional_promedio}")
    print(f"R2: {r2}")

    
crear_utilidad_surrogado_csv(5)
obtener_error_promedio(5)

#for i in range(0,3):
#    modelo_surrogado = i + 1
#    crear_utilidad_surrogado_csv(modelo_surrogado)

# error_fraccional_promedio: 1.5891580863687385
# R2: 0.7233218776191601

# error_fraccional_promedio: 0.8870064493234908
# R2: 0.9128932397825612

# error_fraccional_promedio: 0.530794874565678
# R2: 0.9673946826769915

# error_fraccional_promedio: 0.38536381620633586
# R2: 0.9845900463065392

# error_fraccional_promedio: 0.11119795156947954
# R2: 0.9984744596308178
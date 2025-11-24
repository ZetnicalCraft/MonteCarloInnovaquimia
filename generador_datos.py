import random

class Estequiometria:
    
    def __init__(self):
        # Los siguientes valores se modifican de ser necesario. Ojo, esto es USD / TM
        self.OXIDO = [694, 75]
        self.CHATARRA = [352, 79]
        self.SULFURICO = [349, 71]
        self.FERROSO = [508, 125]
        self.FERRICO = [508, 125]
        # Los siguientes valores van en dolares.
        self.OXIDO_COSTO = 0
        self.CHATARRA_COSTO = 0
        self.SULFURICO_COSTO = 0
        self.FERROSO_PRECIO = 0
        self.FERRICO_PRECIO = 0
        # Los siguientes valores van en kilogramos.
        self.cantidad_oxido = 0
        self.cantidad_sulfurico = 0
        self.cantidad_chatarra = 0
        # Los siguientes valores van de [0,1].
        self.coeficiente_sulfurico = 0
        self.coeficiente_chatarra = 0
        self.coeficiente_ferrico = 0
        # Los valores dependientes de otros. Empezando por sulfurico.
        self.cantidad_sulfurico_A = 0
        self.cantidad_sulfurico_B = 0
        # Chatarra
        self.cantidad_chatarra_B = 0
        self.cantidad_chatarra_C = 0
        # Ferrico
        self.cantidad_ferrico_venta = 0
        self.cantidad_ferrico_A = 0
        self.cantidad_ferrico_C = 0
        # Ferroso
        self.cantidad_ferroso_C = 0
        self.cantidad_ferroso_B = 0
        self.cantidad_ferroso_venta = 0
        # Utilidad
        self.Utilidad = 0
        # String a imprimir
        self.fila = ""

    def aleatorio_de_promedio(self, promedio, desviacion_estandar):
        limite_inferior = promedio - (3 * desviacion_estandar)
        if limite_inferior < 0:
            limite_inferior = 0
        limite_superior = promedio + (3 * desviacion_estandar)
        valor_aleatorio = random.randint(limite_inferior, limite_superior)
        return valor_aleatorio
        
    # A partir de ahora es de procesos.
    def inicializar_costos(self):
        self.OXIDO_COSTO = self.aleatorio_de_promedio(self.OXIDO[0], self.OXIDO[1]) / 1000
        self.CHATARRA_COSTO = self.aleatorio_de_promedio(self.CHATARRA[0], self.CHATARRA[1]) / 1000
        self.SULFURICO_COSTO = self.aleatorio_de_promedio(self.SULFURICO[0], self.SULFURICO[1]) / 1000
        self.FERRICO_PRECIO = self.aleatorio_de_promedio(self.FERRICO[0], self.FERRICO[1]) / 1000
        self.FERROSO_PRECIO = self.aleatorio_de_promedio(self.FERROSO[0], self.FERROSO[1]) / 1000

    def inicializar_cantidades(self):
        self.cantidad_oxido = random.randint(0, 10000)
        self.cantidad_sulfurico = random.randint(0, 10000)
        self.cantidad_chatarra = random.randint(0, 10000)

    def inicializar_coeficientes(self):
        self.coeficiente_sulfurico = random.uniform(0, 1)
        self.coeficiente_chatarra = random.uniform(0, 1)
        self.coeficiente_ferrico = random.uniform(0, 1)

    def realizar_proceso_A(self):
        self.cantidad_sulfurico_A = self.coeficiente_sulfurico * self.cantidad_sulfurico
        self.cantidad_sulfurico_B = (1 - self.coeficiente_sulfurico) * self.cantidad_sulfurico
        oxido_da_teorico_ferrico_A = self.cantidad_oxido * (0.3998778 / 0.1596882)
        sulfurico_da_teorico_ferrico_A = self.cantidad_sulfurico_A * (0.3998778 / (3 * 0.0980785))
        if oxido_da_teorico_ferrico_A <= sulfurico_da_teorico_ferrico_A:
            self.cantidad_ferrico_A = oxido_da_teorico_ferrico_A
        else:
            self.cantidad_ferrico_A = sulfurico_da_teorico_ferrico_A

    def realizar_proceso_B(self):
        self.cantidad_chatarra_B = self.coeficiente_chatarra * self.cantidad_chatarra
        self.cantidad_chatarra_C = (1 - self.coeficiente_chatarra) * self.cantidad_chatarra
        chatarra_da_teorico_ferroso_B = self.cantidad_chatarra_B * (0.1519076 / 0.0558450)
        sulfurico_da_teorico_ferroso_B = self.cantidad_sulfurico_B * (0.1519076 / 0.0980785)
        if chatarra_da_teorico_ferroso_B <= sulfurico_da_teorico_ferroso_B:
            self.cantidad_ferroso_B = chatarra_da_teorico_ferroso_B
        else: 
            self.cantidad_ferroso_B = sulfurico_da_teorico_ferroso_B

    def realizar_proceso_C(self):
        self.cantidad_ferrico_venta = self.coeficiente_ferrico * self.cantidad_ferrico_A
        self.cantidad_ferrico_C = (1 - self.coeficiente_ferrico) * self.cantidad_ferrico_A
        chatarra_da_teorico_ferroso_C = self.cantidad_chatarra_C * ((3 * 0.1519076) / 0.0558450)
        ferrico_da_teorico_ferroso_C = self.cantidad_ferrico_C * ((3 * 0.1519076) / 0.3998778)
        if chatarra_da_teorico_ferroso_C <= ferrico_da_teorico_ferroso_C:
            self.cantidad_ferroso_C = chatarra_da_teorico_ferroso_C
        else:
            self.cantidad_ferroso_C = ferrico_da_teorico_ferroso_C
        self.cantidad_ferroso_venta = self.cantidad_ferroso_B + self.cantidad_ferroso_C

    def obtener_utilidad(self):
        self.Utilidad = (self.cantidad_ferrico_venta * self.FERRICO_PRECIO) + (self.cantidad_ferroso_venta * self.FERROSO_PRECIO) - (self.cantidad_oxido * self.OXIDO_COSTO) - (self.cantidad_sulfurico * self.SULFURICO_COSTO) - (self.cantidad_chatarra * self.CHATARRA_COSTO)

    def unir_datos_en_string(self):
        self.fila = f"{self.OXIDO_COSTO},{self.CHATARRA_COSTO},{self.SULFURICO_COSTO},{self.FERROSO_PRECIO},{self.FERRICO_PRECIO},{self.cantidad_oxido},{self.cantidad_sulfurico},{self.cantidad_chatarra},{self.coeficiente_sulfurico},{self.coeficiente_chatarra},{self.coeficiente_ferrico},{self.Utilidad}\n"

    def realizar_todo(self):
        self.inicializar_costos()
        self.inicializar_cantidades()
        self.inicializar_coeficientes()
        self.realizar_proceso_A()
        self.realizar_proceso_B()
        self.realizar_proceso_C()
        self.obtener_utilidad()
        self.unir_datos_en_string()

objeto = Estequiometria()

nombre_archivo = "datos_monte_carlo_i.csv"

with open(nombre_archivo, "a") as archivo:
    archivo.write("OXIDO_COSTO,CHATARRA_COSTO,SULFURICO_COSTO,FERROSO_PRECIO,FERRICO_PRECIO,cantidad_oxido,cantidad_sulfurico,cantidad_chatarra,coeficiente_sulfurico,coeficiente_chatarra,coeficiente_ferrico,Utilidad\n")

iteraciones = 200000 # 1,000,000
for x in range(iteraciones):
    objeto.realizar_todo()
    with open(nombre_archivo, "a") as archivo:
        archivo.write(objeto.fila)
    if x % 1000 == 0:
        print(f"{x}/{iteraciones}")

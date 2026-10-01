# Resumen de resultados


## Predicción del precio del dólar

Registros: 500

`Precio_Dolar = 3985.7833 + 4.9843·Dia - 870.7317·Inflacion - 1.3774·Tasa_interes`

- Dia: por cada unidad adicional, Precio_Dolar aumenta 4.9843 pesos (resto constante). significativo (p < 0,05), p = 0; beta estandarizado = 0.998.
- Inflacion: por cada unidad adicional, Precio_Dolar disminuye 870.7317 pesos (resto constante). NO significativo (p ≥ 0,05), p = 0.096; beta estandarizado = -0.006.
- Tasa_interes: por cada unidad adicional, Precio_Dolar disminuye 1.3774 pesos (resto constante). NO significativo (p ≥ 0,05), p = 0.797; beta estandarizado = -0.001.
- Mayor impacto: Dia. Orden: Dia > Inflacion > Tasa_interes.
- Prueba: R² = 0.9963 (99.6% de la variabilidad explicada), MSE = 2376.9709, RMSE = 48.7542 pesos.


## Predicción del nivel de glucosa en sangre

Registros: 2000

`Nivel_Glucosa = 65.8609 + 1.2266·Edad + 0.9334·IMC - 2.0853·Actividad_Fisica`

- Edad: por cada unidad adicional, Nivel_Glucosa aumenta 1.2266 mg/dL (resto constante). significativo (p < 0,05), p = 0; beta estandarizado = 0.787.
- IMC: por cada unidad adicional, Nivel_Glucosa aumenta 0.9334 mg/dL (resto constante). significativo (p < 0,05), p = 0; beta estandarizado = 0.136.
- Actividad_Fisica: por cada unidad adicional, Nivel_Glucosa disminuye 2.0853 mg/dL (resto constante). significativo (p < 0,05), p = 0; beta estandarizado = -0.223.
- Mayor impacto: Edad. Orden: Edad > Actividad_Fisica > IMC.
- Prueba: R² = 0.6814 (68.1% de la variabilidad explicada), MSE = 233.6930, RMSE = 15.2870 mg/dL.


## Predicción del consumo de energía eléctrica

Registros: 10000

`Consumo_Energia = 101.2882 + 9.9529·Temperatura + 5.0198·Hora - 3.0312·Dia_Semana`

- Temperatura: por cada unidad adicional, Consumo_Energia aumenta 9.9529 kWh (resto constante). significativo (p < 0,05), p = 0; beta estandarizado = 0.780.
- Hora: por cada unidad adicional, Consumo_Energia aumenta 5.0198 kWh (resto constante). significativo (p < 0,05), p = 0; beta estandarizado = 0.546.
- Dia_Semana: por cada unidad adicional, Consumo_Energia disminuye 3.0312 kWh (resto constante). significativo (p < 0,05), p = 0; beta estandarizado = -0.095.
- Mayor impacto: Temperatura. Orden: Temperatura > Hora > Dia_Semana.
- Prueba: R² = 0.8968 (89.7% de la variabilidad explicada), MSE = 429.5187, RMSE = 20.7248 kWh.

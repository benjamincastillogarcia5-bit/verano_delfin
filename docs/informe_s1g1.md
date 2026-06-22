

**1. Análisis Exploratorio y Depuración Inicial**
Como primer paso, previo a la codificación e implementación de modelos predictivos, fue necesario realizar un análisis detallado del conjunto de microdatos (*dataset*) proporcionado. Al evaluar la integridad y estructura del contenido, se determinó la necesidad estricta de aplicar un protocolo de limpieza y normalización de datos. Esto incluyó la implementación de técnicas de observabilidad para detectar registros nulos o corruptos en columnas críticas (como unidades de medida y lugares de descargue), procediendo a su eliminación controlada para garantizar la calidad de la información antes de la fase de entrenamiento. Asimismo, se aplicó una corrección temporal concatenando fechas y horas operativas para mitigar anomalías de desfase de formato (error de 1899).

**2. Normalización Matemática y Estandarización de Variables**
Para evitar que el algoritmo de aprendizaje automático sesgara sus predicciones basándose en la magnitud bruta de las variables numéricas, se aplicaron dos criterios fundamentales de transformación:

* **Escalamiento Min-Max (Normalización Financiera):** La variable correspondiente al valor pactado del flete presentaba altas dispersiones y caracteres especiales. Tras su limpieza, se aplicó la fórmula de normalización Min-Max para comprimir los valores financieros en un rango uniforme de $[0, 1]$. La ecuación utilizada fue:

$$X_{norm} = \frac{X - X_{min}}{X_{max} - X_{min}}$$



Esto garantiza que el modelo evalúe el costo logístico de forma proporcional, sin que los grandes volúmenes de dinero opaquen a otras variables de menor magnitud.
* **Vectorización de Masas mediante Procesamiento de Lenguaje Natural (NLP):** Se detectó una inconsistencia física en la base de datos, donde las cargas se reportaban mezclando kilogramos y galones. Para unificar el volumen a masa, se desarrolló un algoritmo de búsqueda por palabras clave (*Keywords*) en la descripción del producto. Dependiendo de la familia del producto (ej. hidrocarburos, lácteos, aceites), se estimó su densidad específica y se aplicó el factor de conversión correspondiente, transformando todos los galones a kilogramos. Para productos no identificados, se utilizó la densidad estándar del agua ($3.785 \text{ kg/gal}$).

**3. Enriquecimiento Espacial y Cálculo de Distancias**
El conjunto de datos original carecía de una métrica de distancia física entre los puntos de origen y destino. Para proveer al algoritmo de un contexto espacial real, se construyó un diccionario de coordenadas geográficas (latitud y longitud) abarcando los principales nodos logísticos del país. Utilizando estos datos, se aplicó la **fórmula del semiverseno (Haversine)** de manera vectorizada para calcular la distancia geodésica (en línea recta) entre ciudades:

$$d = 2r \arcsin\left(\sqrt{\sin^2\left(\frac{\Delta\phi}{2}\right) + \cos\phi_1\cos\phi_2\sin^2\left(\frac{\Delta\lambda}{2}\right)}\right)$$


*Donde $r$ es el radio de la Tierra (6371 km), $\phi$ representa las latitudes y $\lambda$ las longitudes.*

**4. Ingeniería de Equidad (Fairness Engineering)**
Para habilitar el motor de auditoría algorítmica, se formalizaron las dos variables centrales del estudio de sesgo territorial:

* **Atributo Protegido ($A$):** Mediante la extracción de la raíz del código DIVIPOLA, se segmentó geográficamente el país. Se asignó el valor $A=0$ (Grupo Protegido) a los viajes dirigidos hacia la infraestructura periférica e intermodal, y $A=1$ (Grupo Favorecido) a los nodos de la red carretera central tradicional.
* **Variable Objetivo ($Y$):** Se estableció un umbral crítico de ineficiencia basado en el percentil 75 ($\tau$) de las horas de espera en descargue. Si el tiempo de espera de un viaje superaba dicho umbral ($\tau = 0.62$ horas), se etiquetó como un riesgo o retraso crítico ($Y=1$); de lo contrario, se consideró una operación estándar ($Y=0$).

**5. Modelamiento Predictivo y Auditoría de Sesgo**
Con el *dataset* transformado, se procedió a la fase de inferencia. Las variables categóricas (nodos de origen y destino) fueron codificadas numéricamente. Los datos se dividieron mediante un muestreo estratificado (70% para entrenamiento, 30% para validación) para preservar el balance de clases.

Se entrenó un modelo base de *Random Forest* (Bosque Aleatorio). Posteriormente, las predicciones del modelo fueron sometidas a un motor de auditoría de equidad (*Fairness ML*) para calcular tres métricas fundamentales:

1. **Diferencia de Paridad Estadística (SPD):** Para medir la independencia de las predicciones respecto al territorio.
2. **Impacto Dispar (DI):** Verificando el cumplimiento de la regla legal del 80% en la asignación de predicciones de riesgo.
3. **Diferencia de Igualdad de Oportunidades (EOD):** Contrastando la métrica de *Recall* (Sensibilidad) entre los grupos $A=0$ y $A=1$, para cuantificar la "ceguera algorítmica" o la incapacidad del modelo para detectar retrasos reales en los nodos periféricos frente a la infraestructura central.


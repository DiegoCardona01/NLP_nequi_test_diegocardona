# Prueba técnica Machine Learning Nequi - NLP

Nombre: Diego Fernando Cardona Pineda

correo: diegocp031293@gmail.com

# Introducción

Este repositorio, contiene la solución a la prueba técnica para ingeniero de Machine Learning, como fuente de datos para esta se utiliza el conjunto de datos *Consumer Complaint Dataset* que se encuentra en Kaggle (enlace directo: https://www.kaggle.com/datasets/namigabbasov/consumer-complaint-dataset?resource=download). Este set de datos contiene información bancaria de personas estadounidense, fue recolectada por *The Consumer Financial Protection Bureau* (CFPB), entre las variables que tiene se encuentran comentarios de los clientes sobre sus quejas o comentarios y asociados a estas una clasificación tentativa del tipo de queja o comentario al que se refiere. Por ejemplo, un comentario como el siguiente:

- *I sent a letter to the Credit Bureaus requesting to reinvestigate the disputed Accounts from my cred...*

Corresponde a la categoría de *Credit Reporting*. Las distintas categorías existentes son:

0. Credit Reporting.
1. Debt Collection.
2. Loans.
3. Credit Card Services.
4. Bank Accounts and Services.

En el dataset, se encuentran distintas variables que pueden ser de utilidad, pero para practicidad de la prueba nos enfocaremos en el siguiente tipo de modelo:

- Tenemos el conjunto de datos de entrenamiento, datos mencionados, que para cada tipo de comentario (variable X feature) nos va a asignar una categoría correspondiente (variable Y target).
- Entonces nuestro modelo al entrenarse con estos datos, va a poder predecir a qué tipo de categoría, entre las 5 ya mencionadas, corresponde el comentario.
- Este tipo de modelo es útil ya que se puede usar con la información de redes sociales o comentarios de servicio al cliente y ver preferencias, gustos, sentimientos, categorías, etc del cliente. De esta forma se le puede dar un mejor enfoque y tratamiento dentro de una entidad bancaria al cliente.

# 1. Propuesta de arquitectura en la nube

Para desarrollar el proyecto, se optó por una arquitectura basada en nube de Azure Web Service (AWS), junto con la gestión del versionamiento en GitHub y la automatización de los diferentes pipelines con GitHub Actions.

A continuación se muestra la arquitectura en alto nivel:

![Diagrama Arquitectura](imagenes/arquitectura_nube_nequi.png)

Antes de continuar hablando sobre las componentes, cabe mencionar la metodología utilizada, esta se puede entender en el siguiente flujo:

![lifecycle](imagenes/machinelearning_lifecycle.jpg)

Esta propuesta de trabajo, empieza por entender el problema, para este caso nuestro objetivo fue el ya mencionado clasificador de categorías bancarias sobre las quejas de los clientes. En esta parte se hizo un análisis exploratorio de los datos y la primer parte experimental del modelamiento, esta se encuentra consignada en la carpeta notebooks.
En la primer etapa de experimentación, se hizo un análisis de la nulidad de los datos, tipos de datos, cantidad, comportamientos, en esta primer parte se observo que los principales temas abordados por los clientes bancarios son los temas de reportes crediticios,seguido de cobro de deudas. Esto nos muestra las tendencias de clientes y sus principales cuestionamientos a los que hay que revisar el por qué tantos comentarios (¿son buenos o malos? ¿debemos mejorar como entidad? ¿esto hace perder valor a otros productos ofrecidos?).

![categorias](imagenes/categorias_bank.png)

En la parte experimental, se hizo una reducción de los datos ya que la capacidad del equipo con que se cuenta es baja, también por cuestiones de tiempo se hicieron optimizaciones en los datos que se mencionarán más adelante. Originalmente la fuente de datos cuenta con aprox 2 millones de datos y se redujeron a 1 millon de registros.

Los datos originales vienen desde el 2011 hasta el 2024 y descartando la principal categoría de los temas mencionados por los clientes, vemos que hay una tendencia al crecimiento sobre los temas de cuentas bancarias y tarjetas de crédito, pero estas tienen un crecimiento exponencial y desbordado en 2023. Este periodo, coincide con el regreso a la normalidad pos-pandemia, ¿puede ser este un punto de inflexión? podríamos hacer una regresión con datos de casos de covid-19 por ejemplo, pero deberá ser minuciosa dado que correlación no implica causalidad como se sabe.

Entre los principales crecimientos, vemos que también puntea en 2023 el tema relacionado con cobro de deuda, este puede ser un buen indicio para responder algunas de las anteriores preguntas. Recordando que en epoca de pandemia las tasas de interés estuvieron en mínimos ehistóricos las personas pudieron haber sufrido por la subida de estas y la tentativa de recesión mundial.

Posterior al análisis hecho (análisis corto, puede ser más profundo pero nuestro interes es optimizar el tiempo de todos los pasos del proyecto y priorizar el despliegue), se plantea un modelo de clasificación utilizando el algoritmo de modelamiento LightGBM, este framework consta de un modelo gradiente con boost. El modelo crea una serie de árboles de decisión a partir de los datos suministrados y las categorías referenciadas, de esta forma aprende a decidir a qué categoría corresponde algún texto.

Para una correcta implementación del modelo, se implementó una encodización de las categorías, en vez de tener las 5 categorías ya mencionadas, se les cambia por valores numéricos enteros (0,1,2,3,4) para que el modelo los interprete mejor y sea más optimo.

## Escalabilidad

- Amazon S3 se utiliza como repositorio centralizado para almacenar grandes volúmenes de datos, modelos y artefactos. Su arquitectura distribuida permite manejar datasets de tamaño prácticamente ilimitado sin impactar el desempeño.
- Elastic Beanstalk, en combinación con Docker, permite desplegar de forma automática y sencilla la aplicación basada en FastAPI. La capacidad de autoescalado de Elastic Beanstalk garantiza que la infraestructura crezca o disminuya según la demanda del tráfico o la carga de trabajo de inferencia.
- FastAPI es un framework asíncrono y de alto rendimiento, ideal para manejar peticiones simultáneas y realizar inferencias en tiempo real de manera eficiente.
- El entrenamiento y la preparación de datos se realizan fuera del entorno de producción, cabe decir que la solución funciona tanto en bach como en tiempo real, el modelo se entrena cada domingo en la tarde y después se pueden hacer predicciones sobre datos almacenados. Además, el punto final (endpoint) creado, permite que se puedan obtener respuestas en el momento.

## Confiabilidad

- Elastic Beanstalk monitorea continuamente la salud de los entornos desplegados. Si un contenedor Docker falla, Elastic Beanstalk lo reinicia o escala nuevas instancias sin intervención manual.
- La persistencia de los datos y los modelos en S3 asegura que la información crítica no se pierda en caso de una interrupción del servicio o fallo del servidor.
- GitHub Actions se encarga de la automatización de los pipelines de CI/CD, asegurando que las nuevas versiones del modelo y el código se desplieguen de forma consistente, fiable y auditada.
- La separación clara de componentes (preprocesamiento, entrenamiento, despliegue) permite una fácil observabilidad, debugging y mantenimiento, lo que aumenta la confiabilidad del sistema global.

## Elección de los componentes para la preparación de datos, trabajos ETL, implementación de modelos y su papel en la arquitectura

### 1. **Data Preparation y ETL:**

- **Apache Airflow:**
  - Orquestación de los flujos ETL.
  - Permite la programación y monitoreo de tareas en pipelines de datos.
  - Ideal para tareas recurrentes, manejo de dependencias y reintentos automáticos.

- **Python (pandas / numpy):**
  - Procesamiento y limpieza de datos antes de su almacenamiento o consumo.
  - Transformaciones ligeras y manipulación de datasets.

- **Apache Spark (opcional para grandes volúmenes):**
  - Procesamiento distribuido para datasets de gran tamaño.
  - Escalable y eficiente para pipelines de datos pesados.

### 2. **Almacenamiento de Datos:**

- **Snowflake:**
  - Data warehouse en la nube.
  - Permite almacenar datos estructurados, optimizando consultas y tiempos de respuesta.
  - Uso de esquemas particionados por país y ciudad para segmentar datos y mejorar el performance.

### 3. **Model Training e Implementación:**

- **scikit-learn / XGBoost / LightGBM:**
  - Entrenamiento de modelos predictivos, en particular para predicciones de órdenes activas.
  - Flexibilidad para ajustar hiperparámetros y experimentar con diferentes algoritmos.

- **MLflow:**
  - Seguimiento de experimentos, métricas de modelos y control de versiones.
  - Facilita la trazabilidad y comparación entre modelos.

- **Docker:**
  - Empaquetado de los modelos para garantizar consistencia entre entornos de desarrollo y producción.
  - Asegura que las dependencias del modelo se mantengan estables.

### 4. **Deployment y Monitoreo de Modelos:**

- **FastAPI:**
  - Creación de servicios REST para exponer las predicciones de los modelos.
  - Ligero y de alto rendimiento, ideal para APIs de predicciones en tiempo real.

- **Prometheus + Grafana:**
  - Monitoreo de la performance de los servicios de predicción.
  - Métricas como latencia, throughput y estado de los endpoints.

### 5. **Arquitectura General y Roles:**

- **Airflow** coordina la ejecución de los pipelines ETL y el refresco de los modelos.
- **Snowflake** almacena la información necesaria para el entrenamiento y el

## Detalle de la preparación de datos y trabajos ETL

El proceso de preparación de datos y ETL dentro del proyecto sigue una estructura automatizada y controlada a través de pipelines CI/CD, orquestados principalmente con **Airflow** y definidos en los workflows de GitHub Actions.

### **Fuentes de Datos:**
- Datos históricos y en tiempo real extraídos desde **Snowflake**.
- Información de órdenes activas, históricos por ciudad y país, métricas de performance de RTs.

### **Pasos principales del ETL:**

1. **Extracción**:
   - Consulta y extracción de datos desde **Snowflake**, segmentado por `country` y `city_id`.
   - Se asegura la disponibilidad de datos recientes para mantener actualizado el entrenamiento y scoring.

2. **Transformación**:
   - Limpieza y normalización de datos con **pandas**.
   - Cálculo de nuevas variables y features relevantes para el modelo (por ejemplo: variaciones por hora, día de la semana, tendencias recientes).

3. **Carga**:
   - Los datos procesados se almacenan nuevamente en **Snowflake**, en tablas específicas para entrenamiento (`training_data`) y scoring (`scoring_data`).
   - Se crean vistas particionadas para facilitar el acceso eficiente a los datos.

---

## Detalle del entrenamiento del modelo

El proceso de entrenamiento del modelo sigue una lógica controlada y reproducible:

1. **Preprocesamiento**:
   - Se filtran y limpian datos atípicos (outliers) que puedan distorsionar el modelo.
   - Normalización de variables numéricas.
   - Se crean datasets balanceados para evitar sesgos.

2. **Entrenamiento**:
   - Uso de **scikit-learn**, **XGBoost** o **LightGBM** según la tarea y el volumen de datos.
   - Se realiza el ajuste de hiperparámetros mediante validación cruzada.
   - El modelo se entrena a nivel país y ciudad para capturar comportamientos específicos de cada segmento.

3. **Evaluación**:
   - Métricas de validación: MAE, RMSE y R2 Score.
   - Comparación de modelos para asegurar la mejor performance antes del deployment.

4. **Registro**:
   - Se guarda el mejor modelo con **MLflow**.
   - Se almacenan las métricas de cada experimento para trazabilidad.

5. **Deployment**:
   - El modelo se empaqueta en **Docker** y se despliega a través de **Elastic Beanstalk** como una API REST usando **FastAPI**.
   - **Elastic Beanstalk** gestiona el escalado automático y la disponibilidad del servicio.

---

## Pipelines de CI/CD y su Rol en el Proyecto

El pipeline del proyecto se fundamenta en los siguientes workflows de GitHub Actions:

- `ci_cd.yml`:
  - Se ejecuta automáticamente cuando se hace **push** a la rama `main`.
  - Contiene todas las etapas del pipeline, incluidas las definidas en `ct.yml`.
  - Realiza pruebas, validaciones y, en caso de éxito, ejecuta el proceso de **continuous deployment** hacia **Elastic Beanstalk**.

- `ct.yml`:
  - Se ejecuta en los **push** a la rama `develop`.
  - Corre las pruebas automáticas (unitarias y de integración).
  - Valida que el código esté correcto antes de pasar a producción.

### **Resumen del flujo:**
1. Push a `develop` → se ejecuta `ct.yml` para validar el código.
2. Merge a `main` → se ejecuta `ci_cd.yml` para validar nuevamente y hacer el despliegue automático.

---

## Monitoreo y Observabilidad

El entorno desplegado en **Elastic Beanstalk** cuenta con las siguientes herramientas para monitoreo y logging:

- **CloudWatch Logs**: Recolección de logs de la aplicación y del entorno EC2.
- **Alarmas de CloudWatch**: Alertas configuradas para eventos críticos como fallos en el servicio, latencia elevada o errores HTTP 5xx.
- **Elastic Beanstalk Health Dashboard**: Visibilidad sobre el estado de las instancias y métricas básicas (CPU, memoria, latencia).
- **Prometheus + Grafana** (opcional en ambientes locales o de testing): Para monitoreo avanzado de métricas internas durante el desarrollo.

---



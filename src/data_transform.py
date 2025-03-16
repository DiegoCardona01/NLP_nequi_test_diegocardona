# Importar las librerías necesarias
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import boto3
from io import BytesIO
from datetime import datetime

# Configuración de S3
S3_BUCKET_NAME = 'nequi-mlops-data-bucket'
s3_client = boto3.client('s3')

# Función para cargar un archivo CSV desde S3
def load_data_from_s3(s3_path: str) -> pd.DataFrame:
    """Carga los datos desde S3 y devuelve un DataFrame"""
    response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=s3_path)
    body = response['Body'].read()
    return pd.read_csv(BytesIO(body))

# Función para guardar un objeto como pickle en S3
def save_pickle_to_s3(obj, s3_path: str) -> None:
    """Guarda un objeto como pickle en S3"""
    buf = BytesIO()
    joblib.dump(obj, buf)
    buf.seek(0)
    s3_client.upload_fileobj(buf, S3_BUCKET_NAME, s3_path)

# Función para limpiar y preparar los datos
def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocesa el DataFrame aplicando limpieza y transformación de texto"""
    # Definir el vectorizador TF-IDF con las stopwords en inglés predefinidas
    tfidf = TfidfVectorizer(
        max_features=500,
        ngram_range=(1, 2),
        stop_words='english'  # Stopwords de sklearn
    )

    # Aplicar el vectorizador a la columna 'narrative'
    X_tdfidf = tfidf.fit_transform(df['narrative'])

    # Imprimir el tamaño de la matriz generada
    print(f"TF-IDF matrix shape: {X_tdfidf.shape}")

    return tfidf, X_tdfidf

# Función principal que integra la carga, procesamiento y guardado de los datos
def main():
    """Función principal que ejecuta el flujo completo de carga, procesamiento y guardado de datos"""
    # Obtener la fecha actual para incluirla en los nombres de los archivos
    date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # Cargar los datos desde S3
    print("Cargando los datos desde S3...")
    df_sample_v2 = load_data_from_s3('silver/cleaned_dataset.csv')

    # Limpiar y transformar los datos
    print("Preprocesando los datos...")
    tfidf, X_tdfidf = prepare_data(df_sample_v2)

    # Definir las rutas para guardar los archivos con la fecha en el nombre
    # vectorizer_dir = 'silver'
    date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    vectorizer_dir = f'model/v_{date}'
    data_transform_dir = 'silver'
    vectorizer_path = f'{vectorizer_dir}/vectorizer_{date}.pkl'
    matrix_path = f'{data_transform_dir}/tfidf_matrix_{date}.pkl'

    # Guardar el vectorizador y la matriz generada en S3
    print(f'Vectorizador guardado en: {vectorizer_path}')
    save_pickle_to_s3(tfidf, vectorizer_path)

    print(f'Matriz TF-IDF guardada en: {matrix_path}')
    save_pickle_to_s3(X_tdfidf, matrix_path)

    print("Proceso completado exitosamente.")

# Verifica si este archivo es el principal y ejecuta el flujo
if __name__ == '__main__':
    main()

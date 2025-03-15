import re
import pandas as pd
import boto3
import io
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


# S3 CONFIG
BUCKET_NAME = 'nequi-mlops-data-bucket'
BRONZE_KEY = 'complaints_v1.csv'
SILVER_KEY = 'silver/cleaned_dataset.csv'
s3_client = boto3.client('s3')


# Regular expressions
REPLACE_BY_SPACE_RE = re.compile(r'[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')



STOPWORDS = ENGLISH_STOP_WORDS

def clean_text(text):
    text = str(text).lower()
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    text = BAD_SYMBOLS_RE.sub('', text)
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)
    return text


def load_sample_from_s3(bucket, key, frac=0.1, chunksize=100_000):
    print(f'Descargando {key} desde el bucket {bucket} en chunks...')
    
    response = s3_client.get_object(Bucket=bucket, Key=key)
    content = response['Body']

    # Aquí vamos a ir leyendo por partes
    sample_list = []

    # Lee en chunks
    reader = pd.read_csv(content, chunksize=chunksize)

    for i, chunk in enumerate(reader):
        print(f'Procesando chunk {i}...')
        
        # Saca un sample de cada chunk
        sample_chunk = chunk.sample(frac=frac, random_state=42)
        sample_list.append(sample_chunk)
    
    # Une todos los samples en un solo dataframe
    sample_df = pd.concat(sample_list, ignore_index=True)
    
    print(f'Shape del sample final: {sample_df.shape}')
    return sample_df


def save_data_to_s3(df, bucket, key):
    """
    Guarda un DataFrame de pandas como CSV en S3.
    """
    print(f'Guardando datos en {bucket}/{key}...')
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    
    s3_client.put_object(Bucket=bucket, Key=key, Body=csv_buffer.getvalue())
    print('Guardado exitoso en S3.')

def process_data(df):
    """
    Procesa el DataFrame:
    - Aplica el mapeo de categorías
    - Limpia el texto
    - Calcula el TF-IDF
    """
    print('Iniciando procesamiento de datos...')

    # Mapeo de categorías
    category_mapping = {
        'Credit Reporting': 0,
        'Debt Collection': 1,
        'Loans': 2,
        'Credit Card Services': 3,
        'Bank Accounts and Services': 4
    }

    df['product_5'] = df['product_5'].map(category_mapping)

    # Limpiar el texto de la columna narrative
    df['cleaned_narrative'] = df['narrative'].apply(clean_text)

    # Aplicar TF-IDF solo si quieres generar el vector, pero no lo guardes en CSV
    tfidf = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
    X_tfidf = tfidf.fit_transform(df['cleaned_narrative'])

    print('Procesamiento completado.')
    return df, X_tfidf

def save_sparse_to_s3(sparse_matrix, bucket, key):
    """
    Guarda un sparse matrix en S3 usando joblib.
    """
    print(f'Guardando sparse matrix en {bucket}/{key}...')
    buffer = io.BytesIO()
    joblib.dump(sparse_matrix, buffer)
    buffer.seek(0)

    s3_client.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())
    print('Sparse matrix guardada exitosamente en S3.')

def main():
    df_bronze = load_sample_from_s3(BUCKET_NAME, BRONZE_KEY)
    
    # Procesar
    df_silver, X_tfidf = process_data(df_bronze)

    # Guardar CSV normal
    save_data_to_s3(df_silver, BUCKET_NAME, SILVER_KEY)

    # Guardar sparse matrix en formato binario
    TFIDF_KEY = 'silver/tfidf_features.joblib'
    save_sparse_to_s3(X_tfidf, BUCKET_NAME, TFIDF_KEY)

if __name__ == '__main__':
    main()


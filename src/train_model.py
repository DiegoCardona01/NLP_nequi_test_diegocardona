import pandas as pd
import joblib
import os
import time
import io
import boto3
import mlflow
import mlflow.lightgbm
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
import logging

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier

DATE = '2025-03-15_23-35-18'

# Config
BUCKET_NAME = 'nequi-mlops-data-bucket'
SILVER_KEY = 'silver/cleaned_dataset.csv'
# TFIDF_KEY = 'silver/tfidf_features.joblib'
TFIDF_KEY = f'silver/tfidf_matrix_{DATE}.pkl'

LOCAL_MODEL_ROOT = 'model'
MODEL_FILENAME = 'lgbm_model.pkl'
VECTORIZER_FILENAME = 'tfidf_vectorizer.pkl'
CONFUSION_MATRIX_FILENAME = 'confusion_matrix.png'
METRICS_FILENAME = 'metrics.json'

s3_client = boto3.client('s3')


# ================================
# FUNCIONES S3
# ================================
def load_csv_from_s3(bucket, key):
    logging.info(f'Cargando {key} desde {bucket}...')
    response = s3_client.get_object(Bucket=bucket, Key=key)
    df = pd.read_csv(io.BytesIO(response['Body'].read()))
    logging.info(f'Datos cargados con shape {df.shape}')
    return df


def load_joblib_from_s3(bucket, key):
    logging.info(f'Cargando joblib {key} desde {bucket}...')
    response = s3_client.get_object(Bucket=bucket, Key=key)
    obj = joblib.load(io.BytesIO(response['Body'].read()))
    logging.info('Archivo joblib cargado correctamente.')
    return obj


def upload_file_to_s3(local_path, bucket, s3_key):
    logging.info(f'Subiendo {local_path} a s3://{bucket}/{s3_key}...')
    s3_client.upload_file(local_path, bucket, s3_key)
    logging.info(f'Subida completada: s3://{bucket}/{s3_key}')


# ================================
# PLOT MATRIZ DE CONFUSIÓN
# ================================
def plot_confusion_matrix(cm, class_names, output_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig(output_path)
    plt.close()
    logging.info(f'Matriz de confusión guardada en {output_path}')


# ================================
# ENTRENAMIENTO Y EVALUACIÓN
# ================================
def train_and_evaluate(df, X_tfidf):
    y = df['product_5']
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

    logging.info(f'Dimensiones del split -> X_train: {X_train.shape}, X_test: {X_test.shape}')

    clf = LGBMClassifier(random_state=42)

    start_train = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - start_train
    logging.info(f'Tiempo de entrenamiento: {train_time:.2f} segundos')

    start_pred = time.time()
    y_pred = clf.predict(X_test)
    pred_time = time.time() - start_pred
    logging.info(f'Tiempo de predicción: {pred_time:.2f} segundos')

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    logging.info(f'Accuracy: {accuracy:.4f}')

    date = datetime.now().strftime("%Y-%m-%d")
    # date2 = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    model_dir = os.path.join(LOCAL_MODEL_ROOT, f'v_{date}')
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, f'model_{date}.pkl')
    # vectorizer_path = os.path.join(model_dir, f'{model_dir}/vectorizer_{date2}.pkl')
    cm_path = os.path.join(model_dir, CONFUSION_MATRIX_FILENAME)
    metrics_path = os.path.join(model_dir, METRICS_FILENAME)

    joblib.dump(clf, model_path)
    # joblib.dump(X_tfidf, vectorizer_path)

    logging.info(f'Modelo guardado en {model_path}')
    # logging.info(f'Vectorizador guardado en {vectorizer_path}')

    plot_confusion_matrix(cm, class_names=clf.classes_, output_path=cm_path)

    # se guardan las métricas en JSON
    metrics = {
        'accuracy': accuracy,
        'train_time': train_time,
        'prediction_time': pred_time,
        'classification_report': report
    }
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    logging.info(f'Métricas guardadas en {metrics_path}')

    s3_model_key = f'model/v_{date}/model_{date}.pkl'
    # s3_vectorizer_key = f'model/v_{date}/vectorizer_{date}.pkl'
    s3_cm_key = f'model/v_{date}/{CONFUSION_MATRIX_FILENAME}'
    s3_metrics_key = f'model/v_{date}/{METRICS_FILENAME}'

    upload_file_to_s3(model_path, BUCKET_NAME, s3_model_key)
    # upload_file_to_s3(vectorizer_path, BUCKET_NAME, s3_vectorizer_key)
    upload_file_to_s3(cm_path, BUCKET_NAME, s3_cm_key)
    upload_file_to_s3(metrics_path, BUCKET_NAME, s3_metrics_key)

    return clf, metrics, cm_path, model_path


# ================================
# MLFLOW EXPERIMENTO
# ================================
def log_experiment_mlflow(metrics, model_path, cm_path):
    mlflow.set_experiment('complaints_classification')

    with mlflow.start_run(run_name='lightgbm_tfidf_model') as run:
        mlflow.log_param('model_type', 'LGBMClassifier')
        mlflow.log_param('vectorizer', 'TfidfVectorizer')

        mlflow.log_metric('accuracy', metrics['accuracy'])
        mlflow.log_metric('train_time', metrics['train_time'])
        mlflow.log_metric('prediction_time', metrics['prediction_time'])

        mlflow.log_artifact(model_path, artifact_path='models')
        mlflow.log_artifact(cm_path, artifact_path='plots')

        metrics_path = os.path.join(os.path.dirname(model_path), METRICS_FILENAME)
        mlflow.log_artifact(metrics_path, artifact_path='metrics')

        logging.info(f'MLFLOW Run {run.info.run_id} completado')


# ================================
# MAIN
# ================================
def main():
    # carga de datos
    df_silver = load_csv_from_s3(BUCKET_NAME, SILVER_KEY)
    X_tfidf = load_joblib_from_s3(BUCKET_NAME, TFIDF_KEY)

    # entrenamiento y evaluación
    clf, metrics, cm_path, model_path = train_and_evaluate(df_silver, X_tfidf)

    log_experiment_mlflow(metrics, model_path, cm_path)


if __name__ == '__main__':
    main()

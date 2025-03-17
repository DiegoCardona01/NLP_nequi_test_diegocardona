import boto3
from botocore.exceptions import NoCredentialsError, ClientError


def test_s3_connection():
    """
    El objetivo de este script es añadir un ejemplo básico de testeo, en este caso
    verifica la conexión a s3. En mlops se debe tener scripts de testeo por buenas prácticas
    en el continous integration y continous deployment.
    """
    try:

        s3 = boto3.client('s3', region_name='us-east-1')

        response = s3.list_buckets()

        assert 'Buckets' in response
        print("Conexión a S3 exitosa.")

    except NoCredentialsError:
        assert False, "No se encontraron credenciales de AWS."

    except ClientError as e:
        assert False, f"Error en la conexión a S3: {e}"

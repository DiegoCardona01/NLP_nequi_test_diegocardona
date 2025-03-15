import boto3
from botocore.exceptions import NoCredentialsError, ClientError

def test_s3_connection():
    try:

        s3 = boto3.client('s3', region_name='us-east-1')
        
        # Intenta listar los buckets
        response = s3.list_buckets()
        
        # Chequea si recibimos un resultado
        assert 'Buckets' in response
        print("Conexión a S3 exitosa.")
    
    except NoCredentialsError:
        assert False, "No se encontraron credenciales de AWS."
    
    except ClientError as e:
        assert False, f"Error en la conexión a S3: {e}"


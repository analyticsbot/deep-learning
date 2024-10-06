from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import subprocess

def run_spark_job():
    # Command to run spark-submit in the Spark container
    command = "docker exec recbox-spark-1 /opt/bitnami/spark/bin/spark-submit --jars /sparkdata/postgresql-42.7.4.jar /sparkdata/recommendation.py"
    
    # Execute the command
    result = subprocess.run(command, shell=True, check=True)
    
    # Log the result (optional)
    print(f"Command executed with return code: {result.returncode}")

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 10, 3),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
}

dag = DAG(
    'recommendation_dag_pythonOP',
    default_args=default_args,
    schedule_interval=None,  # Set to None for manual triggering
)

# Use PythonOperator to execute the Spark job
run_spark = PythonOperator(
    task_id='run_spark_job',
    python_callable=run_spark_job,
    dag=dag,
)


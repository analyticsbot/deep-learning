from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
from subprocess import call

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}

def run_spark_job():
    call(["spark-submit", "/sparkdata/recommendation.py"])

dag = DAG('recommendation_dag', default_args=default_args, schedule_interval='*/2 * * * *')

run_spark = PythonOperator(
    task_id='run_recommendation_system',
    python_callable=run_spark_job,
    dag=dag
)

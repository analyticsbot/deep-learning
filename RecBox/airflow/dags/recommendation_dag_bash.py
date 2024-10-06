from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 10, 3),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
}

dag = DAG(
    'recommendation_dag_bashOP',
    default_args=default_args,
    schedule_interval=None,  # Set to None for manual triggering
)

# Use BashOperator to execute the Spark job
run_spark = BashOperator(
    task_id='run_spark_job',
    bash_command="docker exec recbox-spark-1 /opt/bitnami/spark/bin/spark-submit --jars /sparkdata/postgresql-42.7.4.jar /sparkdata/recommendation.py",
    dag=dag,
)

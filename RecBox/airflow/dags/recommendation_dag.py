from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 10, 3),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}

dag = DAG(
    'recommendation_dag',
    default_args=default_args,
    schedule_interval='*/3 * * * *',  # Run every 2 minutes
)


run_spark = BashOperator(
    task_id='run_spark_job',
    bash_command="docker exec recbox-spark-1 /opt/bitnami/spark/bin/spark-submit --jars /sparkdata/postgresql-42.7.4.jar /sparkdata/recommendation.py",
    dag=dag
)

# run_spark.execute(context={})
# && docker exec recbox-spark-1 /opt/bitnami/spark/bin/spark-submit --jars /sparkdata/postgresql-42.7.4.jar /sparkdata/recommendation.py
from airflow import DAG
from airflow.providers.ssh.operators.ssh import SSHOperator
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
    'recommendation_dag_ssh',
    default_args=default_args,
    schedule_interval=None,  # Set to None for manual triggering
)

# Use SSHOperator to execute the Spark job
run_spark_job = SSHOperator(
    task_id='run_spark_job',
    ssh_conn_id='spark_ssh_connection',  # Connection ID created earlier
    command='export JAVA_HOME=/opt/bitnami/java && export PYSPARK_PYTHON=/opt/bitnami/python/bin/python3 && /opt/bitnami/spark/bin/spark-submit --jars /sparkdata/postgresql-42.7.4.jar /sparkdata/recommendation.py',
    dag=dag,
    retries=3)

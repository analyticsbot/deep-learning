from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from datetime import datetime, timedelta

# Define default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,  # Do not depend on past executions
    'start_date': datetime(2024, 10, 3),  # Start date of the DAG
    'email_on_failure': False,  # No email on failure
    'email_on_retry': False,  # No email on retry
    'retries': 1,  # Retry once in case of failure
    'retry_delay': timedelta(minutes=2),  # Delay between retries
}

# Create a new DAG with a 24-hour interval
dag = DAG(
    'recommendation_dag',
    default_args=default_args,
    schedule_interval='@daily',  # Run the DAG every 24 hours
)

now = datetime.now()
# Format the current time to create a job ID
job_id = now.strftime('%Y-%m-%d_%H-%M-%S')


# Define the DockerOperator task to run the Spark job in the Spark container
run_spark = DockerOperator(
    task_id='run_recommendation_system',
    image='spark:latest',  # Name of the Spark image
    command=["spark-submit", "/sparkdata/recommendation.py", "--job-id", job_id],  # Pass job_id to the spark-submit command
    # volumes=["/path/to/sparkdata:/sparkdata"],  # Bind mount the Spark data directory
    network_mode='mlflow_network',  # Set the correct Docker network to communicate with Spark
    dag=dag,
)

from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
from subprocess import call

# Define default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,  # Do not depend on past executions
    'start_date': datetime(2024, 1, 1),  # Start date of the DAG
    'email_on_failure': False,  # No email on failure
    'email_on_retry': False,  # No email on retry
    'retries': 1,  # Retry once in case of failure
    'retry_delay': timedelta(minutes=2),  # Delay between retries
}

# Define the function to run the Spark job
def run_spark_job(execution_date):
    # Convert execution_date to a datetime object
    # Updated format string to handle the timezone part
    execution_datetime = datetime.strptime(execution_date, '%Y-%m-%dT%H:%M:%S.%f%z')  # Adjust the format to include timezone
    # Use the execution date as the job ID
    job_id = execution_datetime.strftime('%Y-%m-%d_%H-%M-%S')
    print(f"Running Spark job with ID: {job_id}")
    call(["spark-submit", "/sparkdata/recommendation.py", "--job-id", job_id])  # Pass the job_id as argument

# Create a new DAG with a 24-hour interval
dag = DAG(
    'recommendation_dag',
    default_args=default_args,
    schedule_interval='@daily',  # Run the DAG every 24 hours
)

# Define the PythonOperator task to run the Spark job
run_spark = PythonOperator(
    task_id='run_recommendation_system',
    python_callable=run_spark_job,
    op_kwargs={'execution_date': '{{ execution_date }}'},  # Pass the execution date to the function
    dag=dag,
)

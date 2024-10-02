#!/bin/bash

# Initialize the Airflow DB
airflow db init

# Start the webserver
airflow webserver

# Fix permissions for Airflow user
chown -R airflow:airflow $(which spark-submit)
chmod u+x $(which spark-submit)

# Run the original entrypoint
exec "$@"
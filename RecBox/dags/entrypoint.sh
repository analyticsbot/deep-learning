#!/bin/bash



# Initialize the Airflow DB
airflow db init

# Fix permissions for Airflow user (if needed)
# chown -R airflow:airflow $(which spark-submit)
# chmod u+x $(which spark-submit)

# Start the webserver in the background
airflow webserver


# Create the docker group (ignore if it exists)
groupadd docker 2>/dev/null

# Add the airflow user to the docker group
usermod -aG docker airflow

# Wait for the webserver to start (optional, depends on your needs)
#sleep 10  # Adjust time based on your environment

# Run the original entrypoint (this should be the last command)
exec "$@"

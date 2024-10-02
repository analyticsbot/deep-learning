# RecBox

**RecBox** is a containerized application that leverages multiple technologies like Airflow, MLflow, Spark, and Streamlit to generate and display basic product recommendations.

## Overview

This project consists of the following components:

- **Airflow**: Manages and schedules recommendation generation jobs.
- **MLflow**: Tracks machine learning experiments and model versions.
- **Spark**: Processes data and generates basic recommendations.
- **Streamlit**: Frontend app to display the recommendations in real-time.

The application is fully containerized using Docker, allowing easy deployment and management.

## Architecture

- **Airflow**: Container responsible for orchestrating and scheduling recommendation tasks.
- **MLflow**: Container to manage experiment tracking, storing models and their versions.
- **Spark**: Container used for processing data and generating basic recommendations.
- **Streamlit**: Container for the user interface, which displays the generated recommendations.

## Project Structure

```bash
├── airflow/                 # Contains Airflow DAGs and configurations
├── mlflow/                  # Contains MLflow configurations and tracking server setup
├── spark/                   # Spark application files for generating recommendations
├── streamlit/               # Streamlit app for displaying recommendations
├── docker-compose.yml       # Docker Compose file to orchestrate all services
└── README.md                # This file
```

## Prerequisites

Before running the project, ensure you have the following installed:

- [Docker](https://www.docker.com/get-started)
- [Docker Compose](https://docs.docker.com/compose/install/)

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/RecBox.git
   cd RecBox

2. Build the Docker images & Start the Containers:
> docker-compose up --build

3. Access the services:

- Airflow: http://localhost:8081 (default credentials: airflow/airflow)
- MLflow: http://localhost:5001
- Streamlit App: http://localhost:8501

4. To stop the containers:
> docker-compose down

5. Project Components
    1. Airflow
    Purpose: To schedule jobs for data processing and model training.
    How to Use:
    Start Airflow using Docker Compose.
    Define DAGs for scheduling your ML jobs and recommendation tasks.

    2. MLflow
    Purpose: To track and store machine learning experiments, models, and metrics.
    How to Use:
    MLflow is automatically started in the MLflow container.
    Access MLflow UI at http://localhost:5000 to track models.


    4. Streamlit
    Purpose: Display recommendations to end users.
    How to Use:
    Once all services are running, open the Streamlit app on http://localhost:8501 to see the recommendations.

6. To-Do
- Implement additional recommendation algorithms.
- Add data preprocessing steps in Spark.
- Integrate authentication for the Streamlit app.
- Add tests for each component.
- Set up Airflow DAGs for scheduling Spark jobs.
- Improve the UI/UX of the Streamlit app.

7. License
This project is licensed under the MIT License.
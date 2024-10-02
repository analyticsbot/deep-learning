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

## Project Learnings
#### How to remove all containers and start from scratch?

Step 1: Stop and Remove Containers
If you have running containers, stop them first:
> docker-compose down
This command will stop and remove all containers defined in your docker-compose.yml.

Step 2: Remove Docker Images
To remove all images (including the ones you've built), use the following command:
> docker rmi $(docker images -q) --force
If you get an error that image is being used by another container, please use
> docker stop $(docker ps -a -q)

Check for container images
> docker ps
> docker rm $(docker ps -a -q)

Step 3: Remove All Volumes
If you want to remove all volumes (be cautious, as this will delete any data stored in volumes), use:

> docker volume prune -f

Step 4: Clear Build Cache
You can also clear the Docker build cache by running:

> docker builder prune -a -f

Step 5: Start from Scratch
Now, you can rebuild your Docker images using Docker Compose:

> docker-compose up --build


#### Dockerfile vs docker-compose:
1. https://www.theserverside.com/blog/Coffee-Talk-Java-News-Stories-and-Opinions/Dockerfile-vs-docker-compose-Whats-the-difference#:~:text=The%20key%20difference%20between%20the,used%20to%20run%20Docker%20containers.
2. https://stackoverflow.com/questions/29480099/whats-the-difference-between-docker-compose-vs-dockerfile

#### Docker workshop

This 45-minute workshop contains step-by-step instructions on how to get started with Docker. This workshop shows you how to:

- Build and run an image as a container.
- Share images using Docker Hub.
- Deploy Docker applications using multiple containers with a database.
- Run applications using Docker Compose.

https://docs.docker.com/get-started/workshop/

#### Bridge network in docker

A bridge network in Docker is a default network type that allows containers to communicate with each other while being isolated from external networks. It acts as a private internal network for the containers.

How it helps:
- Container-to-container communication: Containers can easily communicate via their container names, without needing to expose ports to the host.
- Isolation: Containers are isolated from the host machine and other networks, providing better security.
- Custom control: You can create custom bridge networks to define which containers should be able to communicate.

Issue with putting everything on the same network:
- Security risk: All services can communicate with each other, which may expose services unnecessarily to potential attacks.
- Resource contention: Containers might compete for resources like bandwidth, affecting performance.
- Complexity: Troubleshooting network issues becomes harder as the network grows with more services.

Best practices:
- Use multiple networks: Isolate services that don’t need to communicate (e.g., frontend and backend).
- Leverage network modes: Use bridge for inter-container communication and host for direct network access when needed.
- Apply network policies: Use firewall rules or Docker network security policies to limit communication.
- Use depends_on cautiously: It handles service startup order but doesn’t wait for dependencies to be fully ready.
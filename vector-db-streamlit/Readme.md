# Vector Database with Streamlit UI and Weaviate

This project sets up a vector database using Weaviate and a Streamlit web UI for inserting and querying questions. The application allows users to input questions and retrieves similar questions based on embeddings.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)

## Project Structure
```
project-root/
│
├── docker-compose.yaml
│
├── streamlit/                # Folder for Streamlit code
│   ├── Dockerfile
│   └── app.py
│
├── jupyterlab/               # Folder for JupyterLab code
│   ├── Dockerfile
│
└── notebooks/                # Folder for Jupyter Notebooks
|   └── your_notebook.ipynb   # Example notebook to connect to Weaviate
└── .env                     # Environment variables for Weaviate connection
```
## Prerequisites

- [Docker](https://www.docker.com/get-started) installed on your machine
- [Docker Compose](https://docs.docker.com/compose/) installed

## Getting Started

Follow these steps to set up and run the application locally.

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd vector-db-streamlit
   ```

2. **Build and run the Docker containers**

```bash
docker-compose up --build
```

3. **Streamlit UI & Jupyterlab**

You’ll be able to access:
- Streamlit at http://localhost:8501
- JupyterLab at http://localhost:8888 (No token required)

4. **Usage**

- Access the Streamlit Application:
Open your web browser and go to http://localhost:8501.
Use the "Insert Question" tab to input questions.
Switch to the "View Questions" tab to see all inserted questions.

- Access JupyterLab:
Open your web browser and go to http://localhost:8888.
Use this interface to run analyses and connect to the Weaviate database.

- Inserting Questions:
When you first submit a question, the application will retrieve similar questions that have been previously inserted.
If you decide to submit a different question after viewing similar ones, it will insert the new question into the database.


5. **Project Details**
- Vector Database: The application uses Weaviate to store and retrieve questions based on embeddings.
- Streamlit UI: A simple and interactive web interface for users to interact with the vector database.

6. **Technologies Used**
- Python
- Streamlit
- Docker
- Weaviate
- Sentence Transformers

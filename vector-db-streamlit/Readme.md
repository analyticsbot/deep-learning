# Vector Database with Streamlit UI and Weaviate

This project sets up a vector database using Weaviate and a Streamlit web UI for inserting and querying questions. The application allows users to input questions and retrieves similar questions based on embeddings.

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
- In the input field, enter your question.
- The application will display questions that are the same but may be phrased differently.

5. **Project Details**
- Vector Database: The application uses Weaviate to store and retrieve questions based on embeddings.
- Streamlit UI: A simple and interactive web interface for users to interact with the vector database.

6. **Technologies Used**
- Python
- Streamlit
- Docker
- Weaviate
- Sentence Transformers

# Vector Database with Streamlit UI

This project sets up a vector database and a Streamlit web UI for inserting and querying questions. The application allows users to input questions and retrieves the same questions (even if phrased differently) from the database.

## Project Structure
vector-db-streamlit/
│
├── docker-compose.yml      # Ensure this file has the correct content
├── streamlit/
│   ├── app.py
│   ├── Dockerfile
│   └── requirements.txt
└── db/
    └── questions.db



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

3. **Access the Streamlit UI**

Open your web browser and navigate to http://localhost:8501 to access the Streamlit UI.

4. **Usage**
- In the input field, enter your question.
- The application will display questions that are the same but may be phrased differently.

5. **Project Details**
- Vector Database: The application uses an SQLite database to store questions persistently.
- Streamlit UI: A simple and interactive web interface for users to interact with the vector database.

6. **Technologies Used**
- Python
- Streamlit
- Docker
- SQLite

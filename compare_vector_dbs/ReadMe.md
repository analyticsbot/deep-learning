# Vector Database Performance Tester

This project is a solution to measure the performance of various vector databases, including LanceDB and Qdrant. It provides a user-friendly interface for testing the insertion and querying functionalities of different databases.

## Features

- **User Interface**: Built with Streamlit or Gradio to facilitate interactions.
- **Database Integration**: Supports multiple vector databases like LanceDB and Qdrant.
- **Insert and Query Testing**: Users can specify how many rows to insert and query.
- **Docker Support**: Each database runs in its own Docker container, managed by Docker Compose.
- **Dynamic Configuration**: Add new databases easily by providing their Docker Hub path.

## Requirements

- Python 3.9 or higher
- Docker
- Docker Compose

## Project Structure
```
vector-db-performance-tester/
│
├── Dockerfile.lancedb          # Custom Dockerfile for LanceDB
├── docker-compose.yml          # Docker Compose to manage both LanceDB and Qdrant
├── start_lancedb.py           # Script to start the LanceDB service
├── app.py                      # Streamlit app to handle the UI
├── databases/                  # Folder for insert and query scripts
│   ├── lancedb_insert.py
│   ├── lancedb_query.py
│   ├── qdrant_insert.py
│   ├── qdrant_query.py
└── lancedb_data/              # Persisted data directory
```

## Getting Started

1. **Clone the Repository**
   ```bash
   git clone <this-repo-url>
   cd compare_vector_dbs
    ```

2. **Build and Start Containers**: Ensure you have Docker and Docker Compose installed. Run the following command to build and start the containers

```bash
docker-compose up --build -d
```

3. **Access the User Interface**
Open your web browser and navigate to http://localhost:8501 (or the relevant port if using Gradio) to access the application.

4. **Usage**
- Inserting Data:

    -- Specify the number of rows you want to insert into the selected database.
    -- The application will use a publicly available text corpus to insert data.

- Querying Data:

    -- Specify the number of rows you want to query.
    -- You can also define the embedding length for the queries.

- Adding New Databases:

    -- To add a new vector database, provide the Docker Hub path through the UI.

- View Available Databases:

    -- The UI will display a list of currently available databases for use.

5. **Running Tests**
To measure the performance of the databases, follow the instructions in the UI to run insert and query tests. The results will be displayed directly on the interface.

6. **Contributing**
Contributions are welcome! If you have suggestions or improvements, please create a pull request or open an issue.
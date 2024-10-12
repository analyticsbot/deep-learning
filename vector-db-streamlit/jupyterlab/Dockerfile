# Use an official JupyterLab base image
FROM jupyter/base-notebook:latest

# Install required packages as root
USER root  
# Switch to root to install packages

RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Switch back to the jovyan user
USER jovyan
COPY requirements.txt .
# Install Python packages
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port for JupyterLab
EXPOSE 8888

# Start JupyterLab
CMD ["start-notebook.sh", "--NotebookApp.token=''", "--NotebookApp.allow_origin='*'"]

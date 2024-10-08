# Use the official Python base image with JupyterLab support
FROM python:3.8-slim

# Set the working directory
WORKDIR /app

# Install necessary dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install JupyterLab and other tools for running notebooks
RUN pip install jupyterlab

# Copy the project files into the container
COPY . .

# Expose port 8888 for JupyterLab
EXPOSE 8888

# Command to start JupyterLab
CMD ["jupyter-lab", "--ip=0.0.0.0", "--no-browser", "--allow-root", "--NotebookApp.token=''"]
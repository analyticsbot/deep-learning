import os
import random
from datetime import timedelta

import git
import torch
import torch.nn as nn
import torch.optim as optim
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms

# Constants
BATCH_SIZE = 32
NUM_EPOCHS = 5
MODEL_PATH = "./models/pytorch_model.pth"


# Download and load dataset
def load_data():
    transform = transforms.Compose(
        [
            transforms.Resize((150, 150)),
            transforms.ToTensor(),
        ]
    )
    dataset = datasets.ImageFolder(root="data/CatsAndDogs", transform=transform)

    # 80% train, 20% validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    return train_dataset, val_dataset


# Randomly select a proportion of the train dataset (0.7 - 1.0)
def get_random_subset(train_dataset):
    subset_size = random.uniform(0.7, 1.0)
    train_subset_size = int(subset_size * len(train_dataset))
    return random_split(train_dataset, [train_subset_size, len(train_dataset) - train_subset_size])[
        0
    ]


# Model architecture
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 75 * 75, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 75 * 75)
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


# Train the model
def train_model():
    train_dataset, val_dataset = load_data()
    train_subset = get_random_subset(train_dataset)

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Model, criterion, optimizer
    model = SimpleCNN()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            labels = labels.unsqueeze(1).float()  # BCELoss expects labels of shape [batch_size, 1]
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {running_loss/len(train_loader)}")

    # Save the trained model
    torch.save(model.state_dict(), MODEL_PATH)

    # Commit to GitHub
    repo = git.Repo(".")
    repo.git.add(A=True)
    repo.index.commit(f"Trained model at epoch {NUM_EPOCHS}")
    repo.git.push()


# Define the Airflow DAG
default_args = {
    "owner": "airflow",
    "start_date": days_ago(1),
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

dag = DAG(
    dag_id="train_cat_vs_dog_pytorch_model",
    default_args=default_args,
    schedule_interval="*/5 * * * *",  # Every 5 minutes
)

train_task = PythonOperator(
    task_id="train_pytorch_model_task",
    python_callable=train_model,
    dag=dag,
)

import torch
import torch.nn as nn
from kafka import KafkaConsumer, KafkaProducer
from PIL import Image
from torchvision import transforms


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


MODEL_PATH = "./models/pytorch_model.pth"

# Load trained model
model = SimpleCNN()
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()


# Prediction function
def predict_image(image_path):
    transform = transforms.Compose([transforms.Resize((150, 150)), transforms.ToTensor()])

    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        return "Dog" if output.item() > 0.5 else "Cat"


# Kafka setup
consumer = KafkaConsumer("prediction_topic", bootstrap_servers="localhost:9092")
producer = KafkaProducer(bootstrap_servers="localhost:9092")

for msg in consumer:
    random_number = int(msg.value.decode("utf-8"))
    image_path = f"./data/test_images/{random_number}.jpg"
    prediction = predict_image(image_path)

    producer.send("prediction_result", prediction.encode("utf-8"))
    producer.flush()

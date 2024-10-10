from kafka import KafkaConsumer, KafkaProducer
import torch
from torchvision import transforms
from PIL import Image

MODEL_PATH = '/path/to/saved/pytorch_model.pth'

# Load trained model
model = SimpleCNN()
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# Prediction function
def predict_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor()
    ])
    
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(image)
        return "Dog" if output.item() > 0.5 else "Cat"

# Kafka setup
consumer = KafkaConsumer('prediction_topic', bootstrap_servers='localhost:9092')
producer = KafkaProducer(bootstrap_servers='localhost:9092')

for msg in consumer:
    random_number = int(msg.value.decode('utf-8'))
    image_path = f"/path/to/test_images/{random_number}.jpg"
    prediction = predict_image(image_path)

    producer.send('prediction_result', prediction.encode('utf-8'))
    producer.flush()

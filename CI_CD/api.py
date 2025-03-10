import tensorflow as tf
from kafka import KafkaConsumer, KafkaProducer

MODEL_PATH = "./models/tensorflow_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)


def predict_image(image_path):
    # Load and preprocess image
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(150, 150))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = tf.expand_dims(img_array, 0)

    # Prediction
    return "Dog" if model.predict(img_array)[0][0] > 0.5 else "Cat"


# Kafka Consumer setup
consumer = KafkaConsumer("prediction_topic", bootstrap_servers="localhost:9092")
producer = KafkaProducer(bootstrap_servers="localhost:9092")

for msg in consumer:
    random_number = int(msg.value.decode("utf-8"))
    image_path = f"./data/test_images/{random_number}.jpg"
    result = predict_image(image_path)

    producer.send("prediction_result", result.encode("utf-8"))
    producer.flush()

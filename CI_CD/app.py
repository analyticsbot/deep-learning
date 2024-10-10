import streamlit as st
import requests
import kafka
import random

# Kafka Producer setup
from kafka import KafkaProducer

producer = KafkaProducer(bootstrap_servers='localhost:9092')

def predict(random_number):
    # Send data to Kafka
    producer.send('prediction_topic', str(random_number).encode('utf-8'))
    producer.flush()

    # Kafka Consumer to listen for predictions
    consumer = kafka.KafkaConsumer('prediction_result', bootstrap_servers='localhost:9092')
    for msg in consumer:
        prediction = msg.value.decode('utf-8')
        return prediction

st.title("Dog vs Cat Prediction App")

# Ask user for a random number to select an image
random_number = st.number_input("Enter a random number:", min_value=0, max_value=1000, value=42)

if st.button('Predict'):
    result = predict(random_number)
    st.write(f'Prediction: {result}')

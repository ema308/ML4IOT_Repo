# Voice-Activated Environmental Monitoring System

## Project Overview:
A Raspberry Pi–based system that uses a custom voice interface to trigger real-time humidity and temperature data collection. The system listens for the keywords “go” and “stop”, activates sensors accordingly, stores results as time series in Redis, and exposes them through a REST API.
This project combines embedded hardware, keyword spotting, time-series data processing, and backend API development.

## Project structure 
Code is organized into Homeworks in the following way: 
- HW1: basics (analysis of the VAD dataset, Voice User Interface, connection to Redis)
- HW2: training (training a model able to perform voice recognition)
- HW3: REST API for Redis storage (made of publisher, subscriber, client, and server)
```
src/
├── HW1_Group8/
│   ├── RedisConnection(ex1).py
│   ├── VUI(ex2).py
│   └── Audios_analysis(HW1_ ex2.1).ipynb
├── HW2_Group8/
│   ├── Redis&VUI(ex1).py
│   └── training.ipynb
├── HW3_Group8/
│   ├── publisher.py
│   ├── rest_client.ipynb
│   ├── rest_server.ipynb
│   └── subscriber.ipynb

```
## How does it work 

HW1: Voice Activation
- The system continuously listens for speech and detects voice activity
- use of Spectrogram techniques for analysis
- Voice User Interface (VUI) that starts or stops environmental data collection based on keyword recognition.
- Humidity and temperature readings are captured at regular intervals.
- Sensor data is timestamped and pushed into Redis using time-series structures.

HW2: Model training
- A custom classification model was trained using the VAD dataset.
- Audio frames are processed in real time.
- When the model predicts:

  “go” → begin collecting humidity and temperature readings.
  “stop” → stop collecting sensor data.


HW3: Redis Time-Series Storage
- Separate series for: temperature and humidity
- A lightweight REST API exposes the stored sensor data.
- The API automatically updates with new Redis entries, allowing external clients (apps, dashboards, scripts) to consume the data in real time.

### Requirements 
- Raspberry Pi 

- Microphone

- DHT-series or similar humidity/temperature sensor

- Redis server

- Python 3.8+

## Results
The results in terms of accuracy, size, and latency are the following: <br>
- Accuracy: 99.5%
- Model Size: 40.9 kB
- Latency: 27.ms

## Acknoledgments 
This project was part of the final exam for the "Machine Learning for IOT" course at Politecnico di Torino

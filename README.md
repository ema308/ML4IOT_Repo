**Voice-Activated Environmental Monitoring System**

A Raspberry Pi–based system that uses a custom voice interface to trigger real-time humidity and temperature data collection. The system listens for the keywords “go” and “stop”, activates sensors accordingly, stores results as time series in Redis, and exposes them through a REST API.


Project Overview:
This project combines embedded hardware, keyword spotting, time-series data processing, and backend API development.
Using a Raspberry Pi equipped with:

- Microphone (voice recorder)

- Humidity sensor

- Temperature sensor


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

# Voice-Activated Environmental Monitoring System

## Project Overview:
A Raspberry Pi–based system that uses a custom voice interface to trigger real-time humidity and temperature data collection. The system listens for the keywords “up” and “down”, activates sensors accordingly, stores results as time series in Redis, and exposes them through a REST API.
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

  “up” → begin collecting humidity and temperature readings.
  “down” → stop collecting sensor data.


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

## Relevant classes and methods

### Voice Activity Detection (VAD) class
The VAD class takes as inputs audio files from the VAD dataset, it extracts relevant features and uses them to determine whether an audio file is silent or not using the ```is_silent``` method. 
The method works in the following way:

**Computation of the Spectrogram**
A Spectrogram is a visual representation of how frequencies change over time (color intensity = loudness). It maps on the x-axis the time, and on the y-axis the frequency variation. It is often computed using the FFT, which is an algorithm that computes the frequency components of a signal in a very fast way. 

In particular:
- ```sampling_rate```: conventionally set at 1600 kHz (one sample of the audio every second)
- ```frame_length_in_s```: temporal window in which the FFT is computed, conventionally set at 0.04 s
- ```frame_step_in_s```: temporal distance between a window and the next one, conventionally set at 0.01 s (overlap between windows of 75%)

**Detect non-silent frames**
The Spectrogram is converted into dB and the average energy per frame is computed. To detect non-silent audios:

- average energy frames are compared with ```db.Tresh```, if the energy is greater than the treshold the frame is loud enough
- loud enough frames are summed up in order to estimate speech duration
- the estimated speech duration is compared with ```duration_tresh```, if the speech is grater than the treshold then it can be recognized as valid and flagged with 0 (approved)

**Accuracy constraint**
We can then choose a search space for each of the previously mentioned parameters a perform a grid search to find the optimal values. In particular, we want to compute the accuracy on the VAD dataset, by checking how many audios are correctly classified as silent.
We then print the top 5 parameters combination sorted by the highest accuracy values. 



## Results
The results in terms of accuracy, size, and latency are the following: <br>
- Accuracy: 99.5%
- Model Size: 40.9 kB
- Latency: 27.ms

## Acknoledgments 
This project was part of the final exam for the "Machine Learning for IOT" course at Politecnico di Torino

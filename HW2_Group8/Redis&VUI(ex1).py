#!/usr/bin/env python3
import sounddevice as sd
from time import time
from scipy.io.wavfile import write
import os
import adafruit_dht
import uuid
import time
from datetime import datetime
from board import D4
import numpy as np
import tensorflow as tf
from scipy import signal
import redis
import zipfile



"""Connect to Redis"""
#in the command line: python directory --host your host --port your port --user default --password your password

import argparse

# Create an argument parser
parser = argparse.ArgumentParser(description="Redis Cloud Connection Script")

# Define the command-line arguments
parser.add_argument('--host', type=str, required=True, help='The Redis Cloud host.')
parser.add_argument('--port', type=int, required=True, help='The Redis Cloud port.')
parser.add_argument('--user', type=str, required=True, help='The Redis Cloud username.')
parser.add_argument('--password', type=str, required=True, help='The Redis Cloud password.')

# Parse the arguments
args = parser.parse_args()

# Use the parsed arguments to connect to Redis
redis_client = redis.Redis(host=args.host, port=args.port, username=args.user, password=args.password)
"""create the time-series"""
# Set the retention of "temperature" and "humidity" to 30 days
thirty_day_in_ms = 30 * 24 * 60 * 60 * 1000
try:
    redis_client.ts().create("temperature", retention_msecs=thirty_day_in_ms, uncompressed=False, chunk_size=128,)
except redis.ResponseError:
    pass

try:
    redis_client.ts().create("humidity", retention_msecs=thirty_day_in_ms, uncompressed=False, chunk_size=128,)
except redis.ResponseError:
    pass

#constants
SAMPLE_RATE = 48000
TARGET_SAMPLE_RATE = 16000
DATA_COLLECTION_INTERVAL = 2
AUDIO_CHUNCK_DURATION = 1
UP_KEYWORD_THRESHOLD = 0.99
DOWN_KEYWORD_THRESHOLD = 0.99

"""return current time in seconds"""
def get_current_time():

    return time.time()

"""VAD"""
class Spectrogram():
    def __init__(self, sampling_rate, frame_length_in_s, frame_step_in_s):
        self.frame_length = int(frame_length_in_s * sampling_rate)
        self.frame_step = int(frame_step_in_s * sampling_rate)

    def get_spectrogram(self, audio):
        stft = tf.signal.stft(
            audio, 
            frame_length=self.frame_length,
            frame_step=self.frame_step,
            fft_length=self.frame_length
        )
        spectrogram = tf.abs(stft)

        return spectrogram

    def get_spectrogram_and_label(self, audio, label):
        spectrogram = self.get_spectrogram(audio)

        return spectrogram, label

class MelSpectrogram():
    def __init__(
        self, 
        sampling_rate,
        frame_length_in_s,
        frame_step_in_s,
        num_mel_bins,
        lower_frequency,
        upper_frequency
    ):
        self.spectrogram_processor = Spectrogram(sampling_rate, frame_length_in_s, frame_step_in_s)
        num_spectrogram_bins = self.spectrogram_processor.frame_length // 2 + 1

        self.linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=num_mel_bins,
            num_spectrogram_bins=num_spectrogram_bins,
            sample_rate=sampling_rate,
            lower_edge_hertz=lower_frequency,
            upper_edge_hertz=upper_frequency
        )

    def get_mel_spec(self, audio):
        spectrogram = self.spectrogram_processor.get_spectrogram(audio)
        mel_spectrogram = tf.matmul(spectrogram, self.linear_to_mel_weight_matrix)
        log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)

        return log_mel_spectrogram

    def get_mel_spec_and_label(self, audio, label):
        log_mel_spectrogram = self.get_mel_spec(audio)

        return log_mel_spectrogram, label

class MFCC():
    def __init__(
        self, 
        sampling_rate,
        frame_length_in_s,
        frame_step_in_s,
        num_mel_bins,
        lower_frequency,
        upper_frequency,
        num_coefficients
    ):
        self.mel_spec_processor = MelSpectrogram(
            sampling_rate, frame_length_in_s, frame_step_in_s, num_mel_bins, lower_frequency, upper_frequency
        )
        self.num_coefficients = num_coefficients

    def get_mfccs(self, audio):
        log_mel_spectrogram = self.mel_spec_processor.get_mel_spec(audio)
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
        mfccs = mfccs[..., :self.num_coefficients]

        return mfccs

    def get_mfccs_and_label(self, audio, label):
        mfccs = self.get_mfccs(audio)

        return mfccs, label

class VAD():
    def __init__(
        self,
        sampling_rate,
        frame_length_in_s,
        frame_step_in_s,
        dBthres,
        duration_thres,
    ):
        self.frame_length_in_s = frame_length_in_s
        self.frame_step_in_s = frame_step_in_s
        self.spec_processor = Spectrogram(
            sampling_rate, frame_length_in_s, frame_step_in_s,
        )
        self.dBthres = dBthres
        self.duration_thres = duration_thres

    def is_silence(self, audio):
        spectrogram = self.spec_processor.get_spectrogram(audio)
        
        dB = 20 * tf.math.log(spectrogram + 1.e-6)
        energy = tf.math.reduce_mean(dB, axis=1)
        min_energy = tf.reduce_min(energy)

        rel_energy = energy - min_energy
        non_silence = rel_energy > self.dBthres
        non_silence_frames = tf.math.reduce_sum(tf.cast(non_silence, tf.float32))
        non_silence_duration = self.frame_length_in_s + self.frame_step_in_s * (non_silence_frames - 1)

        if non_silence_duration > self.duration_thres:
            return 0 
            
        else:
            return 1 

"""pre processing audio"""
class AudioProcessor:
    def __init__(self, target_rate):
        self.target_rate = target_rate
        
    def process(self, audio_data, samplerate):
        # Cast to float32
        audio_tensor = tf.cast(audio_data, tf.float32)

        # Downsample to 16kHz
        downsampling_factor = int(samplerate / self.target_rate)
        audio_resampled = signal.resample_poly(audio_tensor, up=1, down=downsampling_factor)

        # Convert to tensor and squeeze 
        audio_tensor = tf.convert_to_tensor(audio_resampled)
        audio_tensor = tf.squeeze(audio_tensor)
        
        return audio_tensor

class Normalization():
    def __init__(self, bit_depth):
        self.max_range = bit_depth.max

    def normalize_audio(self, audio):
        #already casted to 32b
        audio_normalized = audio / self.max_range

        return audio_normalized

    def normalize(self, audio, label):
        audio_normalized = self.normalize_audio(audio)

        return audio_normalized, label


"""class Voice User Interface"""
class VUI:
    #initialize the VUI class
    def __init__(self, vad_instance, processor_instance, normalizer_instance,  tflite_model_path,
                 samplerate=SAMPLE_RATE, channels=1, dtype='int16'):
        
        self.samplerate = samplerate
        self.channels = channels
        self.dtype = dtype
        self.audio_buffer = None
        self.vad = vad_instance #Instance of the VAD class to detect silence or non-silence.
        self.processor = processor_instance #Instance of the AudioProcessor class for audio processing.
        self.normalizer = normalizer_instance #Instance of the Normalizer class for audio normalization.
        self.data_collection_enabled = 0  # Start with data collection disabled
        self.toggle_lock_time = 5  # Minimum time in seconds before toggling state
        self.last_toggle_time = 0  # Time when the last toggle occurred
        self.last_state = None

        
        # Data collection setup
        self.mac_address = hex(uuid.getnode())
        self.dht_device = adafruit_dht.DHT11(D4)

        if tflite_model_path.endswith('.zip'):
            with zipfile.ZipFile(tflite_model_path, 'r') as fp:
                fp.extractall('/tmp/')
                model_filename = fp.namelist()[0]
                tflite_model_path = '/tmp/' + model_filename
            
        # Load TFLite model
        self.interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Add MFCC instance
        self.mfcc_processor = MFCC(
            sampling_rate=TARGET_SAMPLE_RATE,
            frame_length_in_s=0.032,
            frame_step_in_s=0.016,
            num_mel_bins=21,
            lower_frequency=40,
            upper_frequency=5000,
            num_coefficients=14  # Ensure compatibility with the model
        )


    def _audio_callback(self, indata, frames, time, status):
        """
        Audio callback function that processes recorded audio, checks for silence,
        and performs model inference if audio is non-silent.
        """

        #Flatten the incoming audio data
        self.audio_buffer = indata.squeeze()

        #Process the raw audio (resample and convert)
        processed_audio = self.processor.process(self.audio_buffer, self.samplerate)

        #Normalize the processed audio
        normalized_audio = self.normalizer.normalize_audio(processed_audio)

        #Check for silence using VAD
        is_silence = self.vad.is_silence(normalized_audio)
        current_time = get_current_time()

        # If audio is non-silent, check if enough time has passed to toggle data collection
        if is_silence == 0:  # Non-silence detected
            #If audio is non-silent, process the audio for keyword detection
            # Compute MFCC features from the normalized audio
            mfcc_features = self.mfcc_processor.get_mfccs(normalized_audio)

            # Step 7: Expand dimensions to match the model's expected input shape
            mfcc_features = tf.expand_dims(mfcc_features, axis=0)  # Add batch dimension
            mfcc_features = tf.expand_dims(mfcc_features, axis=-1)  # Add channel dimension


            #check if 5 seconds passed
            if current_time - self.last_toggle_time >= self.toggle_lock_time:  # 5 seconds lock
                self.interpreter.set_tensor(self.input_details[0]['index'], mfcc_features.numpy())
                self.interpreter.invoke()

                # Get the output probabilities
                predictions = self.interpreter.get_tensor(self.output_details[0]['index'])
                down_prob, up_prob = predictions[0]

                if up_prob > UP_KEYWORD_THRESHOLD:
                    if not self.data_collection_enabled:
                        self.data_collection_enabled = True
                        print("Data collection enabled.")
                elif down_prob > DOWN_KEYWORD_THRESHOLD:
                    if self.data_collection_enabled:
                        self.data_collection_enabled = False
                        print("Data collection disabled.")
                
                self.last_toggle_time = current_time

        else:
            pass  # Silence detected, do nothing


    #Collect data from the sensor and print it.
    def collect_data(self):

        try:
            timestamp = int(time.time()*1000) #get timestamp in ms
            formatted_time = datetime.fromtimestamp(timestamp / 1000).strftime('%Y-%m-%d %H:%M:%S.%f')
            temperature = self.dht_device.temperature
            humidity = self.dht_device.humidity

            print(f'{formatted_time}-{self.mac_address}:temperature = {temperature}')
            print(f'{formatted_time}-{self.mac_address}:humidity = {humidity}')

            #Add temperature and humidity data to Redis time series
            redis_client.ts().add('temperature', timestamp, temperature)
            redis_client.ts().add('humidity', timestamp, humidity)

        except Exception as e:
            print(f'Sensor failure: {e}')
            self.dht_device.exit()
            self.dht_device = adafruit_dht.DHT11(D4)  # Reconnect to the sensor

    #Run the VUI loop with continuous audio streaming and processing.
    def run(self, duration=AUDIO_CHUNCK_DURATION):

        frames = int(self.samplerate * duration) #duration : Duration of each audio chunk in seconds (1s)
        with sd.InputStream(device=1, channels=self.channels, dtype=self.dtype,
                            samplerate=self.samplerate, blocksize=frames,
                            callback=self._audio_callback):
            print("VUI is running")
            while True:
                if self.data_collection_enabled:
                    self.collect_data()
                    time.sleep(2)  



#usage
if __name__ == "__main__":

    processor_instance = AudioProcessor(TARGET_SAMPLE_RATE) #initialize instance of the AudioProcessor
    normalizer_instance = Normalization(tf.int16) #initialize instance of the normalization class
    #initialize VAD instance with the parameter of ex. 2.1
    VAD_instance = VAD(sampling_rate=TARGET_SAMPLE_RATE, frame_length_in_s=0.032, frame_step_in_s=0.016, dBthres=10, duration_thres=0.150)
    # Path to the zipped model file
    tflite_model_path = "/home/ml4iot/workdir/model8.tflite.zip"
    #instance of the VUI class
    recorder_instance = VUI(VAD_instance, processor_instance, normalizer_instance, tflite_model_path)
    #start recording data
    recorder_instance.run(AUDIO_CHUNCK_DURATION)

    

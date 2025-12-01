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

#constants
SAMPLE_RATE = 48000
TARGET_SAMPLE_RATE = 16000
DATA_COLLECTION_INTERVAL = 2
AUDIO_CHUNCK_DURATION = 1


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
    def __init__(self, vad_instance, processor_instance, normalizer_instance,
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

        # Data collection setup
        self.mac_address = hex(uuid.getnode())
        self.dht_device = adafruit_dht.DHT11(D4)

    #Audio callback function that processes recorded audio and checks for silence.
    def _audio_callback(self, indata, frames, time, status):

        # Flatten the incoming audio data
        self.audio_buffer = indata.squeeze()

        # Transform and normalize audio
        processed_audio = self.processor.process(self.audio_buffer, self.samplerate)
        normalized_audio = self.normalizer.normalize_audio(processed_audio)

        # Check for silence using VAD
        is_silence = self.vad.is_silence(normalized_audio)
        current_time = get_current_time()

        if is_silence == 0:  # Non-silence detected
            if current_time - self.last_toggle_time >= self.toggle_lock_time: #if more than 5 seconds passed from current time
                self.data_collection_enabled = not self.data_collection_enabled #change data collection state
                #change name of the state
                state = "enabled" if self.data_collection_enabled else "disabled"
                print(f"Data collection {state}.")
                #update last toggle time
                self.last_toggle_time = current_time
        else: 
            pass
        # Collect data if enabled
        """ NON QUA!!!
        if self.data_collection_enabled:
            self.collect_data()
        """

    #Collect data from the sensor and print it.
    def collect_data(self):

        try:
            timestamp = time.time()
            formatted_time = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S.%f')
            temperature = self.dht_device.temperature
            humidity = self.dht_device.humidity

            print(f'{formatted_time}-{self.mac_address}:temperature = {temperature}')
            print(f'{formatted_time}-{self.mac_address}:humidity = {humidity}')

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
                #corretto!!!
                if self.data_collection_enabled:
                    self.collect_data()
                    time.sleep(2)  



#usage
if __name__ == "__main__":

    processor_instance = AudioProcessor(TARGET_SAMPLE_RATE) #initialize instance of the AudioProcessor
   
    normalizer_instance = Normalization(tf.int16) #initialize instance of the normalization class
    #initialize VAD instance with the parameter of ex. 2.1
    VAD_instance = VAD(sampling_rate=TARGET_SAMPLE_RATE, frame_length_in_s=0.032, frame_step_in_s=0.016, dBthres=10, duration_thres=0.150)
    #instance of the VUI class
    recorder_instance = VUI(VAD_instance, processor_instance, normalizer_instance)
    #start recording data
    recorder_instance.run(AUDIO_CHUNCK_DURATION)

    

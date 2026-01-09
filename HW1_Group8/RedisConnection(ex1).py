import adafruit_dht
import uuid
import time
from datetime import datetime
from board import D4
import redis
from time import sleep


# Connect to Redis
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



# Test the connection
try:
    is_connected = redis_client.ping()
    print('Redis Connected:', is_connected)


except redis.AuthenticationError:
    print("Authentication Error: Please check your Redis Cloud credentials.")
except Exception as e:
    print(f"Error: {e}")


"""creating the time series for temperature and humidity"""
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


# Create timeseries for averages, minimums, and maximums of temperature
#set retention period to 365 days
oneY_in_ms = 365 * 24 * 60 * 60 * 1000
try:
    redis_client.ts().create('temperature_avg', retention_msecs = oneY_in_ms, chunk_size=128,)
    redis_client.ts().createrule('temperature', 'temperature_avg', 'avg', bucket_size_msec=3600000)
except redis.ResponseError:
    pass

try:
    redis_client.ts().create('temperature_min', retention_msecs = oneY_in_ms, chunk_size=128,)
    redis_client.ts().createrule('temperature', 'temperature_min', 'min', bucket_size_msec=3600000)
except redis.ResponseError:
    pass

try:
    redis_client.ts().create('temperature_max', retention_msecs = oneY_in_ms, chunk_size=128,)
    redis_client.ts().createrule('temperature', 'temperature_max', 'max', bucket_size_msec=3600000)
except redis.ResponseError:
    pass

# Create timeseries for averages, minimums, and maximums of humidity
#set retention period to 365 days
try:
    redis_client.ts().create('humidity_avg', retention_msecs = oneY_in_ms, chunk_size=128,)
    redis_client.ts().createrule('humidity', 'humidity_avg', 'avg', bucket_size_msec=3600000)
except redis.ResponseError:
    pass


try:
    redis_client.ts().create('humidity_min', retention_msecs = oneY_in_ms, chunk_size=128,)
    redis_client.ts().createrule('humidity', 'humidity_min', 'min', bucket_size_msec=3600000)
except redis.ResponseError:
    pass



try:
    redis_client.ts().create('humidity_max', retention_msecs = oneY_in_ms, chunk_size=128,)
    redis_client.ts().createrule('humidity', 'humidity_max', 'max', bucket_size_msec=3600000)
except redis.ResponseError:
    pass


"""loop for data collection"""

mac_address = hex(uuid.getnode())
dht_device = adafruit_dht.DHT11(D4)  # Connect to the specified pin
while True:
    timestamp = int(time.time() * 1000)  # Get current time in milliseconds
    formatted_time = datetime.fromtimestamp(timestamp / 1000).strftime('%Y-%m-%d %H:%M:%S.%f')


    try:
        temperature = dht_device.temperature
        humidity = dht_device.humidity

        print(f'{formatted_time}-{mac_address}:temperature = {temperature}')
        print(f'{formatted_time}-{mac_address}:humidity = {humidity}')

        # Add temperature and humidity data to Redis time series
        redis_client.ts().add('temperature', timestamp, temperature)
        redis_client.ts().add('humidity', timestamp, humidity)
        

    except Exception as e:
        print('Sensor failure:', e)
        dht_device.exit()
        dht_device = adafruit_dht.DHT11(D4)  # New connection to the sensor

    sleep(2)  # Wait for 2 seconds before the next reading




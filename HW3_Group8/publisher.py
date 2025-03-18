import adafruit_dht
import uuid
import time
from datetime import datetime
from board import D4
import json
import paho.mqtt.client as mqtt


# Create a new MQTT client
client = mqtt.Client()

# Connect to the MQTT broker
client.connect('mqtt.eclipseprojects.io', 1883)

mac_address = hex(uuid.getnode())
dht_device = adafruit_dht.DHT11(D4)

while True:
    """data collection"""
    timestamp = time.time()
    formatted_datetime = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S.%f')
    try:
        temperature = dht_device.temperature
        humidity = dht_device.humidity

        print(f'{formatted_datetime} - {mac_address}:temperature = {temperature}')
        print(f'{formatted_datetime} - {mac_address}:humidity = {humidity}')
    except:
        print(f'{formatted_datetime} - sensor failure')
        dht_device.exit()
        dht_device = adafruit_dht.DHT11(D4)


    """json message creation"""
    my_dict = {
    'mac_address': mac_address,
    'timestamp': timestamp,
    'temperature': temperature,
    'humidity': humidity,
    }

    """
    # Encode dict to JSON file
    #ridondante
    with open('message.json', 'w') as fp:
        json.dump(my_dict, fp)
    """
    # Encode a dict to JSON string
    message = json.dumps(my_dict)
    
    # Publish a message to a topic. Use your studend ID as topic
    client.publish('s333550', message) #aggiungere stampa messaggio

    time.sleep(2)
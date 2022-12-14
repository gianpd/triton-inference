import redis
import msgpack
import cv2
import numpy as np
from queue import Queue

from typing import Optional, Union

import requests
import io
import paho.mqtt.client as mqtt
import logging_utils
logger = logging_utils.get_logger('TritonClient')

from config import *

FPS = 5
RATE = 1 / FPS
KEY = 'NJ'


class NJRedisClient:
    def __init__(self, 
                 host: Optional[str] = 'localhost', 
                 port: Optional[int] = 6379, 
                 key: Optional[str] = KEY,
                 cnt_key: Optional[str] = 'rock_cnt',
                 unix_socket_path: Optional[str] = None, 
                 db: Optional[int] = 0, 
                 fps: Optional[int] = 8):
        self._host = host
        self._port = port
        self._key = key
        self._cntKey = cnt_key
        self._db = db
        self._fps = fps
        self._msgpack = None
        self._nj_redis = redis.Redis(host=host, port=port, db=db, unix_socket_path=unix_socket_path)
        
    @property
    def get_cnt(self):
        return self._nj_redis.get(self._cntKey)
        
        
    
    @property
    def get_msg(self):
        payload = self._nj_redis.get(self._key)
        if payload is None:
            return payload
        return msgpack.unpackb(payload, raw=False)
    
    def get_np_img(self, buffer_img, height, width) -> np.array:
        # return np.frombuffer(buffer_img, dtype=np.uint8).reshape((height, width, 3))
        # img = np.asarray(bytearray(buffer_img), dtype='uint8')
        # img = np.frombuffer(buffer_img, dtype=np.uint8)
        # img = cv2.cvtColor(cv2.imdecode(img, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
        img = np.frombuffer(buffer_img, dtype=np.uint8).reshape((height, width, 3))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
        

    
class NjMQTTPub():
    def __init__(self, clientID, broker="mqtt.eclipseprojects.io", port=1883, topic="", qos=1):
        self.clientID_ = clientID
        self.broker_ = broker
        self.port_ = port
        self.mqtt_client = mqtt.Client(self.clientID_, clean_session=False)
        self.topic_ = topic
        self.qos_ = qos
        self.queue_ = Queue()

        self.mqtt_client.on_connect = self.OnConnect
        self.mqtt_client.on_disconnect = self.OnDisconnect
        logger.info(f"User {self.clientID_} initialized.")

    def OnConnect(self, mqtt_client, userdata, flags, rc):
        """ myOnConnect function called by on_connect callback:
        Called upon connection to the broker. Everything goes well if rc == 0
        otherwise we have some connection issues with the broker. If so it is
        printed in the terminal and the notify() method of the notifier is
        called so that an appropriate action can be taken.
        Args:
            mqtt_client (:obj: MQTT.Client): client instance of the callback
            userdata (str): user data as set in Client (not used here)
            flags (int): flag to notify if the user's session is still
                available (not used here)
            rc (int): result code
        """
        errMsg = ""

        if rc == 0:
            logger.info(f"MQTT client {self.clientID_} successfully connected to broker!")
            return str(rc)

        # If we go through this we had a problem with the connection phase
        elif 0 < rc <= 5:
            errMsg = "/!\ " + self.clientID_ + " connection to broker was " \
                                              "refused because of: "
            if rc == 1:
                errMsg.append("the use of an incorrect protocol version!")
            elif rc == 2:
                errMsg.append("the use of an invalid client identifier!")
            elif rc == 3:
                errMsg.append("the server is unavailable!")
            elif rc == 4:
                errMsg.append("the use of a bad username or password!")
            else:
                errMsg.append("it was not authorised!")
        else:
            errMsg = "/!\ " + self.clientID_ + " connection to broker was " \
                                              "refused for unknown reasons!"
        logger.error(errMsg)


    def publish(self, msg):
        """ myPublish:
                Method that makes the MQTT client publish to the the broker a message
                under a specific topic and with a particular QoS, which by default is 2.
                Args:
                    topic (str): topic to which you desire to publish
                    msg (str): message you wish to publish
                    qos (int, optional): desired QoS, default to 2
                """
        logger.info(f"MQTT client {self.clientID_} publishing {msg} with topic {self.topic_}.")
        # publish a message with a certain topic

        self.mqtt_client.publish(self.topic_, msg, self.qos_)
        # self.mqtt_client.loop()

    def OnDisconnect(self, mqtt_client, userdata, rc):
        """ myOnDisconnect function called by on_disconnect callback:
        Can be triggered in one of two cases:
        - in response to a disconnect(): normal case, it was asked
        - in response to an unexpected disconnection: in that case the client
        will try to reconnect
        In both cases we log it.
        Args:
            mqtt_client (:obj: MQTT.Client): client instance of the callback
            userdata (str): user data as set in Client (not used here)
            rc (int): result code
        """
        if rc == 0:
            logger.info(f"MQTT client {self.clientID_} successfully disconnected!")
        else:
            logger.warning(f"Unexpected disconnection of MQTT client {self.clientID_}. "\
                           "Reconnecting right away!")
            # The reconnection is performed automatically by our client since
            # we're using loop_start() so no need to manually tell our client
            # to reconnect.

    def start(self):
        self.mqtt_client.connect(host=self.broker_, port=self.port_)
        self.mqtt_client.loop_start()
        logger.info(f"Client {self.clientID_} connected to the Broker {self.broker_}")

    def stop(self):
        self.mqtt_client.loop_stop()
        self.mqtt_client.disconnect()
        logger.info(f"Client {self.clientID_} disconnected from the Broker")
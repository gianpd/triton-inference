import redis
import msgpack
import cv2
import numpy as np

from typing import Optional

FPS = 5
RATE = 1 / FPS
KEY = 'NJ'

class NJRedisClient:
    def __init__(self, host: Optional[str] = 'localhost', port: Optional[int] = 6379, key: Optional[str] = KEY, unix_socket_path: Optional[str] = None, db: Optional[int] = 0, fps: Optional[int] = 8):
        self._host = host
        self._port = port
        self._key = key
        self._db = db
        self._fps = fps
        self._msgpack = None
        self._nj_redis = redis.Redis(host=host, port=port, db=db, unix_socket_path=unix_socket_path)
    
    @property
    def get_msg(self):
        payload = self._nj_redis.get(self._key)
        if payload is None:
            return payload
        return msgpack.unpackb(payload, raw=False)
    
    def get_np_img(self, buffer_img, height, width) -> np.array:
        # return np.frombuffer(buffer_img, dtype=np.uint8).reshape((height, width, 3))
        # img = np.asarray(bytearray(buffer_img), dtype='uint8')
        img = np.frombuffer(buffer_img, dtype=np.uint8)
        img = cv2.cvtColor(cv2.imdecode(img, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
        return img
        
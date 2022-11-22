"""
{
"data": <bytes> #Byte array RGB ordinato per righe, colonne, canali (Height, Width, 3),
"timestamp": <INT>, # timestamp in nanosecondi del momento di acquisizione del frame
"size": {
"width": <INT>, # numero colonne
"height": <INT> # numero righe
},
"position": { # dati da ultima lettura del GPS, in generale antecedente il momento di acquisizione del frame
	"timestamp": <INT>, # timestamp in nanosecondi ultima lettura del GPS
            "longitude": <FLOAT>,
            "latitude": <FLOAT>,
            "altitude": <FLOAT>,
            "track": <FLOAT>, # angolo rispetto a Nord magnetico
            	"speed": <FLOAT>, # velocità in m/s
"eph": <FLOAT>, # stima dell’errore di posizionamento orizzontale in metri
"eps": <FLOAT>, # stima dell’errore di velocità orizzontale in metri/s
"epd": <FLOAT> # stima dell’errore di track in gradi

},
"exposure": {
	"shutter_speed": <INT>, # Velocità shutter in µs
	"gain": <FLOAT>, # Guadagno in db, o ISO se troviamo conversione
"aperture": <FLOAT>, # Apertura iride
"flocal_length": <FLOAT> # focale della lente
}
}

"""

from datetime import datetime
from pathlib import Path
import string
import random
import redis
import time
import msgpack

import argparse 

parser = argparse.ArgumentParser()
parser.add_argument('--host', type=str, default='key-db-server')
parser.add_argument('--port', type=int, default=6379)
parser.add_argument('--test', action='store_true')
args = parser.parse_args()
print(args)

FPS = 10
RATE = 1 / FPS
KEY = 'NJ'
IMG_BYTES = Path("image_2022-10-12_13-38-31.496364.jpg").read_bytes()


grabber_redis = redis.Redis(host=args.host, port=args.port, db=0)
ascii_letters = string.ascii_letters
N = len(ascii_letters)
start = datetime.now()
STEPS = 1 if args.test else 1000
for i in range(STEPS):
    time.sleep(RATE)
    idx = random.randint(0, N-1)
    _d = ascii_letters[idx:]
    payload = {'data': IMG_BYTES, 'timestamp': 12345, 'd3': _d, 'EXIT': None}
    payload = msgpack.packb(payload, use_bin_type=True)
    grabber_redis.set(KEY, payload)
print(f'GRABBER DONE - elapsed: {(datetime.now() - start).total_seconds()} [s]') # 101 [s]

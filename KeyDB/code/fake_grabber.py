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
#IMG_BYTES = Path("image_2022-10-12_13-38-31.496364.jpg").read_bytes()


grabber_redis = redis.Redis(unix_socket_path='/tmp/docker/keydb.sock')
# ascii_letters = string.ascii_letters
# N = len(ascii_letters)
# start = datetime.now()
# folder = './Single_Test/' if args.test else './batch_prova_test_final/'
# img_paths = list(map(lambda x: x, Path(folder).rglob('*.jpg')))

# print(f'Fake-Grabber> Retrived  {len(img_paths)} test images')

start = datetime.now()
dump_paths = list(Path('./dumps').glob('*.data'))
print(f'Sending {len(dump_paths)} dumps ...')
for dump_path in dump_paths: 
    with open(dump_path, 'rb') as f:
        payload = f.read()
    grabber_redis.set(KEY, payload)
print(f'GRABBER DONE - elapsed: {(datetime.now() - start).total_seconds()} [s]')


# for img_path in img_paths:
#     time.sleep(RATE)
# #     idx = random.randint(0, N-1)
# #     _d = ascii_letters[idx:]
# #     payload = {
# #         'data': img_path.read_bytes(), 
# #         'size': {'width': 2048, 'height': 1500},  
# #         'timestamp': 12345, 
# #         'd3': _d, 
# #         'EXIT': None}
#     payload = msgpack.packb(payload, use_bin_type=True)
#     grabber_redis.set(KEY, payload)
# print(f'GRABBER DONE - elapsed: {(datetime.now() - start).total_seconds()} [s]') # 101 [s]

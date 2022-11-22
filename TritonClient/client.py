import numpy as np
import time
from tritonclient.utils import *
from PIL import Image
import tritonclient.http as httpclient
import redis_utils as rutils
from wasabi import msg


def main():

    # 0. Set up the connection with the server
    client = httpclient.InferenceServerClient(url="localhost:8000")
    # 1. Start the communication channel with the redis server, for receiving the requests to be passed to the models
    rs_client = rutils.NJRedisClient(host='172.17.0.2', port=6379, db=0, key='NJ')
    # 1a. Start the receiving loop: how the loop ends ? (S1:it could be ended when a particular msgpack is received (?))
    while True:
        payload = rs_client.get_msg
        if payload['EXIT']:
            msg.info('Exiting TritonClient main ...')
            break
        img = rs_client.get_np_img(payload['data'])
        msg.good(f'Initial img shape: {img.shape}')
        # preprocess input img
        # img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_LANCZOS4)
        # msg.good(f'After resize {img.shape}')
        # img = img / 255.0
        # img = np.asarray(np.expand_dims(img, axis=0), dtype=np.float32)
        img = np.expand_dims(img, axis=0)
        
        msg.good(f'After normalization {img.dtype, img.shape}')
        input_raw_img = httpclient.InferInput("input_tensor", img.shape, np_to_triton_dtype(img.dtype))
        input_raw_img.set_data_from_numpy(img)
        # msg.good(f'INPUT_RAW_IMG: {input_raw_img}')
        
        # od_outputs  = [
        #     httpclient.InferRequestedOutput("od_scores"),
        #     httpclient.InferRequestedOutput("od_boxes"),
        #     # httpclient.InferRequestedOutput("detection_classes"),
        # ]
        dc_outputs  = [
            httpclient.InferRequestedOutput("dc_scores"),
            # httpclient.InferRequestedOutput("detection_classes"),
        ]
        
        query_response = client.infer(model_name='pipeline',
                                     inputs=[input_raw_img],
                                     outputs=dc_outputs)
        
        scores_dict = query_response.as_numpy("dc_scores")
        #boxes_dict = query_response.as_numpy("od_boxes")
        # classes_dict = query_response.as_numpy("detection_classes")
        msg.good(f'Scores dict:\n {scores_dict}')
        #msg.good(f'Boxes dict:\n {boxes_dict}')
        # msg.good(f'Classes dict:\n {classes_dict}')
        break

if __name__ == "__main__":
    start = time.time()
    main()
    print("Time taken: ", time.time() - start)
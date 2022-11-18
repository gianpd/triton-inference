import numpy as np
import time
from tritonclient.utils import *
from PIL import Image
import tritonclient.http as httpclient
import cv2
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
        msg.good(img.dtype)
        img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_LANCZOS4)
        msg.good(f'After resize {img.shape}')
        img = img / 255.0
        img = np.asarray(np.expand_dims(img, axis=0), dtype=np.float32)
        msg.good(f'After normalization {img.dtype, img.shape}')
        input_raw_img = httpclient.InferInput("input", img.shape, np_to_triton_dtype(img.dtype))
        input_raw_img.set_data_from_numpy(img)
        print(f'INPUT_RAW_IMG: {input_raw_img}')
        
        output_od_scores = httpclient.InferRequestedOutput("detection_scores")
        

        
        query_response = client.infer(model_name='pipeline',
                                     inputs=[input_raw_img],
                                     outputs=[output_od_scores])
        
        pred_dict = query_response.as_numpy("detection_scores")
        print('IOOOO')
        print(pred_dict)
        break
        
        

    # prompt = "Pikachu with a hat, 4k, 3d render"
    # text_obj = np.array([prompt], dtype="object").reshape((-1, 1))

    # input_text = httpclient.InferInput("prompt", text_obj.shape,
    #                                    np_to_triton_dtype(text_obj.dtype))
    # input_text.set_data_from_numpy(text_obj)

    # output_img = httpclient.InferRequestedOutput("generated_image")

    # query_response = client.infer(model_name="pipeline",
    #                               inputs=[input_text],
    #                               outputs=[output_img])

    # image = query_response.as_numpy("generated_image")
    # im = Image.fromarray(np.squeeze(image.astype(np.uint8)))
    # im.save("generated_image2.jpg")

if __name__ == "__main__":
    start = time.time()
    main()
    print("Time taken: ", time.time() - start)
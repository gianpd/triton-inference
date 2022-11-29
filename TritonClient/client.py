import numpy as np
import time
from tritonclient.utils import *
import tritonclient.http as httpclient
import tritonclient.grpc as grpc
import cv2
import redis_utils as rutils
from wasabi import msg


from preprocessing import Preproccessing_and_predict_dc




class TritonClient:
    
    def __init__(self, url='localhost:8001'):
        self._client = grpc.InferenceServerClient(url=url)
        # self._client = httpclient.InferenceServerClient(url=url)  
        # grpc.InferenceServerClient(url=url, verbose=True) tritonclient.utils.InferenceServerException: [StatusCode.UNAVAILABLE] failed to connect to all addresses
        
    @property
    def get_od_outputs(self):
        return [grpc.InferRequestedOutput("detection_scores"), grpc.InferRequestedOutput("detection_boxes")]
        # return [httpclient.InferRequestedOutput("od_scores"), httpclient.InferRequestedOutput("od_boxes")]
    
    @property
    def get_predc_outputs(self):
        return [grpc.InferRequestedOutput("cropped_images")]    
    
    @property
    def get_dc_outputs(self):
        return [grpc.InferRequestedOutput("output")]  
    
    
    def get_input(self, img: np.ndarray):
        infer_input = grpc.InferInput("input_tensor", img.shape, np_to_triton_dtype(img.dtype))
        # infer_input = httpclient.InferInput("input", img.shape, np_to_triton_dtype(img.dtype))
        infer_input.set_data_from_numpy(img)
        return [infer_input]
    
    def make_request(self, inputs: list, outputs: list, model_name='od'):
        return self._client.infer(model_name=model_name, inputs=inputs, outputs=outputs)
        
TRITONCLIENT = TritonClient()
RESIZE_OUTPUT_IMAGES_SHAPE = (260,260)
THRESHOLD_OD = 0.8
THRESHOLD_DC = 0.6
INTERNAL_PERCENTAGE_IMAGE_TOTAKE = 0.6
REDIS_URL = 'localhost:8001'

preproccessing_and_predict_dc = Preproccessing_and_predict_dc(RESIZE_OUTPUT_IMAGES_SHAPE,THRESHOLD_OD,THRESHOLD_DC,INTERNAL_PERCENTAGE_IMAGE_TOTAKE,REDIS_URL)


def main():
    ### 0. Start the communication channel with the redis server, for receiving the requests to be passed to the models
    #client = httpclient.InferenceServerClient(url="localhost:8000")
    # 1. Start the communication channel with the redis server, for receiving the requests to be passed to the models
    rs_client = rutils.NJRedisClient(host='172.17.0.2', port=6379, db=0, key='NJ')
    
    i = 0
    while True:
        i += 1
       ### 1. Get the payload from KeyDB server
        payload = rs_client.get_msg
        if payload is None:
            msg.info('Received an empty payload')
        msg.info(f'Received payload with keys: {payload.keys()}')
        #raw_img = rs_client.get_np_img(payload['data'], payload['size']['height'], payload['size']['width']) # np turns (Y,X,C)
        raw_img = rs_client.get_np_img(payload['data'])
        msg.info(raw_img.shape, raw_img.dtype)
        
        
        img = cv2.resize(raw_img, (min(raw_img.shape[:2]),min(raw_img.shape[:2]))).astype(np.uint8)
        img = np.asarray(img, dtype=np.uint8)
        img = np.expand_dims(img, axis=0)
        
        ### 2. Make request to TritonServer
        pipe_input =TRITONCLIENT.get_input(img)
        pipe_outputs = TRITONCLIENT.get_od_outputs
        query_response = TRITONCLIENT.make_request(pipe_input, pipe_outputs,model_name='od')
        
        ### 3. Show results
        scores = query_response.as_numpy("detection_scores")
        boxes = query_response.as_numpy("detection_boxes")
        msg.good(f'Scores dict:\n {scores}')
        msg.good(f'Boxes dict:\n {boxes}')
        
        #raw_img = cv2.resize(raw_img, (1500, 1500))
        #raw_img = np.expand_dims(raw_img, axis=0)
        
        bbox_test = boxes[0][1]
        
        msg.good(f'bbox test: {bbox_test}')
        
        dc_response = preproccessing_and_predict_dc.preprocessing(raw_img,scores,boxes)
        
        msg.good(f'DC RESPONSE : {dc_response}')
        
        
       
        break
    

if __name__ == "__main__":
    start = time.time()
    main()
    print("Time taken: ", time.time() - start)
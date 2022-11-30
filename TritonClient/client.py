import numpy as np
import time
from tritonclient.utils import *
import tritonclient.http as httpclient
import tritonclient.grpc as grpc
import cv2
import redis_utils as rutils
from wasabi import msg
import time 


from preprocessing import Preproccessing_and_predict_dc
import logging_utils

logger= logging_utils.get_logger('client')

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
     
try :
   TRITONCLIENT = TritonClient()
except Exception as e : logger.error(e)

logger.info('LOADED TRITON CLIENT')

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
    
    try :
       rs_client = rutils.NJRedisClient(host='172.17.0.2', port=6379, db=0, key='NJ')
    except Exception as e : logger.error(e)
    
    logger.info("CONNECTED TO REDIS")
    
    i = 0
    while True:
        i += 1
        try :
            ### 1. Get the payload from KeyDB server
            payload = rs_client.get_msg
            if payload is None:
                msg.info('Received an empty payload')
            #raw_img = rs_client.get_np_img(payload['data'], payload['size']['height'], payload['size']['width']) # np turns (Y,X,C)
            raw_img = rs_client.get_np_img(payload['data'])
        except Exception as e : logger.error(e)
        
        logger.info("RECEIVED PAYLOAD FROM REDIS")
        
        try :
            img_resize = cv2.resize(raw_img, (min(raw_img.shape[:2]),min(raw_img.shape[:2])))
            img = np.expand_dims(img_resize.astype(np.uint8)
                                , axis=0)
            
            ### 2. Make request to TritonServer
            pipe_input =TRITONCLIENT.get_input(img)
            pipe_outputs = TRITONCLIENT.get_od_outputs
            query_response = TRITONCLIENT.make_request(pipe_input, pipe_outputs,model_name='od')
        except Exception as e : logger.error(e)
        
        logger.info("OD REQUEST EXCECUTED")
        
        try :
            ### 3. Show results
            scores = query_response.as_numpy("detection_scores")
            boxes = query_response.as_numpy("detection_boxes")
            
            #raw_img = cv2.resize(raw_img, (1500, 1500))
            #raw_img = np.expand_dims(raw_img, axis=0)

            ### 4. Make DC request
            
            # Preproccessing with python
            #dc_response = preproccessing_and_predict_dc.preprocessing(raw_img,scores,boxes)
            
            # Preproccessing with model
            dc_response = preproccessing_and_predict_dc.preprocessing_model(img_resize,scores,boxes)
            
            logger.info("DC REQUEST EXCECUTED")
        
            logger.info(f'RESULT : {dc_response}')
            
        except Exception as e : logger.error(e)
        
        
        
        break

        
       

    

if __name__ == "__main__":
    start = time.time()
    main()
    print("Time taken: ", time.time() - start)
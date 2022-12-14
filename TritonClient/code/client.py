import numpy as np
import time
from tritonclient.utils import *
# import tritonclient.http as httpclient
import tritonclient.grpc as grpc
import cv2
import logging_utils
import msgpack
import sys
import base64
import json
from uuid import uuid4
from pympler import asizeof
from simplejpeg import encode_jpeg
import requests


from typing import List, Dict, Union, Optional

logger = logging_utils.get_logger('TritonClient')

#PRIIVATE LIB
import utils
from config import *


def make_anonymizer_request(
    payload: bytes, 
    url: str ='http://localhost:5020/anonymize', 
    method: str ='POST', 
    header: dict ={'Content-Type': 'application/octet-stream'}) -> Union[requests.Response, None]:

    try:
        return requests.request(method, url=url, data=payload, headers=header)
    except requests.ConnectionError as e:
        print(e)
        return None

def create_anonymizer_payload(img : np.array) -> bytes :
    payload = {
        "meta": {
        "size": {
            "height": img.shape[0],
            "width": img.shape[1]
        }
        },
        "image": img.tobytes()
    }

    return msgpack.packb(payload, use_bin_type=True)


class TritonClient:
    
    def __init__(self, url='localhost:4001'):
        #SET URL SERVER
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
    
    def make_od_request(self, od_img):
        # Prepare model input
        od_input = self.get_od_input(od_img)
        od_outputs = self.get_od_outputs
        ### 3. Make request to OD model
        query_response = self.make_request(od_input, od_outputs, model_name='od')
        ### 4. Extract scores and bboxes
        scores = query_response.as_numpy("detection_scores")[0][:5]
        boxes = query_response.as_numpy("detection_boxes")[0][:5]
        return scores, boxes
    
    #Prepare od input
    def get_od_input(self, img: np.ndarray):
        infer_input = grpc.InferInput("input", img.shape, np_to_triton_dtype(img.dtype))
        # infer_input = httpclient.InferInput("input", img.shape, np_to_triton_dtype(img.dtype))
        infer_input.set_data_from_numpy(img)
        return [infer_input]
    
    def make_request(self, inputs: list, outputs: list, model_name='od'):
        return self._client.infer(model_name=model_name, inputs=inputs, outputs=outputs)
        
TRITONCLIENT = TritonClient()


def main():
    # start = time.time()
    ### 0. Start the communication channel with the redis server, for receiving the requests to be passed to the models
    logger.info('##### START REQUESTS #####')
    # Connect to Redis
    rs_client = utils.NJRedisClient(unix_socket_path="/tmp/docker/keydb.sock")
    # Connect to MQTT Server e Topic
    pub = utils.NjMQTTPub(clientID='rock_nj_pub', topic='test_event_alert')
    pub.start()
    # i = 0
    while True:
         ### 1. Get the payload from KeyDB server
        # Get Redis Client
        current_cnt = rs_client.get_cnt
        if current_cnt == None:     
            logger.warning('Waiting Grabber')
            time.sleep(1)
            continue
        # GRABBER
        if int(current_cnt) % DOWNSAMPLING != 0: # il grabber lavora a 15 FPS noi a 5 FPS quindi facciamo un downsampling di 3 e aspettiamo 0.2 s
            time.sleep(0.2)
            continue
        
        # Get Payload from Redis
        grabber_payload = rs_client.get_msg
        if grabber_payload is None:
            logger.warning('Received an empty grabber_payload')
            # raise ValueError('Received an empty grabber_payload')
            continue


        # Read Payload from Redis
        raw_img = rs_client.get_np_img(grabber_payload['data'], grabber_payload['size']['height'], grabber_payload['size']['width']) # np turns (Y,X,C)
        logger.info(f'Received RAW IMG with shape {raw_img.shape}')
        # first resize for preparing OD input
        od_img = np.expand_dims(cv2.resize(raw_img, (640,640)).astype(np.float32), axis=0)
        
        ### 2. Make OD request to TritonServer
        logger.info('Making OD request to TritonServer ...')
        scores, boxes = TRITONCLIENT.make_od_request(od_img)
        logger.info(f'From OD Received: {scores}, {boxes}')
                    
        ### 3. Filtering based on the od scores and bboxes
        bbox_ls = [boxes[ind] for ind in range(len(boxes)) if scores[ind] > TH_SCORE_OD]
        logger.info(f'Filtered bboxes with a TH of {TH_SCORE_OD}. Remained {len(bbox_ls)} bboxes.')
                    
        if len(bbox_ls) == 0:
            logger.info('No NJ in the frame. Continue')
            continue
        
        ### 4. Some NJ have been detected. Prepare DC inputs to make request
        logger.info('Preparing INPUTS/OUTUPS for PREDC and DC requests ...')
        raw_img_resized = np.expand_dims(cv2.resize(raw_img, (1500, 1500)), axis=0)
        predc_inputs = [
                grpc.InferInput('img', raw_img_resized.shape, np_to_triton_dtype(raw_img_resized.dtype)),
                grpc.InferInput('bboxes', bbox_ls[0].shape, np_to_triton_dtype(bbox_ls[0].dtype))
            ]
        
        predc_inputs[0].set_data_from_numpy(raw_img_resized)
        predc_outputs = TRITONCLIENT.get_predc_outputs
        
        ### 5. For over all the detected NJs: not all are centered
        for idx, bbox in enumerate(bbox_ls):
            predc_inputs[1].set_data_from_numpy(bbox)
            logger.info(f'{idx}-th Making PREDC request with NJ bbox: {bbox}')
            
            ### 6. Make request to the dc preprocessing model
            query_response = TRITONCLIENT.make_request(predc_inputs, predc_outputs, model_name='predc')
    
            ### 7. Extract the cropped image (which can be a zeros tensor without batch dimension if not dywidag)
            cropped_img = query_response.as_numpy('cropped_images')
            if cropped_img.shape[0] != 1:
                logger.info('### Dywidag excluded!')
                continue
            
            ### 8. Make request to the DC model
            logger.info('### Dywidag detected!')
            logger.info('Preparing DC request ...')
            # Preparing data for DC prediction
            dc_inputs = [grpc.InferInput('input_1', cropped_img.shape, np_to_triton_dtype(cropped_img.dtype))]
            dc_inputs[0].set_data_from_numpy(cropped_img)
            dc_outputs = [grpc.InferRequestedOutput("Identity")]
            # DC model Request
            query_response = TRITONCLIENT.make_request(
                dc_inputs, 
                dc_outputs, model_name='dc').as_numpy('Identity')
            logger.debug(f"RAW DC OUTPUT PREDICTION: {query_response}")
            
            # Casting DC model results with threshold
            query_response_casted = (query_response >= TH_SCORE_DC).astype(int)
            logger.debug(f'query_response_casted: {query_response_casted}')
            if query_response_casted[0][0] or not query_response_casted[0].any():
                logger.info('No defects detected. Skipped.')
            else:
                # Get class of damage
                label_names = PREDS_MAP_DICT.get(str(query_response_casted)) # list with the EVT_CODE
                logger.info(f'Anomaly detected: {label_names}')
                
                ### Call the anonymizer
                # Prepare data for anonymizer
                cropped_image_casted_uint8= cropped_img[0].astype(np.uint8)
                logger.info(f'Original cropped image dimension: {sys.getsizeof(cropped_img[0])}')
                logger.info(f'Casted to uint8 cropped image dimension: {sys.getsizeof(cropped_image_casted_uint8)}')
                
                anonymizer_payload = create_anonymizer_payload(cropped_image_casted_uint8)
                # Make anonymizer request
                response = make_anonymizer_request(anonymizer_payload) # i'm expecting a msgpack
                
                # Read Anonymizer output
                response = msgpack.loads(response.content) #response = msgpack.unpackb(response.content, raw=False)
                cropped_img = response['image'] # the anonymized cropped img to be sent
                logger.info(f'Anonymizer output shape: {cropped_img.shape}')
                
                ### Prepare the alert msg and publish it to the MQTT broker
                events = [{
                    "timestamp": grabber_payload['timestamp'], 
                    "uid": uuid4().hex[:10], 
                    "model": MODEL, 
                    "model_version": MODEL_VERSION,
                    "event_code": x,
                    "position": grabber_payload['position']
                } for x in label_names]
                
                JPEG = encode_jpeg(cropped_image_casted_uint8) # returns bytes
                logger.debug(f'The len of the jpeg-encoded bytes is {len(JPEG)}')
                alert_json = {
                    'rock_event': make_event_json(grabber_payload['camera_id'], events),
                     'body': base64.b64encode(JPEG).decode('utf-8')
                }
                
                #logger.debug(f'Sending alert msg: {alert_json}')
                #logger.debug(asizeof.asizeof(alert_json))
                pub.publish(json.dumps(alert_json))
                
                
    

def make_event_json(camera_id: str, events: List[Dict[str, str]]) -> Dict:  
    return {
        "appliance": {
            "uid": uuid4().hex[:8],
            "category": "inference_model",
            "hardware_version": "",
            "software_version": "AIJ_grabber-od-predc-dc"
            },
        "camera": {
            "uid": camera_id,
            "pointing": "LEFT"
            },
        "object_code": "OBJ_NJ",
        "application_code": "AIJ",
        "events": events,
        "crc": "010111001"
    }


if __name__ == "__main__":
    main()
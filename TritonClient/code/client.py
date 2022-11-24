import numpy as np
import time
from tritonclient.utils import *
# import tritonclient.http as httpclient
import tritonclient.grpc as grpc
import cv2
import redis_utils as rutils
from wasabi import msg

class TritonClient:
    
    def __init__(self, url='localhost:8001'):
        self._client = grpc.InferenceServerClient(url=url)
        # self._client = httpclient.InferenceServerClient(url=url)  
        # grpc.InferenceServerClient(url=url, verbose=True) tritonclient.utils.InferenceServerException: [StatusCode.UNAVAILABLE] failed to connect to all addresses
        
    @property
    def get_od_outputs(self):
        return [grpc.InferRequestedOutput("detection_scores"), grpc.InferRequestedOutput("detection_boxes")]
        # return [httpclient.InferRequestedOutput("od_scores"), httpclient.InferRequestedOutput("od_boxes")]
        
    def get_input(self, img: np.ndarray):
        infer_input = grpc.InferInput("input", img.shape, np_to_triton_dtype(img.dtype))
        # infer_input = httpclient.InferInput("input", img.shape, np_to_triton_dtype(img.dtype))
        infer_input.set_data_from_numpy(img)
        return [infer_input]
    
    def make_request(self, inputs: list, outputs: list, model_name='od'):
        return self._client.infer(model_name=model_name, inputs=inputs, outputs=outputs)
        
TRITONCLIENT = TritonClient()


def main():
    ### 0. Start the communication channel with the redis server, for receiving the requests to be passed to the models
    rs_client = rutils.NJRedisClient(unix_socket_path="/tmp/docker/keydb.sock")
    
    ### 1. Get the payload from KeyDB server
    payload = rs_client.get_msg
    if payload is None:
        msg.info('Received an empty payload')
    msg.info(f'Received payload with keys: {payload.keys()}')
    img = rs_client.get_np_img(payload['data'], payload['size']['height'], payload['size']['width']) # np turns (Y,X,C)
    msg.info(img.shape, img.dtype)
    
    
    img = cv2.resize(img, (640,640)).astype(np.float32)
    # img = np.asarray(img, dtype=np.float32)
    img = np.expand_dims(img, axis=0)
    
    ### 2. Make request to TritonServer
    pipe_input =TRITONCLIENT.get_input(img)
    pipe_outputs = TRITONCLIENT.get_od_outputs
    query_response = TRITONCLIENT.make_request(pipe_input, pipe_outputs)
    ### 3. Show results
    scores_dict = query_response.as_numpy("detection_scores")
    boxes_dict = query_response.as_numpy("detection_boxes")
    msg.good(f'Scores dict:\n {scores_dict}')
    msg.good(f'Boxes dict:\n {boxes_dict}')



# def main():
#     # 1. Start the communication channel with the redis server, for receiving the requests to be passed to the models
#     rs_client = rutils.NJRedisClient(unix_socket_path="/tmp/docker/keydb.sock")
#     with httpclient.InferenceServerClient(url='0.0.0.0:8000') as client:
#         payload = rs_client.get_msg
# #         if payload['EXIT']:
# #             msg.info('Exiting TritonClient main ...')
# #             break
#         img = rs_client.get_np_img(payload['data'])
#         msg.good(f'Initial img shape: {img.shape}')
#         img = np.expand_dims(img, axis=0)
        
#         input_raw_img = httpclient.InferInput("input_tensor", img.shape, np_to_triton_dtype(img.dtype))
#         input_raw_img.set_data_from_numpy(img)
#         # msg.good(f'INPUT_RAW_IMG: {input_raw_img}')
        
#         od_outputs  = [
#             httpclient.InferRequestedOutput("od_scores"),
#             httpclient.InferRequestedOutput("od_boxes"),
#             # httpclient.InferRequestedOutput("detection_classes"),
#         ]
        
#         query_response = client.infer(model_name='pipeline',
#                                      inputs=[input_raw_img],
#                                      outputs=od_outputs)
        
#         scores_dict = query_response.as_numpy("od_scores")
#         boxes_dict = query_response.as_numpy("od_boxes")
#         # classes_dict = query_response.as_numpy("detection_classes")
#         msg.good(f'Scores dict:\n {scores_dict}')
#         msg.good(f'Boxes dict:\n {boxes_dict}')


if __name__ == "__main__":
    start = time.time()
    main()
    print("Time taken: ", time.time() - start)
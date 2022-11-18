"""
Tritonserver pipeline running the R.O.C.K NJ models: DC/OD.
"""

import json
from wasabi import msg
import triton_python_backend_utils as pb_utils

LOG_IDX= 'NJTritonServer>'
class TritonPythonModel:
    
    def initialize(self, args):
        self.output_dtype = pb_utils.triton_string_to_numpy(
            pb_utils.get_output_config_by_name(json.loads(args["model_config"]),
                                               "detection_scores")["data_type"])
    def execute(self, requests):
        response = []
        for request in requests:
            
            input_img = pb_utils.get_input_tensor_by_name(request, "input")
            # input_img = inp.as_numpy()
            msg.info(f'{LOG_IDX} Receiced input: {input_img.shape()}')
            
            
            encoding_request = pb_utils.InferenceRequest(
               model_name='od',
               requested_output_names=['detection_scores'],
               inputs=[input_img]
            )
            
            response = encoding_request.exec()
            msg.info(f'Received response with type {type(response)}')
            if response.has_error():
                raise pb_utils.TritonModelException(response.error().message())
            else:
                od_scores = pb_utils.get_output_tensor_by_name(
                    response, "detection_scores")
                msg.info(f'OD_SCORES: {type(od_scores)}')
                
                
            inference_response = pb_utils.Inferenceresponse(output_tensors=[
                pb_utils.Tensor(
                "detection_scores",
                od_scores
                )
            ])
            responses.append(inference_response)
        return responses
          
         
             
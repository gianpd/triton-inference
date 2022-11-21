"""
Tritonserver pipeline running the R.O.C.K NJ models: DC/OD.
"""

import json
from wasabi import msg

from typing import List, Union, Optional

import numpy as np

import triton_python_backend_utils as pb_utils

OD_SCORE_TH = 0.8
INTERNAL_PERCENTAGE_IMAGE_TOTAKE = 0.6

LOG_IDX= 'NJTritonServer>'
class TritonPythonModel:
    
    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])
        msg.good(f'model_config: {self.model_config}')
        
        self.detection_scores_config = pb_utils.get_output_config_by_name(self.model_config, 'detection_scores')
        msg.good(f'detection_scores config: {self.detection_scores_config}')

        self.detection_boxes_config = pb_utils.get_output_config_by_name(self.model_config, 'detection_boxes')
        msg.good(f'detection_boxes config: {self.detection_boxes_config}') 

        # self.detection_classes_config = pb_utils.get_output_config_by_name(self.model_config, 'detection_classes')
        # msg.good(f'detection_classes config: {self.detection_classes_config}')   
        
    def execute(self, requests):
        """"
        The client sends requests to the pipeline model, which in turn it must:
        1. Make a request to the OD model and getting its response;
        2. BLS: assures there are dywidags in the frame, if no, stop, if yes go to 3;
        3. Make a request to the DC model and getting its response;
        4. BLS: if defects send message with image and defects (alarm message), if no defects send message without alarm.
        """
        responses = []
        for request in requests:
            input_img = pb_utils.get_input_tensor_by_name(request, "input_tensor")
            size = input_img.shape()
            msg.info(f'{LOG_IDX} Receiced input with shape: {size}')
        
            # make an inference request to the OD model with the input_img as received by the pipeline model
            od_scores, od_boxes = self.make_od_request(input_img)
            od_scores, od_boxes = od_scores.as_numpy(), od_boxes.as_numpy()

            msg.info(f'od_scores: {od_scores}')
            msg.info(f'od_boxes: {od_boxes}')

            # check if there dywidags
            boxes_to_consider = self.check_if_dywidags(size[1], size[2], od_scores[0], od_boxes[0])
            if len(boxes_to_consider):
                # make DC request
                # self.make_dc_request(cropped_img)
                od_scores = pb_utils.Tensor('od_scores', od_scores)
                od_boxes = pb_utils.Tensor('od_boxes', od_boxes)
                inference_response = pb_utils.InferenceResponse(output_tensors=[od_scores, od_boxes])
                responses.append(inference_response)
        return responses
          
         
    def make_od_request(self, input_img):
        # make an inference request to the OD model with the input_img as received by the pipeline model
        od_encoding_request = pb_utils.InferenceRequest(
            model_name='od',
            requested_output_names=['detection_scores', 'detection_boxes'],
            inputs=[input_img]
        )
            
        response = od_encoding_request.exec()
        if response.has_error():
            msg.info('Error in pipeline')
            raise pb_utils.TritonModelException(response.error().message())
        else:
            od_scores = pb_utils.get_output_tensor_by_name(
                    response, "detection_scores")
            od_boxes = pb_utils.get_output_tensor_by_name(
                    response, "detection_boxes"
                )
        return od_scores, od_boxes

    def check_if_dywidags(self, w_img, h_img, od_scores, od_boxes) -> List[float]:

        if od_scores[0] >= OD_SCORE_TH:
            # there is at least one dywidag
            bboxes_to_consider = []
            for idx, bbox in enumerate(od_boxes[:10]): # check just the first 10 elements
                score = od_scores[idx]
                if score >= OD_SCORE_TH:
                    # if the bbox is in the 60% internal it is kept, otherwise it is discarded
                    if bbox[1] >= (0.5 - INTERNAL_PERCENTAGE_IMAGE_TOTAKE/2) and bbox[3] <= (0.5 + INTERNAL_PERCENTAGE_IMAGE_TOTAKE/2):
                        bbox_coco = self.bbox_to_coco(bbox, w_img, h_img)
                        bboxes_to_consider.append(bbox_coco)
            return bboxes_to_consider
        return []


    def make_dc_request(self):
        pass
        # # make an inference request to the OD model with the input_img as received by the pipeline model
        # od_encoding_request = pb_utils.InferenceRequest(
        #     model_name='dc',
        #     requested_output_names=['detection_scores', 'detection_boxes'],
        #     inputs=[input_img]
        # )
            
        # response = od_encoding_request.exec()
        # if response.has_error():
        #     raise pb_utils.TritonModelException(response.error().message())
        # else:
        #     od_scores = pb_utils.get_output_tensor_by_name(
        #             response, "detection_scores")
        #     od_boxes = pb_utils.get_output_tensor_by_name(
        #             response, "detection_boxes"
        #         )
        # return od_scores, od_boxes

    def check_if_defects(self):
        pass

    def bbox_to_coco(self, bbox, w_img, h_img):
        """
        Convert tf OD api format (y_m, x_m, Y_M, X_M) to the non-normalized coco format (x_m, y_m, w, h)
        """
        w = (bbox[3] - bbox[1])*w_img
        h = (bbox[2] - bbox[0])*h_img
        x = bbox[1]*w_img
        y = bbox[0]*h_img
        return [x, y, w, h]
"""
Tritonserver pipeline running the R.O.C.K NJ models: DC/OD.
"""

import json
from wasabi import msg

from typing import List, Union, Optional

import numpy as np
import tensorflow as tf
#from PIL import Image
import io

import triton_python_backend_utils as pb_utils

OD_SCORE_TH = 0.8
INTERNAL_PERCENTAGE_IMAGE_TOTAKE = 0.6
DC_SCORE_TH = 0.9
RESIZE_OUTPUT_IMAGES_SHAPE = 260


class TF_preprocessing_model_justtf():

    def __init__(
        self, 
        OD_SCORE_TH: float, 
        INTERNAL_PERCENTAGE_IMAGE_TOTAKE: float, 
        RESIZE_OUTPUT_IMAGES_SHAPE: int) -> None:
        
        #tf.keras.model.__super__()
        self.od_thr= tf.convert_to_tensor(OD_SCORE_TH)
        self.bbox_position_to_take= tf.convert_to_tensor(INTERNAL_PERCENTAGE_IMAGE_TOTAKE)
        self.resize_shape= tf.convert_to_tensor([RESIZE_OUTPUT_IMAGES_SHAPE,RESIZE_OUTPUT_IMAGES_SHAPE])
        
    
    @tf.function
    def filter_and_crop_bboxes(
        self, 
        od_scores: tf.Tensor, 
        od_bboxes: tf.Tensor,
        image: tf.Tensor):
        
        """
        Select bboxes to further analyze basing on the relative score - converting bbox to coco
        """
        od_scores = od_scores[0]
        od_bboxes = od_bboxes[0]
        
        cropped_bboxes = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
        if od_scores[0] >= self.od_thr:
            for ind in tf.range(10):
                if od_scores[ind] >= self.od_thr:
                    if od_bboxes[ind][1] >= (0.5 - self.bbox_position_to_take/2) and od_bboxes[ind][3] <= (0.5 + self.bbox_position_to_take/2):
                        cropped_bboxes = cropped_bboxes.write(ind, tf.image.crop_and_resize(image= image, boxes=tf.convert_to_tensor([od_bboxes[ind]]), box_indices= tf.convert_to_tensor([0]), crop_size= self.resize_shape, method='bilinear'))
        
        #return output = self.Model()
        return cropped_bboxes.stack()


preprocess = TF_preprocessing_model_justtf(OD_SCORE_TH=OD_SCORE_TH,
                                                      INTERNAL_PERCENTAGE_IMAGE_TOTAKE=INTERNAL_PERCENTAGE_IMAGE_TOTAKE,
                                                      RESIZE_OUTPUT_IMAGES_SHAPE=RESIZE_OUTPUT_IMAGES_SHAPE)

LOG_IDX= 'NJTritonServer>'
class TritonPythonModel:
    
    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])
        msg.good(f'model_config: {self.model_config}')
        
        self.detection_scores_config = pb_utils.get_output_config_by_name(self.model_config, 'detection_scores')
        msg.good(f'detection_scores config: {self.detection_scores_config}')

        self.detection_boxes_config = pb_utils.get_output_config_by_name(self.model_config, 'detection_boxes')
        msg.good(f'detection_boxes config: {self.detection_boxes_config}') 

        self.detection_classes_config = pb_utils.get_output_config_by_name(self.model_config, 'output')
        msg.good(f'detection_classes config: {self.detection_classes_config}')   
        
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
            od_scores, od_boxes = tf.convert_to_tensor(od_scores.as_numpy(),dtype=tf.float32), tf.convert_to_tensor(od_boxes.as_numpy(),dtype=tf.float32)
            
            boxes_to_consider= preprocess.filter_and_crop_bboxes(od_scores=od_scores,od_bboxes=od_boxes,image=tf.convert_to_tensor(input_img.as_numpy(),dtype=tf.uint8))

            # check if there dywidags
            if len(boxes_to_consider[0]):
                for image_cropped in boxes_to_consider[0] :
                    inference_dc = self.inference_single_image_dc(tf.expand_dims(image_cropped,axis=0).numpy())
                    inference_dc = self.check_if_defects(inference_dc)
                    try :
                        dc_score = pb_utils.Tensor('dc_scores', inference_dc)
                    except :
                        dc_score = pb_utils.Tensor('dc_scores',np.zeros((1,3)))
                    inference_response = pb_utils.InferenceResponse(output_tensors=[dc_score])
                    responses.append(inference_response)
        return responses
    
    
    def inference_single_image_dc(self, image_cropped) :
        
            out_tensor_0 = pb_utils.Tensor("input_1",image_cropped)
            dc_scores = self.make_dc_request(out_tensor_0)
            
            return dc_scores 

         
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


    def make_dc_request(self, input_img):
        # # make an inference request to the DC model with the input_img as received by the od model
        dc_encoding_request = pb_utils.InferenceRequest(
             model_name='dc',
             requested_output_names=['output'],
             inputs=[input_img]
         )
            
        response = dc_encoding_request.exec()
        if response.has_error():
             raise pb_utils.TritonModelException(response.error().message())
        else:
             dc_scores = pb_utils.get_output_tensor_by_name(
                     response, "output")
        return dc_scores     
    
    
    def check_if_defects(self, dc_score):
        if dc_score == None :
            msg.info(f'SONO QUA {np.zeros(dc_score.shape)}')
            return np.zeros(dc_score.shape)
        else :
            for score in dc_score.as_numpy()[0] :
                if score > DC_SCORE_TH : 
                    return dc_score.as_numpy()
            return dc_score.as_numpy().fill(0)

    def bbox_to_coco(self, bbox, w_img, h_img):
        """
        Convert tf OD api format (y_m, x_m, Y_M, X_M) to the non-normalized coco format (x_m, y_m, w, h)
        """
        w = (bbox[3] - bbox[1])*w_img
        h = (bbox[2] - bbox[0])*h_img
        x = bbox[1]*w_img
        y = bbox[0]*h_img
        return [x, y, w, h]
    
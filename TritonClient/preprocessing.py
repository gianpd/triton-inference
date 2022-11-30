import cv2
import numpy as np
import tritonclient.grpc as grpc
from tritonclient.utils import *
from wasabi import msg

import logging_utils

logger = logging_utils.get_logger('preprocessing')

class Preproccessing_and_predict_dc :
    def __init__(self, resize_output,threshold_od,threshold_dc,internal_percentage_image_totake,redis_url) :
        self.resize_output = resize_output
        self.threshold_od = threshold_od
        self.threshold_dc = threshold_dc
        self.internal_percentage_image_totake = internal_percentage_image_totake
        self.redis_url = redis_url
        self.factor1_image_totake = 0.5 - self.internal_percentage_image_totake/2
        self.factor2_image_totake = 0.5 + self.internal_percentage_image_totake/2
        
        
    def check_od_result(self,list_of_tuple) :
        
        scores,bbox = list_of_tuple
        
        list_output_dc = []
        
        if scores >= self.threshold_dc:
            #if the bbox is in the 60% internal it is kept, otherwise it is discarded
            #if bbox[1] >= self.factor1_image_totake and bbox[3] <= self.factor2_image_totake :
            
            if  np.logical_and(bbox[1]>=self.factor1_image_totake,bbox[3]<=self.factor2_image_totake) :
                
                #bbox_coco = od_utils.bbox_to_coco(bbox, w_img, h_img)
                
                # ymin = bbox[0]*raw_img.shape[0]
                # xmin = bbox[1]*raw_img.shape[1]
                # ymax = bbox[2]*raw_img.shape[0]
                # xmax = bbox[3]*raw_img.shape[1]
                
                
                list_output_dc.append(self.predict_dc(self.raw_img,
                                            (bbox[0]*self.raw_img.shape[0],
                                            bbox[1]*self.raw_img.shape[1],
                                            bbox[2]*self.raw_img.shape[0],
                                            bbox[3]*self.raw_img.shape[1])))
                
        return list_output_dc
    
    
    def crop_predict_predc(self, list_of_tuples) :
        scores,bbox = list_of_tuples
        
        raw_img = self.raw_img
        
        list_output_dc = []
        
        if scores >= self.threshold_dc:
            if  np.logical_and(bbox[1]>=self.factor1_image_totake,bbox[3]<=self.factor2_image_totake) :
                
                cropped_img = self.predict_predc(np.expand_dims(cv2.cvtColor(raw_img,cv2.COLOR_RGB2BGR),axis=0),bbox)
                
                dc_response = self.predict_dc(cropped_img,bbox,mode='model')
                
                list_output_dc.append(dc_response)
        
        return list_output_dc

    def preprocessing_python(self,raw_img,scores,boxes) :
        
        logger.info(f'PREPROCESSING WITH PYTHON...')
        
        self.raw_img = raw_img
        
        if scores[0][0] >= self.threshold_od :
            #take bboxes with a score higher than the threshold
            #list_of_tuples = list(zip(list(scores[0][:7]), list(boxes[0][:7])))
            list_of_tuples = list((scores[0][x],boxes[0][x]) for x in range(0,7))
            
            try :
                results = map(self.check_od_result,list_of_tuples)
                results = [result for result in results if len(result) > 0]
            except Exception as e : logger.error(e)
            
            logger.info(f'RESULTS : {results}')
            logger.info(f'RESULTS N° : {len(results)}')

            return results
    
   
    def preprocessing_model(self, raw_img, scores,boxes) :
        logger.info("PREPROCESSING WITH MODEL")
        
        
        self.raw_img = raw_img
        
        if scores[0][0] >=  self.threshold_od :
            
            list_of_tuples = list((scores[0][x],boxes[0][x]) for x in range(0,7))
            
            try :
                results = map(self.crop_predict_predc,list_of_tuples)
                results = [result for result in results if len(result) > 0]
            except Exception as e : logger.error(e)
                   
                        
        return results
                    
                    
                            
                        
    def predict_dc(self,raw_img,coordinates,mode = 'python') :
        
        logger.info("START PREPROCESSING DC")
        
        if mode == 'python' :
            cropped_image_resized = np.expand_dims(cv2.cvtColor(cv2.resize(raw_img[int(coordinates[0]):int(coordinates[2]), int(coordinates[1]):int(coordinates[3])]
                                            , self.resize_output, interpolation=cv2.INTER_LANCZOS4), 
                                                cv2.COLOR_RGB2BGR).astype(np.float32), axis=0)
        else :
            cropped_image_resized = raw_img
            
        dc_inputs =[
            grpc.InferInput('input_1',cropped_image_resized.shape, np_to_triton_dtype(cropped_image_resized.dtype)),
            ]
            
        dc_inputs[0].set_data_from_numpy(cropped_image_resized)
        
        dc_outputs = [grpc.InferRequestedOutput("output")] 
        
        logger.info("START INFERENCE DC")
        
        return np.where(grpc.InferenceServerClient(url=self.redis_url).infer(model_name='dc', inputs=dc_inputs, outputs=dc_outputs).as_numpy('output') > self.threshold_dc,1,0)
    
    
    def predict_predc(self,raw_img,bboxes) :
        
        logger.info("START PREPROCESSING PREDC")
        
        predc_inputs = [
            grpc.InferInput('img',raw_img.shape, np_to_triton_dtype(raw_img.dtype)),
            grpc.InferInput('bboxes',bboxes.shape, np_to_triton_dtype(bboxes.dtype)),
            ]
        
        predc_inputs[0].set_data_from_numpy(raw_img)
        predc_inputs[1].set_data_from_numpy(bboxes)
        
        predc_outputs = [grpc.InferRequestedOutput("cropped_images")] 
        
        logger.info("START INFERENCE PREDC")
        
        return grpc.InferenceServerClient(url=self.redis_url).infer(model_name='predc', inputs=predc_inputs, outputs=predc_outputs).as_numpy('cropped_images')
    

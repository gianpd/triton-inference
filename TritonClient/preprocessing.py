import cv2
import numpy as np
import tritonclient.grpc as grpc
from tritonclient.utils import *

class Preproccessing_and_predict_dc :
    def __init__(self, resize_output,threshold_od,threshold_dc,internal_percentage_image_totake,redis_url) :
        self.resize_output = resize_output
        self.threshold_od = threshold_od
        self.threshold_dc = threshold_dc
        self.internal_percentage_image_totake = internal_percentage_image_totake
        self.redis_url = redis_url
    
    def preprocessing(self,raw_img,scores,boxes) :
        list_output_dc = []
        
        if scores[0][0] >= self.threshold_od :
            #take bboxes with a score higher than the threshold
            for ind,bbox in enumerate(boxes[0][:10]):
                score = scores[0][ind]
                if score >= self.threshold_dc:
                    #if the bbox is in the 60% internal it is kept, otherwise it is discarded
                    if bbox[1] >= (0.5 - self.internal_percentage_image_totake/2) and bbox[3] <= (0.5 + self.internal_percentage_image_totake/2):
                        #bbox_coco = od_utils.bbox_to_coco(bbox, w_img, h_img)
                        
                        # ymin = bbox[0]*raw_img.shape[0]
                        # xmin = bbox[1]*raw_img.shape[1]
                        # ymax = bbox[2]*raw_img.shape[0]
                        # xmax = bbox[3]*raw_img.shape[1]
                        
                        
                        list_output_dc.append(self.predict_dc(raw_img,
                                                      (bbox[0]*raw_img.shape[0],
                                                       bbox[1]*raw_img.shape[1],
                                                       bbox[2]*raw_img.shape[0],
                                                       bbox[3]*raw_img.shape[1])))
            return list_output_dc
                        
                        
                        
    def predict_dc(self,raw_img,coordinates) :
            
        cropped_image_resized = np.expand_dims(np.asarray(cv2.cvtColor(cv2.resize(raw_img[int(coordinates[0]):int(coordinates[2]), int(coordinates[1]):int(coordinates[3])]
                                          , self.resize_output, interpolation=cv2.INTER_LANCZOS4), 
                                             cv2.COLOR_RGB2BGR), dtype=np.float32), axis=0)
            
        dc_inputs =[
            grpc.InferInput('input_1',cropped_image_resized.shape, np_to_triton_dtype(cropped_image_resized.dtype)),
            ]
            
        dc_inputs[0].set_data_from_numpy(cropped_image_resized)
        
        dc_outputs = [grpc.InferRequestedOutput("output")] 
        
        return np.where(grpc.InferenceServerClient(url=self.redis_url).infer(model_name='dc', inputs=dc_inputs, outputs=dc_outputs).as_numpy('output') > self.threshold_dc,1,0)
    
    

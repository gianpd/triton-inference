import os

OBJ_CODE = os.getenv("OBJ_CODE", 'OBJ_NJ')
AP_CODE = os.getenv("AP_CODE", 'AIJ')
TH_SCORE_DC =  os.getenv("TH_SCORE_DC", 0.3)
TH_SCORE_OD =  os.getenv("TH_SCORE_OD", 0.8)
MODEL =  os.getenv('MODEL', 'rock_nj_model')
MODEL_VERSION =  os.getenv('MODEL_VERSION', 'v0.1')
DOWNSAMPLING =  os.getenv('DOWNSAMPLING', 3)


PREDS_MAP_DICT = {
    '[[0 0 0]]': ['Not Classified'], 
    '[[1 0 0]]': ['Perfect'],
    '[[1 1 1]]': ['Perfect'],
    '[[1 1 0]]': ['Perfect'],
    '[[1 0 1]]': ['Perfect'],
    '[[0 1 0]]': ['EVN_NIA'],  
    '[[0 0 1]]': ['EVN_FPD'],  
    '[[0 1 1]]': ['EVN_NIA', 'EVN_FPD'],
        }
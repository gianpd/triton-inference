# R.O.C.K NJ Triton Client/Server repository 

1. KeyDB: KeyDB server with the grabber publisher
2. TritonServer: Tritoninference server with the model_repository, containing:
  - od: Object Detection EfficienDet model
  - dc: Damage Classification EfficienNet model 
  - pipeline: Python BSL model

3. TritonClient: Triton python client which requests (GET) payload to the KeyDB and requests predictions to the tritonServer (pipeline model)


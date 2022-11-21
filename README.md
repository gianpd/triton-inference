# R.O.C.K NJ Triton Client/Server repository 

1. KeyDB: KeyDB server with the grabber publisher
2. TritonServer: Tritoninference server with the model_repository, containing:
  a. od;
  b. dc;
  c. pipeline;

3. TritonClient: Triton python client which requests (GET) payload to the KeyDB and requests predictions to the tritonServer (pipeline model)


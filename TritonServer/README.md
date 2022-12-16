# Triton Server

### First steps
0. bash run_docker_triton_server.sh

N.B: the model_repository is mapped to the volume models
### Launch the triton server:
Develop:
  1. tritonserver --exit-on-error=false --log-verbose=1 --model-repository=/models/


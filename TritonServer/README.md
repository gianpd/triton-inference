# Triton Server

### First steps
0. bash run_docker_triton_server.sh
1. from inside the container launch: python3 -m pip install --upgrade pip
2. go to the workspace directory and launch: pip install -r requirements.txt

N.B: the model_repository is mapped to the volume models
### Launch the triton server:
Develop:
  1. tritonserver --exit-on-error=false --log-verbose=1 --model-repository=models/


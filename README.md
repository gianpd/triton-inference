# R.O.C.K NJ Triton Client/Server repository 

1. KeyDB: KeyDB server with the grabber publisher

2. TritonServer: Tritoninference server with the model_repository, containing:
  - od: Object Detection EfficienDet model
  - predc: Keras model for Cropping and Resizing images before dc
  - dc: Damage Classification EfficienNet model 

3. TritonClient: Triton python client which requests (GET) payload to the KeyDB and requests predictions to the tritonServer

## How to Run Inference

Below there are 2 analogous procedures for running an Inference pipeline, the first using a Docker-Based approach and the second using a Docker-Compose approach.

### Docker-Based approach (For production)

In this procedure the docker tool will be used. In particular, the Triton-Server has to be launched first. Afterwards, the KeyDB and the Triton-Client has to be run. Below there are the specific commands that have to be launched:

* *Starting Triton-Server Container*: in the Triton-Server folder, one first has to run the related bash command, which runs the Triton-Server container by exposing 3 ports (8000, 8001 and 8002 by default) and by mapping the 'model_repository' folder (containing od, predc and dc models) to the 'models' volume 

```bash run_docker_triton_server.sh```

* *Launching Triton-Server*: from inside the Triton-Server container, one first has to install and upgrade pip, then installing the related requirements.txt and finally launching the Triton-Server. These are the commands for accomplishing such actions:

```python3 -m pip install --upgrade pip```

```pip install -r requirements.txt```

```tritonserver --exit-on-error=false --log-verbose=1 --model-repository=/models/```

If the 3 models (Object Detection, Image Processing and Damage Classification) have been successfully deployed and Ready to be used, the docker-compose could be finally launched.

* *Launching Triton-Client*: in the Triton-Client folder, one first has build the related container, then running the latter in interactive modality. This will subsequently run the **triton-inference/TritonClient/code/client.py**, whose logs can be seen on the terminal and saved in the related folder

```docker build .```

```docker run -it --entrypoint sh```

### Docker-Compose approach

In this procedure the docker tool, as well as the docker-compose tool, will be used. In particular, the Triton-Server has to be launched first. Afterwards, the docker-compose has to be run, thus starting the KeyDB server and the Triton-Client. Below there are the specific commands that have to be launched:

* *Starting Triton-Server Container*: in the Triton-Server folder, one first has to run the related bash command, which runs the Triton-Server container by exposing 3 ports (8000, 8001 and 8002 by default) and by mapping the 'model_repository' folder (containing od, predc and dc models) to the 'models' volume 

```bash run_docker_triton_server.sh```

* *Launching Triton-Server*: from inside the Triton-Server container, one first has to install and upgrade pip, then installing the related requirements.txt and finally launching the Triton-Server. These are the commands for accomplishing such actions:

```python3 -m pip install --upgrade pip```

```pip install -r requirements.txt```

```tritonserver --exit-on-error=false --log-verbose=1 --model-repository=/models/```

If the 3 models (od (Object Detection), predc (Pre Damage Classification)  and dc (Damage Classification)) have been successfully deployed and ready to be used, the docker-compose could be finally launched.

* *Launching Docker-compose (KeyDB and Triton-Client)*: by running the compose.yaml file with docker-compose, the KeyDB server will be launched (by running the scripts in **triton-inference/scripts**), as well as the Triton-Client. This will subsequently run the **triton-inference/TritonClient/code/client.py**, whose logs can be seen on the terminal and saved in the related folder

```docker-compose up --build --remove-orphans```






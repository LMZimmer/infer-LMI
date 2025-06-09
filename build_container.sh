#!/bin/bash

# build
docker build -t gliodil -f Dockerfile .

# save as tar
docker save -o gliodil.tar gliodil:latest

# remove image
docker rmi gliodil:latest

# load image from tar
#docker load -i gliodil.tar

# test
#docker run -e --gpus '"device=0"' 

# check running containers
#docker ps -a

# remove specific container
#dpcker rm <hash>

# remove image
#docker rmi gliodil:latest



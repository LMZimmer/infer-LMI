#!/bin/bash

# build
docker build -t lmi -f Dockerfile .

# save as tar
docker save -o lmi.tar lmi:latest

# remove image
docker rmi lmi:latest

# load image from tar
#docker load -i lmi.tar

# test
#docker run -e --gpus '"device=0"' 

# check running containers
#docker ps -a

# remove specific container
#dpcker rm <hash>

# remove image
#docker rmi lmi:latest

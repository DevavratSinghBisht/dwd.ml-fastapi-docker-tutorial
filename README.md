# ML Fast API Docker Tutorial
MLOps Session : Tutorial for deploying a ML model using Fast API and Docker for Thakur College of Engineering and Technology, University of Mumbai.

`train` directory contains a Jupyter notebook used for training the CNN Model along with it's weights.

`media` directory contains sample images that can be used for testing the endpoint.

`deploy` contains the FastAPI code and Docker file for deployment.

## Create Docker Image and Run the Container

Go to the project root and follow either of the two methods mentioned below in order to run the docker container.

### Method 1: Using Dockerfile
1. Run `cd deploy` to cd into the deploy directory
3. Run `docker built -t cifar10_cnn:v1 .` to create a docker image named cifar10_cnn with v1 as it's version.
4. Run `docker run -it -p 8080:8080 cifar10_cnn:v1` to run the docker image created in step 2 and connect localhost port 8080 to docker's 8080.

### Method 2: Using docker-compose.yaml
1. Run `docker compose up`

## Infernece

After running the docker container it should host an API at `localhost:8080/api/get_predictions`. It can also be accessed using Swagge UI at  `localhost:8080/docs`. You can pass an image to the API and it should return the prediction for the image from the following classes: 'plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'.

#!/bin/bash

# Build and run script for Smoldyn GPU Docker container

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME_DEV="smoldyn-gpu-dev"
IMAGE_NAME_RELEASE="smoldyn-gpu-release"
CONTAINER_NAME="smoldyn-gpu-container"
DEV_CONTAINER_NAME="smoldyn-gpu-dev"

# Function to display usage
usage() {
    echo "Usage: $0 [build-dev|build-release|run|clean] [options]"
    echo ""
    echo "Commands:"
    echo "  build-dev     Build the development Docker image"
    echo "  build-release Build the release Docker image"
    echo "  run <config>  Run the release container with a configuration file"
    echo "  dev           Start a persistent development container"
    echo "  clean         Remove the Docker images and containers"
    echo "  shell         Start an interactive shell in the development container"
    echo ""
    echo "Examples:"
    echo "  $0 build-dev"
    echo "  $0 build-release"
    echo "  $0 run equil.txt"
    echo "  $0 dev"
    echo "  $0 shell"
    echo "  $0 clean"
}

# Function to build the development Docker image
build_dev_image() {
    echo "Building Smoldyn GPU development Docker image..."
    docker build -t $IMAGE_NAME_DEV -f Dockerfile .
    echo "Development build completed successfully!"
}

# Function to build the release Docker image
build_release_image() {
    echo "Building Smoldyn GPU release Docker image..."
    docker build -t $IMAGE_NAME_RELEASE -f Dockerfile.release .
    echo "Release build completed successfully!"
}

# Function to run the container
run_container() {
    if [ $# -eq 0 ]; then
        echo "Error: Please specify a configuration file"
        exit 1
    fi
    
    CONFIG_FILE="$1"
    
    # Get absolute path of the config file
    if [[ "$CONFIG_FILE" = /* ]]; then
        # Already absolute path
        ABS_CONFIG_PATH="$CONFIG_FILE"
    else
        # Convert relative path to absolute
        ABS_CONFIG_PATH="$(pwd)/$CONFIG_FILE"
    fi
    
    # Check if config file exists
    if [ ! -f "$ABS_CONFIG_PATH" ]; then
        echo "Error: Configuration file '$CONFIG_FILE' not found"
        exit 1
    fi
    
    # Extract just the filename from the path
    CONFIG_FILENAME=$(basename "$ABS_CONFIG_PATH")
    
    echo "Running Smoldyn GPU with configuration: $CONFIG_FILENAME"
    
    # Remove existing container if it exists
    docker rm -f $CONTAINER_NAME 2>/dev/null || true
    
    # Run the container with GPU support
    docker run --gpus all \
               -it \
               --name $CONTAINER_NAME \
               --rm \
               -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
               -e DISPLAY=$DISPLAY \
               -v $(pwd):/data \
               -v "$ABS_CONFIG_PATH:/workspace/smoldyn-gpu/$CONFIG_FILENAME" \
               $IMAGE_NAME_RELEASE \
               $CONFIG_FILENAME
}

# Function to start interactive shell
start_shell() {
    echo "Starting interactive shell in Smoldyn GPU development container..."
    
    # Remove existing container if it exists
    docker rm -f $CONTAINER_NAME 2>/dev/null || true
    
    docker run --gpus all \
               --name $CONTAINER_NAME \
               --rm \
               -it \
               -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
               -v "$PWD:/workspace" \
               -e DISPLAY=$DISPLAY \
               $IMAGE_NAME_DEV \
               /bin/bash
}

# Function to start persistent development container
start_dev_container() {
    echo "Starting persistent development container..."
    
    # Start container in detached mode with a persistent command
    docker run --gpus all \
               --name $DEV_CONTAINER_NAME \
               -d \
               -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
               -v "$PWD:/workspace" \
               -e DISPLAY=$DISPLAY \
               --env="NVIDIA_DRIVER_CAPABILITIES=all" \
               $IMAGE_NAME_DEV \
               tail -f /dev/null
               
    echo "Development container started successfully!"
    echo "Container name: $DEV_CONTAINER_NAME"
    echo "To access the container, run: docker exec -it $DEV_CONTAINER_NAME /bin/bash"
    echo "To stop the container, run: docker stop $DEV_CONTAINER_NAME"
}

# Function to clean up
clean_up() {
    echo "Cleaning up Docker resources..."
    docker rm -f $CONTAINER_NAME 2>/dev/null || true
    docker rm -f $DEV_CONTAINER_NAME 2>/dev/null || true
    docker rmi $IMAGE_NAME_DEV 2>/dev/null || true
    docker rmi $IMAGE_NAME_RELEASE 2>/dev/null || true
    echo "Cleanup completed!"
}

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo "Error: Docker is not running or not accessible"
    exit 1
fi

# Check if nvidia-docker is available
if ! docker run --rm --gpus all nvidia/cuda:12.2-base-ubuntu20.04 nvidia-smi >/dev/null 2>&1; then
    echo "Warning: GPU support may not be available. Make sure nvidia-docker is installed and configured."
fi

# Parse command line arguments
case "${1:-}" in
    build-dev)
        build_dev_image
        ;;
    build-release)
        build_release_image
        ;;
    run)
        shift
        run_container "$@"
        ;;
    dev)
        start_dev_container
        ;;
    shell)
        start_shell
        ;;
    clean)
        clean_up
        ;;
    *)
        usage
        exit 1
        ;;
esac

#!/bin/sh

rm -rf ./launcher && mkdir -p ./launcher
rsync -arvp src/ ./launcher/
rsync -arvp ../common/ ./launcher/

REGISTROY=linkedin
IMAGE_NAME=tfjob-launcher
VERSION_TAG=0.1
VERSIONED_IMAGE_NAME=${REGISTROY}/${IMAGE_NAME}:${VERSION_TAG}

echo "Building image ${VERSIONED_IMAGE_NAME}"
docker build -t ${VERSIONED_IMAGE_NAME} .

# TODO: uncomment to push to dockerhub
# docker push ${VERSIONED_IMAGE_NAME}

rm -rf ./launcher

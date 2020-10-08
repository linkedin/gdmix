#!/bin/sh

rm -rf ./launcher && mkdir -p ./launcher
rsync -arvp src/ ./launcher/
rsync -arvp ../common/ ./launcher/

# TODO: change to LinkedIn dockerhub endpoint and push
REGISTROY=linkedin
IMAGE_NAME=sparkapplication-launcher
VERSION_TAG=0.1
VERSIONED_IMAGE_NAME=${IMAGE_NAME}:${VERSION_TAG}
REMOTE_IMAGE_NAME=${REGISTROY}/${VERSIONED_IMAGE_NAME}

docker build -t ${VERSIONED_IMAGE_NAME} .

# TODO: tag and push to dockerhub
# docker tag ${VERSIONED_IMAGE_NAME} ${REMOTE_IMAGE_NAME}
# docker push ${REMOTE_IMAGE_NAME}

rm -rf ./launcher

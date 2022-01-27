#!/bin/sh

REGISTROY=linkedin
IMAGE_NAME=gdmix-dev
VERSION_TAG=0.4.0
VERSIONED_IMAGE_NAME=${REGISTROY}/${IMAGE_NAME}:${VERSION_TAG}

echo "Building image ${VERSIONED_IMAGE_NAME}"
docker build -t ${VERSIONED_IMAGE_NAME} .

# TODO: uncomment to push to docker hub
# docker push ${VERSIONED_IMAGE_NAME}

rm -rf *.config *.py

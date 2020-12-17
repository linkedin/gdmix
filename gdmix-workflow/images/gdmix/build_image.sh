#!/bin/sh

rsync -arvp ../../examples/movielens-100k/*.yaml .
rsync -arvp ../../../scripts/download_process_movieLens_data.py .

REGISTRY=linkedin
IMAGE_NAME=gdmix
VERSION_TAG=0.3
VERSIONED_IMAGE_NAME=${REGISTRY}/${IMAGE_NAME}:${VERSION_TAG}

echo "Building image ${VERSIONED_IMAGE_NAME}"
docker build --squash -t ${VERSIONED_IMAGE_NAME} .

# TODO: uncomment to push to docker hub
# docker push ${VERSIONED_IMAGE_NAME}

rm -rf *.yaml *.py

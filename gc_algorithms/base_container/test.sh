#!/usr/bin/env bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

./build.sh

VOLUME_SUFFIX=$(dd if=/dev/urandom bs=32 count=1 | md5sum | cut -c 1-10)

MODEL=swin_unetr.tar.gz

DOCKER_FILE_SHARE=picai_base_algorithm-output-$VOLUME_SUFFIX
docker volume create $DOCKER_FILE_SHARE

CONTAINER_NAME="my_processor_container" 

docker run --gpus all \
        --name $CONTAINER_NAME \
        -v $SCRIPTPATH/test/:/input/ \
        -v $DOCKER_FILE_SHARE:/output/ \
        -v $SCRIPTPATH/output/:/host_output/ \
        -v $SCRIPTPATH/models/$MODEL:/tmp/$MODEL \
        --entrypoint "" picai_base_algorithm /bin/bash -c "tar -xzvf /tmp/$MODEL -C /opt/ml/model && python process.py"

docker cp $CONTAINER_NAME:/output/images/cspca-detection-map/cspca_detection_map.mha ./output/cspca_detection_map.mha
docker cp $CONTAINER_NAME:/output/images/prostate/10032_1000032_t2w_prostate.mha ./output/10032_1000032_t2w_prostate.mha
docker cp $CONTAINER_NAME:/output/cspca-case-level-likelihood.json ./output/cspca-case-level-likelihood.json
docker rm $CONTAINER_NAME

# check detection map (at /output/images/cspca-detection-map/cspca_detection_map.mha)
docker run --rm \
        -v $SCRIPTPATH/test/:/input/ \
        -v $DOCKER_FILE_SHARE:/output/ \
        --entrypoint "" picai_umamba_inferer python -c "import sys; import json; import numpy as np; import SimpleITK as sitk; f1 = sitk.GetArrayFromImage(sitk.ReadImage('/output/images/cspca-detection-map/cspca_detection_map.mha')); f2 = sitk.GetArrayFromImage(sitk.ReadImage('/input/cspca-detection-map/cspca_detection_map.mha')); print('max. difference between prediction and reference:', np.abs(f1-f2).max()); sys.exit(int(np.abs(f1-f2).max() > 1e-3));"

if [ $? -eq 0 ]; then
    echo "Detection map test successfully passed..."
else
    echo "Expected detection map was not found..."
fi

# check case_confidence (at /output/cspca-case-level-likelihood.json)
docker run --rm \
        -v $DOCKER_FILE_SHARE:/output/ \
        -v $SCRIPTPATH/test/:/input/ \
        --entrypoint "" picai_umamba_inferer python -c "import sys; import json; f1 = json.load(open('/output/cspca-case-level-likelihood.json')); f2 = json.load(open('/input/cspca-case-level-likelihood.json')); print('Found case-level prediction ' + str(f1) + ', expected ' +str(f2)); sys.exit(int(abs(f1-f2) > 1e-3));"

if [ $? -eq 0 ]; then
    echo "Case-level prediction test successfully passed..."
else
    echo "Expected case-level prediction was not found..."
fi



docker volume rm $DOCKER_FILE_SHARE

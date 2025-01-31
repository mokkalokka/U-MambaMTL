#!/usr/bin/env bash

./build.sh

docker save picai_base_algorithm | gzip -c > picai_base_algorithm_1.0.tar.gz

#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage: $0 <small|middle|big>"
    exit 1
fi

if [[ "$1" != "small" && "$1" != "middle" && "$1" != "big" ]]; then
    echo "Error: Invalid argument. Please use 'small', 'middle', or 'big'"
    exit 1
fi

rm -rf build
rm -f ./bin/MyTest

PROJECT_ROOT=$(pwd)

mkdir -p build
cd build

cmake ..
make

if [ -f "${PROJECT_ROOT}/bin/MyTest" ]; then
    echo "Running with parameter: $1"
    ${PROJECT_ROOT}/bin/MyTest "$1"
else
    echo "error:cannot find ${PROJECT_ROOT}/bin/MyTest"
    find ${PROJECT_ROOT} -name "MyTest" -type f 2>/dev/null
fi

cd ${PROJECT_ROOT}
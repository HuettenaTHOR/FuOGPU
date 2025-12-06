#!/bin/bash

# Compile the CUDA file
nvcc --Werror all-warnings HW6/homework_6.cu -o HW6/compiled.out

# Check if compilation succeeded
if [ $? -eq 0 ]; then
    ./HW6/compiled.out
else
    echo "Compilation failed. Sybau."
fi

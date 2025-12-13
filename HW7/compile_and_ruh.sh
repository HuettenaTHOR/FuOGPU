#!/bin/bash

# Compile the CUDA file
nvcc --Werror all-warnings HW7/homework_7.cu -o HW7/compiled.out

# Check if compilation succeeded
if [ $? -eq 0 ]; then
    ./HW7/compiled.out
else
    echo "Compilation failed. Sybau."
fi

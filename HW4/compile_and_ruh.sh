#!/bin/bash

# Compile the CUDA file
nvcc --Werror all-warnings HW4/homework_4.cu -o HW4/compiled.out

# Check if compilation succeeded
if [ $? -eq 0 ]; then
    ./HW4/compiled.out
else
    echo "Compilation failed. Sybau."
fi

#!/bin/bash

# Compile the CUDA file
nvcc --Werror all-warnings HW5/homework_5.cu -o HW5/compiled.out

# Check if compilation succeeded
if [ $? -eq 0 ]; then
    ./HW5/compiled.out
else
    echo "Compilation failed. Sybau."
fi

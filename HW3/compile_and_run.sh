#!/bin/bash

# Compile the CUDA file
nvcc HW3/homework_3.cu -o HW3/compiled.out

# Check if compilation succeeded
if [ $? -eq 0 ]; then
    echo "Compilation successful. Running program..."
    ./HW3/compiled.out
else
    echo "Compilation failed. Sybau."
fi

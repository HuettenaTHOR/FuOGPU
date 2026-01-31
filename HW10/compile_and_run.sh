#!/bin/bash

# Compile the CUDA file
nvcc --Werror all-warnings HW10/homework_10.cu -o HW10/compiled.out

# Check if compilation succeeded
if [ $? -eq 0 ]; then
    echo "Compilation successful. Running..."
    ./HW10/compiled.out
else
    echo "Compilation failed."
fi

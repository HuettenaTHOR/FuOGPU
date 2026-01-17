#!/bin/bash

# Compile the CUDA file
nvcc --Werror all-warnings HW8/homework_8.cu -o HW8/compiled.out

# Check if compilation succeeded
if [ $? -eq 0 ]; then
    echo "Compilation successful. Running..."
    ./HW8/compiled.out
else
    echo "Compilation failed."
fi

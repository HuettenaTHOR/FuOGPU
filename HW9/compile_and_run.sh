#!/bin/bash

# Compile the CUDA file
nvcc --Werror all-warnings HW9/homework_9.cu -o HW9/compiled.out

# Check if compilation succeeded
if [ $? -eq 0 ]; then
    echo "Compilation successful. Running..."
    ./HW9/compiled.out
else
    echo "Compilation failed."
fi

#!/bin/bash

# Compile the md file into a PDF
pandoc HW8/discussion.md -o HW8/discussion.pdf --pdf-engine=xelatex

pandoc HW8/results.md -o HW8/results.pdf --pdf-engine=xelatex

# Check if compilation succeeded
if [ $? -eq 0 ]; then
    echo "PDF compilation successful."
else
    echo "PDF compilation failed."
fi

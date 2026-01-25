#!/bin/bash

# Compile the md file into a PDF
pandoc HW9/discussion.md -o HW9/discussion.pdf --pdf-engine=xelatex

pandoc HW9/results.md -o HW9/results.pdf --pdf-engine=xelatex

# Check if compilation succeeded
if [ $? -eq 0 ]; then
    echo "PDF compilation successful."
else
    echo "PDF compilation failed."
fi

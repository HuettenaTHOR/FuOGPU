#!/bin/bash

# Compile the md file into a PDF
pandoc HW10/discussion.md -o HW10/discussion.pdf --pdf-engine=xelatex

pandoc HW10/results.md -o HW10/results.pdf --pdf-engine=xelatex

# Check if compilation succeeded
if [ $? -eq 0 ]; then
    echo "PDF compilation successful."
else
    echo "PDF compilation failed."
fi

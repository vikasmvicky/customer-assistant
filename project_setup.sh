#!/bin/bash

mkdir -p src         # creating source directory
mkdir -p research    # creating research directory

# creating files
touch src/__init__.py
touch src/helper.py
touch src/prompt.py
touch .env
touch setup.py
touch app.py
touch research/trails.ipynb
touch README.md

echo "Directory and files created successfully."

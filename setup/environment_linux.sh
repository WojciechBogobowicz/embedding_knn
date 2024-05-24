#!/bin/bash

# Create a virtual environment
python3 -m venv env

# Activate the virtual environment
source env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install the required packages
pip install -r setup/requirements.txt

# Ask the user if they want to install demo requirements
read -p "Do you also want to install requirements requred to run demo? (y/n): " install_demo

if [[ "$install_demo" == [yY] ]]; then
    pip install -r setup/demo_requirements.txt
    echo "Demo requirements installed."
else
    echo "Skipping demo requirements installation."
fi

echo "Environment setup complete. To activate the virtual environment, run 'source env/bin/activate'."

import pickle
import os

# Get the current script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Specify the path to your pickle file relative to the script's directory
model_path = os.path.join(script_dir, 'model.pkl')

# Open and load the pickle file
with open(model_path, 'rb') as file_1: 
  model = pickle.load(file_1)

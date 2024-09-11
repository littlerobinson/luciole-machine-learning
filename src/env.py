import sys
import numpy as np
import pandas as pd
import plotly
import sklearn
import tensorflow as tf
import pkg_resources

print("Python version:", sys.version)
print("Version info:", sys.version_info)
print("NumPy version:", np.__version__)
print("Pandas version:", pd.__version__)
print("Plotly version:", plotly.__version__)
print("scikit-learn version:", sklearn.__version__)
print("TensorFlow version:", tf.__version__)
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

print('─' * 10) 

installed_packages = pkg_resources.working_set
for package in installed_packages:
    print(f'{package.key}=={package.version}')
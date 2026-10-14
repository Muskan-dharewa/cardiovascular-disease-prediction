# This script is run in Google Colab instead of locally, because the 'rtdl' package
# (needed for the FT-Transformer model) fails to install on Python 3.13 / Windows —
# it requires numpy<2, which has no pre-built wheel for 3.13 and needs a C++ compiler
# to build from source.
# Colab notebook:https://colab.research.google.com/drive/1EXlIy6wAfSN7sDCjYasW88cqrxXU2283?usp=sharing
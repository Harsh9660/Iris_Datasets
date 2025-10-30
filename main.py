import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt 
import seaborn as sns
import plotly.express as px

path = "Iris.csv"

df = pd.read_csv(path)
print(df.head())
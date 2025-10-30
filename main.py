import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from matplotlib.pyplot import plt 
import seaborn as sns

path = "Iris.csv"

df = pd.read_csv(path)
print(df.head())
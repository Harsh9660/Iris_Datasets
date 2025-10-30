import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler # Removed LabelEncoder
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

path = "Iris.csv"
target_col = 'Species'

df = pd.read_csv(path)

print(df.head())

print("\n Dataset Info:")
print(df.info())

print("\n Species Summary:")
print(df[target_col].value_counts())

print("\n Missing data from the datasets:", df.isnull().sum())

df = df.drop('Id', axis=1)





scaler = StandardScaler()

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

df[num_cols] = scaler.fit_transform(df[num_cols])

X = df.drop(target_col, axis=1)
y = df[target_col] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n Preprocessed Dataset shape:", df.shape)
print(" Training set shape:", X_train.shape)
print(" Test set shape:", X_test.shape)


fig1 = px.histogram(df, x=target_col, title='Distribution of Species', color_discrete_sequence=['teal'])
fig1.update_layout(title_x=0.5)
fig1.show()


corr = df.corr(numeric_only=True)
fig2 = px.imshow(corr, text_auto=True, color_continuous_scale='Viridis', title='Correlation Heatmap')
fig2.update_layout(title_x=0.5)
fig2.show()


fig3 = px.scatter(df, x='SepalLengthCm', y='SepalWidthCm', color=target_col, title='Sepal Length vs Sepal Width by Species')
fig3.update_traces(marker=dict(size=10, opacity=0.7))
fig3.update_layout(title_x=0.5)
fig3.show()

fig4 = px.scatter(df, x='PetalLengthCm', y='PetalWidthCm', color=target_col, title='Petal Length vs Petal Width by Species')
fig4.update_traces(marker=dict(size=10, opacity=0.7))
fig4.update_layout(title_x=0.5)
fig4.show()


sizes = [len(X_train), len(X_test)]
labels = ['Train', 'Test']
fig5 = px.pie(names=labels, values=sizes, title='Train vs Test Data Split', hole=0.4)
fig5.update_traces(textinfo='label+percent')
fig5.update_layout(title_x=0.5)
fig5.show()


train_labels = pd.DataFrame({'Set': 'Train', 'Target': y_train})
test_labels = pd.DataFrame({'Set': 'Test', 'Target': y_test})
split_df = pd.concat([train_labels, test_labels])
fig6 = px.histogram(split_df, x='Target', color='Set', barmode='group', title='Target Distribution in Train vs Test Sets')
fig6.update_layout(title_x=0.5)
fig6.show()
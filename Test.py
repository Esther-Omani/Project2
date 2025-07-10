#Importing libraries
import pandas as pd
import matplotlib as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

#Loading the dataset
df = pd.read_csv("Mall_Customers.csv")
df.head()
print (df.head())
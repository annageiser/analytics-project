# Load libraries
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Load data
train = pd.read_csv('../data/raw/train.csv', sep='|')
items = pd.read_csv('../data/raw/items.csv', sep='|')

df = train.merge(items, 
                   on='pid', 
                   how='left',
                   validate="m:1")


target_col = 'order'
numerical_cols = []
categorical_cols = []
high_cardinality_cols = []

# %%

from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
pd.options.display.max_columns = None

# %%

df_origin = pd.read_csv('../Data/train.csv')
df = df_origin.copy()
df.shape

# %%

df.columns, len(df.columns)

# %%

# The dependent variable is balanced
df['class'].value_counts()

# %%

# I see non NA values
# And some features as objects that can be transformed to categories
df.info()

# %%

for c in df.select_dtypes('object').columns:
    df[c] = df[c].astype('category')

df.info()

# %%

# Check the feature independece between them

custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)

# Compute the correlation matrix
corr = df.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.show()

# %%

# It allows me to identify outliers

def print_boxplot_grid(df):
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    i = 0
    for c in df.columns:
        if (df[c].dtype == 'category'):
            continue

        ax = fig.add_subplot(4, 4, i + 1)
        sns.boxplot(data=df, y='class', x=c, ax=ax)
        i += 1
    fig.set_size_inches(12, 8)
    plt.show()


print_boxplot_grid(df)

# %%
# Removing outliers

# 1. Body fat % outlier

idx = df[(df['class'] == 'A') & (df['body fat_%'] > 50)].index
print('Drop gordos...\n', df.iloc[idx, :])
df = df.drop(index=idx)
print_boxplot_grid(df)
df = df.reset_index(drop=True)

# %%

# 2. Heigh cm outlier

idx = df[(df['class'] == 'B') & (df['height_cm'] < 140)].index
print('Drop bajos...\n', df.iloc[idx, :])
df = df.drop(index=idx)
print_boxplot_grid(df)
df = df.reset_index(drop=True)

# %%

# 3. Sit and bend forward_cm

idx = df[(df['class'] == 'A') & (df['sit and bend forward_cm'] > 100)].index
print('Drop elasticos...\n', df.iloc[idx, :])
df = df.drop(index=idx)
print_boxplot_grid(df)
df = df.reset_index(drop=True)

# %%

# 4. Systolic

idx = df[(df['class'] == 'A') & (df['systolic'] < 70)].index
print('Drop systoloc?...\n', df.iloc[idx, :])
df = df.drop(index=idx)
df = df.reset_index(drop=True)

idx = df[(df['class'] == 'A') & (df['systolic'] > 190)].index
print('Drop systoloc?...\n', df.iloc[idx, :])
df = df.drop(index=idx)
df = df.reset_index(drop=True)

idx = df[(df['class'] == 'B') & (df['systolic'] < 70)].index
print('Drop systoloc?...\n', df.iloc[idx, :])
df = df.drop(index=idx)
df = df.reset_index(drop=True)

idx = df[(df['class'] == 'B') & (df['systolic'] > 190)].index
print('Drop systoloc?...\n', df.iloc[idx, :])
df = df.drop(index=idx)
df = df.reset_index(drop=True)

idx = df[(df['class'] == 'C') & (df['systolic'] > 190)].index
print('Drop systoloc?...\n', df.iloc[idx, :])
df = df.drop(index=idx)
df = df.reset_index(drop=True)

print_boxplot_grid(df)

# %% Diastolic

idx = df[(df['class'] == 'B') & (df['diastolic'] < 30)].index
print('Drop diastolic?...\n', df.iloc[idx, :])
df = df.drop(index=idx)
df = df.reset_index(drop=True)

idx = df[(df['class'] == 'D') & (df['diastolic'] < 40)].index
print('Drop diastolic?...\n', df.iloc[idx, :])
df = df.drop(index=idx)
df = df.reset_index(drop=True)

print_boxplot_grid(df)

# %%

# broad jump

# idx = df[(df['broad jump_cm'] < 5)].index
# print('Drop jumping?...\n', df.iloc[idx, :])
# df = df.drop(index=idx)
# df = df.reset_index(drop=True)
# print_boxplot_grid(df)

# %%

df.shape

# %%

# The id identifies the entry, so can be removed
len(df['id'].unique()) == df.shape[0]

# %%

df = df.drop('id', axis=1, errors='ignore')
df.info()

# %%

# PSS. transform categorical dependent variable into numbers

X = df.drop('class', axis=1, errors='ignore')
gender = X['gender']
gender_ohc = OneHotEncoder(drop='first')
gender = gender_ohc.fit_transform(gender.values.reshape(-1, 1))
X['man'] = pd.Series(gender.toarray().squeeze())
X = X.drop('gender', axis=1, errors='ignore')

X.tail()

# %%

y = df['class']
le = preprocessing.LabelEncoder()
y = pd.Series(le.fit_transform(y))

y.tail()

# %%

# Save data as preprocessed file

preprocessed = X.copy()
preprocessed['man'] = preprocessed['man'].astype('int')
preprocessed['class'] = y.copy()
preprocessed.to_csv('../Data/preprocessed.csv', index=False)

print(preprocessed.tail())

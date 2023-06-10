```python
import pandas as pd
from sklearn.model_selection import train_test_split

# load data
data = pd.read_csv('/Users/devanhall/Downloads/melb_data.csv')

# select target
y = data.Price

# using only numerical predictors 
melb_predictors = data.drop(['Price'], axis=1)
X = melb_predictors.select_dtypes(exclude=['object'])

# divide data into training and validation subsets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                     random_state=0)
```


```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# function for comparing different approaches
def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=10, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)
```


```python
# get names of the columns with missing values
cols_with_missing = [col for col in X_train.columns
                    if X_train[col].isnull().any()]

# drop columns in training and validation data
reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_valid = X_valid.drop(cols_with_missing, axis=1)
print("MAE from approach 1 (Drop columns with missing values): ")
print(score_dataset(reduced_X_train, reduced_X_valid, y_train, y_valid))
```

    MAE from approach 1 (Drop columns with missing values): 
    183550.22137772635



```python
from sklearn.impute import SimpleImputer

# imputation 
my_imputer = SimpleImputer()
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
imputed_X_valid = pd.DataFrame(my_imputer.fit_transform(X_valid))

# Imputation removed column names put them back
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns 

print("MAE from approach 2 (IMPUTATION): ")
print(score_dataset(imputed_X_train, imputed_X_valid, y_train, y_valid))
```

    MAE from approach 2 (IMPUTATION): 
    179816.89508731329



```python
# make a copy to avoid changing original data

X_train_plus = X_train.copy()
X_valid_plus = X_valid.copy()

# make new col's indicating what will be imputed
for col in cols_with_missing:
    X_train_plus[col + '_was_missing'] = X_train_plus[col
                                                     ].isnull()
    X_valid_plus[col + '_was_missing'] = X_valid_plus[col
                                                     ].isnull()
# imputation
my_imputer = SimpleImputer()
imputed_X_train_plus = pd.DataFrame(my_imputer.fit_transform(
X_train_plus))
imputed_X_valid_plus = pd.DataFrame(my_imputer.transform(
X_valid_plus))

# imputation removed column names; put them back
imputed_X_train_plus.columns = X_train_plus.columns
imputed_X_valid_plus.columns = X_valid_plus.columns

print("MAE from approach 3 (An extension to Imputation):")
print(score_dataset(imputed_X_train_plus, imputed_X_valid_plus, y_train, y_valid))

```

    MAE from approach 3 (An extension to Imputation):
    178927.503183954



```python
print(X_train.shape)

missing_val_count_by_column = (X_train.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0])
```

    (10864, 12)
    Car               49
    BuildingArea    5156
    YearBuilt       4307
    dtype: int64


# Imputation performed better in terms of accuracy compared to simply dropping columns with null data values


```python

```

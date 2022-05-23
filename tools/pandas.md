# Pandas Note
## How to group dataframe rows into list in pandas groupby？
```python=
gp_datas = df.groupby('a').agg({'b':lambda x: list(x)})
# get the value
gp_datas.shop_tag[A_VALUE]
```
![](https://i.imgur.com/6KGksaj.png)


## Replace values based on index
```python=
df.loc[index, '<column_name>'] = value
df.loc[index_start:index_end, '<column_name>'] = value
```

## Create a dictionary of two pandas series
```python=
pd.Series(df[val_name].values,index=df[key_name]).to_dict()
```

## Shift for lagged data
```python=
df['diff'] = df['sales'] - df.shift(1)['sales']
```
### Reference
- https://medium.com/@NatalieOlivo/use-pandas-to-lag-your-timeseries-data-in-order-to-examine-causal-relationships-f8186451b3a9
- https://medium.com/@NatalieOlivo/humanitarian-aid-using-womens-completion-of-secondary-education-to-measure-aid-effectiveness-9da479ce7336
- https://towardsdatascience.com/all-the-pandas-shift-you-should-know-for-data-analysis-791c1692b5e

## Replace Columns Name
```python=
df.columns = df.columns.str.replace('<target>', '<replace-string>')
```

## Concat 2 DataFrame
```python=
concat_df = pd.concat([df1, df2]).reset_index()
```
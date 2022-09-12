###### tags: `Python`
# Python Tips
## Mapping
- 元陣列
```python=

a = [
    [[1, 0.2], [7, 0.4], [8, 0.4]],
    [[18, 0.3], [25, 0.4], [7, 0.4]],
    [[5, 0.6], [7, 0.09], [2, 0.31]]
]
```
- Question: 如何取出資料並呈現以下形式？
    ```python=
    [
        [1, 7],
        [18, 25],
        [5, 7]
    ]
    ```
    - 解法步驟：
        - 在深度1時，先取出目標的前n筆資料
        - 在深度2時，取出陣列的第1筆資料
- Solution
    - step 1:
        ```python=
        list(map(lambda x: x[:2]), a)
        ```
        - ouput:
            ```python=
            [
                [[1, 0.2], [7, 0.4]], 
                [[18, 0.3], [25, 0.4]], 
                [[5, 0.6], [7, 0.09]]
            ]
            ```
    - step 2:
        ```python=
        list(map(lambda x: list(map( y:y[0], x[:2])), a))
        ```
- Benefit    
    ```python=
    x = map(lambda x: np.array(x)[:, 0][:3], a)
    ```
## Python Array
```python=
arr[::2]     # get even rows
arr[1::2]    # get odd rows
```
## Python env
- Generate pipenv's requirments.txt
    ```
    pipenv lock -r > requirements.txt
    ```

## Python dict keys
```python
item2cat = pd.Series(train_df['new_item_id'].values,index=train_df['item_id']).to_dict()
user2cat = pd.Series(train_df['new_user_id'].values,index=train_df['user_id']).to_dict()
cat2item = {v:k for k,v in item2cat.items()}
cat2user = {v:k for k,v in user2cat.items()}
```

## Best way to transform category
- [Ref](https://stackoverflow.com/questions/39475187/how-to-speed-labelencoder-up-recoding-a-categorical-variable-into-integers)
- Compare
    | LabelEncoder                                                          | astype('category').cat.codes                                |
    | --------------------------------------------------------------------- | ----------------------------------------------------------- |
    | CPU times: user 6.28 s, sys: 48.9 ms, total: 6.33s, Wall time: 6.37 s | user 301 ms, sys: 28.6 ms, total: 330 ms, Wall time: 331 ms |
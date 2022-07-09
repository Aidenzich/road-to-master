#%%


from tomlkit import string



data = df_train['欄位名稱']
all = ""
for i in data: 
    all += i

all = list(set(" ".join(all).split(" ")))
all

# %%

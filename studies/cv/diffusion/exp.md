# Experiments
some records
### p sample
$$
P_\theta
$$
- reverse
- p_losses
    - model 接 x_noisy 與 time torch
    - x_noisy 來自 q_sample 自 batch image
    - 算 Model 預測的 noisy 與 x_noisy 間的誤差

### q sample
- forward
- fixed

#### schedule
- Pre-difined timestep number
- linear_beta_schedule

#### sample from noise
- noise's shape equals to input's shape

### Inference
- p_sample_loop 回傳 timestep 長度的影像陣列
    - 每個元素都是該 step 下的圖片
    - 在 inference 也是跑跟訓練時相同的timestep (實際上也可以跳過部分step來訓練)
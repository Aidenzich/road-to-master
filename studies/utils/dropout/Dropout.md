# Dropout
- Schemetic Diagram
    ![dropout](https://i.imgur.com/cxwFdJG.png)
- During training, ==randomly zeroes some of the elements== of the input tensor with probability p using samples from a ***Bernoulli distribution***. 
- Each channel will be zeroed out independently on every forward call.
  - `p` or `p_emd`
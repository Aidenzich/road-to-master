## max_norm
- Schemetic Diagram
    ![Clipping Gradients](https://i.imgur.com/7zRko9w.jpg)
    $$
        W = W - \eta \dfrac{\partial\mathcal{L}}{\partial W}
    $$
- **Purpose** :To prevent Exploding Gradient Problem
- **Heine Borel**
- **Gradient Clipping**
    $$
        \dfrac{\partial \mathcal{L}}{\partial W} \in [-M, M]
    $$
- **limit**
    $$
    \lim_{n \to -\infty} a_n = x \\
    \forall \epsilon \in \mathbb{R}, \exists N \in X 
    $$
- **Code**
    - [torch.nn.utils.clip_grad_norm](https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html)
    - Use Gradient Clipping to limit the gradient.
    $$\partial L/\partial W_n \in [-M, M ]$$
    
### pytorch
- [clip_grad_norm_](https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html)
- Ref: https://machinelearningjourney.com/index.php/2020/08/07/vanishing-and-exploding-gradients/
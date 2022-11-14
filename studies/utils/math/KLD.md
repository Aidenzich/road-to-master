$$ 
\begin{aligned}  
D_\text{KL}( q_\phi({z}|{x}) \| p_\theta({z}|{x}) )  &=\int q_\phi({z}|{x}) \log\frac{q_\phi({z} | {x})}{\color{teal}{p_\theta(z|x)}} d{z}  \\ 
&=\int q_\phi({z} | {x}) \log\frac{q_\phi({z} | {x})\color{teal}{p_\theta(x)}}{\color{teal}{p_\theta(z, x)}} d{z}  \\ 
&=\int q_\phi({z} | {x}) \big( \log p_\theta({x}) + \log\frac{q_\phi({z} | {x})}{p_\theta({z}, {x})} \big) d{z}  \\ 
&=\log p_\theta({x}) + \int q_\phi({z} | {x})\log\frac{q_\phi({z} | {x})}{\color{teal}{p_\theta(z, x)}} d{z} \\ 
&=\log p_\theta({x}) + \int q_\phi({z} | {x})\log\frac{q_\phi({z} | {x})}{\color{teal}{p_\theta(x|z)p_\theta(z)}} d{z} \\ 
&=\log p_\theta({x}) + \mathbb{E}_{{z}\sim q_\phi({z} | {x})}\Big[\log \frac{q_\phi({z} | {x})}{p_\theta({z})} - \log p_\theta({x} | {z})\Big] \\ 
&=\log p_\theta({x}) + D_\text{KL}(q_\phi({z}|{x}) \| p_\theta({z})) - \mathbb{E}_{{z}\sim q_\phi({z}|{x})}\log p_\theta({x}|{z})  
\end{aligned} 
$$
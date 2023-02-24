# Fields
## 4 operations
- `+` Addition 
- `·` Multiplication 
- `-` Subtraction 
- `÷` Division

## Conditions
#### F.1 `Commutativity` of addition and multiplication 

$$
a + b = b + a  \quad \text{and} \quad a \cdot b = b \cdot a
$$

- `交換律`

### F.2 `Associativity` of addition and multiplication 

$$
(a + b) + c = a + (b + c) \quad \text{and} \quad (a \cdot b) \cdot c = a \cdot (b \cdot c)
$$

- `結合律`

### F.3 `Existence of identity elements` for addition and multiplication 

$$
    0 + a = a \quad \text{and} \quad 1 \cdot a = a
$$

- `恆等單位`
### F.4 `Existence of inverses` for addition and multiplication 

$$
    a + c = 0 \quad \text{and} \quad b \cdot d = 1
$$

- `反元素`
### F.5 `Distributivity` of multiplication over addition 

$$
    a \cdot (b+c) = a \cdot b + a \cdot c
$$

- `分配律`
## Theorem
### C.1 Cancellation Laws
For arbitrary elements a, b, and c in a field, the following statements are ture.
- (a) $\text{If} \ a+b = c+b, \ \text{then} \ a=c$ 
- (b) $\text{If} a \cdot b = c \cdot b \ and \ b \neq 0, \text{then} \ a = c$ 
### C.2
Let a and b be arbitrary elements of a field. Then each of the following statements are true.
- (a) $a \cdot 0 = 0$
- (b) $(-a) \cdot b = a \cdot (-b) = -(a \cdot b)$
- (c) $(-a) \cdot (-b) = a \cdot b$
#### Prove
$$
\forall a \in F,  \quad (-1) \cdot a = -a \\
\begin{align}
a + (-1)a &= [1+(-1)]a \\
&= 0 \cdot a \\
&= 0 \\
&= a + (-a) \\
&\Rightarrow (-1)a = -a
\end{align}
$$

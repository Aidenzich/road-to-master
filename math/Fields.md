# Fields

## 4 operations
- `+` Addition 
- `·` Multiplication 
- `-` Subtraction 
- `÷` Division

## Conditions
### F.1 `Commutativity` of addition and multiplication `交換律`
$$
a + b = b + a  \quad \text{and} \quad a \cdot b = b \cdot a
$$
### F.2 `Associativity` of addition and multiplication `結合律`
$$
(a + b) + c = a + (b + c) \quad \text{and} \quad (a \cdot b) \cdot c = a \cdot (b \cdot c)
$$
### F.3 `Existence of identity elements` for addition and multiplication `恆等單位`
$$
    0 + a = a \quad \text{and} \quad 1 \cdot a = a
$$
### F.4 `Existence of inverses` for addition and multiplication `反元素`
$$
    a + c = 0 \quad \text{and} \quad b \cdot d = 1
$$
### F.5 `Distributivity` of multiplication over addition `分配律`
$$
    a \cdot (b+c) = a \cdot b + a \cdot c
$$
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
#### prove(Need reconstruct)
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

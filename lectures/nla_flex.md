---
title: "Some NLA practice"
author: Daniil Merkulov
institute: Numerical Linear Algebra. Skoltech
format: 
    beamer:
        pdf-engine: pdflatex
        aspectratio: 169
        fontsize: 9pt
        section-titles: true
        incremental: true
        include-in-header: ../files/header_nla.tex  # Custom LaTeX commands and preamble
header-includes:
  - \newcommand{\bgimage}{../files/back9.jpeg}
---

# Lectures 7-8 recap

## Matrix decompositions and linear systems

In a least-squares, or linear regression, problem, we have measurements $X \in \mathbb{R}^{m \times n}$ and $y \in \mathbb{R}^{m}$ and seek a vector $\theta \in \mathbb{R}^{n}$ such that $X \theta$ is close to $y$. Closeness is defined as the sum of the squared differences: 
$$ 
\sum\limits_{i=1}^m (x_i^\top \theta - y_i)^2 \qquad \|X \theta - y\|^2_2 \to \min_{\theta \in \mathbb{R}^{n}} \qquad X \theta^* = y
$$

![Illustration of linear system aka least squares](lls_idea.pdf)

## Matrix decompositions and linear systems. Approaches

### Moore–Penrose inverse
If the matrix $X$ is relatively small, we can write down and calculate exact solution:

$$
\theta^* = (X^\top X)^{-1} X^\top y = X^\dagger y, 
$$

. . .

where $X^\dagger$ is called [pseudo-inverse](https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse) matrix. However, this approach squares the condition number of the problem, which could be an obstacle in case of ill-conditioned huge scale problem. 

. . .

### QR decomposition
For any matrix $X \in \mathbb{R}^{m \times n}$ there is exists QR decomposition:

$$
X = Q \cdot R,
$$

. . .

where  $Q$ is an orthogonal matrix (its columns are orthogonal unit vectors) meaning  $Q^\top Q=QQ^\top=I$ and $R$ is an upper triangular matrix. It is important to notice, that since $Q^{-1} = Q^\top$, we have:

$$
QR\theta = y \quad \longrightarrow \quad R \theta = Q^\top y
$$

Now, process of finding theta consists of two steps:

1. Find the QR decomposition of $X$.
1. Solve triangular system $R \theta = Q^\top y$, which is triangular and, therefore, easy to solve.

## Matrix decompositions and linear systems. Approaches

### Cholesky decomposition
For any positive definite matrix $A \in \mathbb{R}^{n \times n}$ there is exists Cholesky decomposition:

$$
X^\top X = A = L^\top \cdot L,
$$

where  $L$ is an lower triangular matrix. We have:

$$
L^\top L\theta = y \quad \longrightarrow \quad L^\top z_\theta = y
$$

Now, process of finding theta consists of two steps:

1. Find the Cholesky decomposition of $X^\top X$.
1. Find the $z_\theta = L\theta$ by solving triangular system $L^\top z_\theta = y$
1. Find the $\theta$ by solving triangular system $L\theta = z_\theta$

Note, that in this case the error stil proportional to the squared condition number.

## Matrix decompositions and linear systems. Approaches

![Illustration](lls_times.pdf){width=78%}

## Matrix decompositions and linear systems. Non-linear data

![Illustration](non_linear_fit.pdf){width=78%}

## Gram–Schmidt process

**Input:** $n$ linearly independent vectors $u_0, \ldots, u_{n-1}$.

**Output:** $n$ linearly independent vectors, which are pairwise orthogonal $d_0, \ldots, d_{n-1}$.

![Illustration of Gram-Schmidt orthogonalization process](GS1.pdf)

## Gram–Schmidt process {.noframenumbering} 

**Input:** $n$ linearly independent vectors $u_0, \ldots, u_{n-1}$.

**Output:** $n$ linearly independent vectors, which are pairwise orthogonal $d_0, \ldots, d_{n-1}$.

![Illustration of Gram-Schmidt orthogonalization process](GS2.pdf)

## Gram–Schmidt process {.noframenumbering}

**Input:** $n$ linearly independent vectors $u_0, \ldots, u_{n-1}$.

**Output:** $n$ linearly independent vectors, which are pairwise orthogonal $d_0, \ldots, d_{n-1}$.

![Illustration of Gram-Schmidt orthogonalization process](GS3.pdf)

## Gram–Schmidt process {.noframenumbering}

**Input:** $n$ linearly independent vectors $u_0, \ldots, u_{n-1}$.

**Output:** $n$ linearly independent vectors, which are pairwise orthogonal $d_0, \ldots, d_{n-1}$.

![Illustration of Gram-Schmidt orthogonalization process](GS4.pdf)

## Gram–Schmidt process {.noframenumbering}

**Input:** $n$ linearly independent vectors $u_0, \ldots, u_{n-1}$.

**Output:** $n$ linearly independent vectors, which are pairwise orthogonal $d_0, \ldots, d_{n-1}$.

![Illustration of Gram-Schmidt orthogonalization process](GS5.pdf)

## Gram–Schmidt process

:::: {.columns}
::: {.column width="20%"}

![](GS5.pdf)

![](Projection.pdf)

:::

::: {.column width="80%"}

**Input:** $n$ linearly independent vectors $u_0, \ldots, u_{n-1}$.

. . .

**Output:** $n$ linearly independent vectors, which are pairwise orthogonal $d_0, \ldots, d_{n-1}$.
$$
\begin{aligned}
\uncover<+->{ d_0 &= u_0 \\ }
\uncover<+->{ d_1 &= u_1 - \pi_{d_0}(u_1) \\ }
\uncover<+->{ d_2 &= u_2 - \pi_{d_0}(u_2) - \pi_{d_1}(u_2) \\ }
\uncover<+->{ &\vdots \\ }
\uncover<+->{ d_k &= u_k - \sum\limits_{i=0}^{k-1}\pi_{d_i}(u_k) }
\end{aligned}
$$

. . .

$$
d_k = u_k + \sum\limits_{i=0}^{k-1}\beta_{ik} d_i \qquad \beta_{ik} = - \dfrac{\langle d_i, u_k \rangle}{\langle d_i, d_i \rangle}
$$ {#eq-GS}
:::
::::

Here’s how you can structure the final slide to illustrate the **Gram-Schmidt process** in matrix form via QR decomposition:

---

## Gram–Schmidt process in Matrix Form via QR Decomposition {.noframenumbering}

**Step-by-step process in matrix notation:**

- Given a matrix $A$ with columns $u_0, u_1, \ldots, u_{n-1}$, the goal is to decompose $A$ into:
  $$
  A = QR
  $$
  where:
  - $Q$: an orthogonal matrix whose columns are the orthonormal vectors $q_0, q_1, \ldots, q_{n-1}$.
  - $R$: an upper triangular matrix.

. . .

**Illustration:**

$$
v_k = u_k - \sum_{i=0}^{k-1} \langle u_k, q_i \rangle q_i \qquad q_k = \frac{v_k}{\|v_k\|} \qquad R_{ij} = \langle u_j, q_i \rangle \qquad \text{for } i \leq j
$$

$$
\text{For } A = 
\begin{bmatrix}
| & | & & | \\
u_0 & u_1 & \cdots & u_{n-1} \\
| & | & & |
\end{bmatrix}
\quad \rightarrow \quad Q = 
\begin{bmatrix}
| & | & & | \\
q_0 & q_1 & \cdots & q_{n-1} \\
| & | & & |
\end{bmatrix},
\quad R = 
\begin{bmatrix}
r_{00} & r_{01} & \cdots & r_{0(n-1)} \\
0 & r_{11} & \cdots & r_{1(n-1)} \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & r_{(n-1)(n-1)}
\end{bmatrix}
$$

## QR decomposition

![Decomposition](QR.png){width=85%}

## Schur form

![Decomposition](Schur.png)

## QR algorithm

- The QR algorithm was independently proposed in 1961 by Kublanovskaya and Francis.   

- Do not **mix** QR algorithm and QR decomposition!

- QR decomposition is the representation of a matrix, whereas QR algorithm uses QR decomposition to compute the eigenvalues!

## SVD

![Decomposition](QR.png){width=85%}

## Singular value decomposition

Suppose $A \in \mathbb{R}^{m \times n}$ with rank $A = r$. Then $A$ can be factored as

$$
A = U \Sigma V^T 
$$

. . .

where $U \in \mathbb{R}^{m \times m}$ satisfies $U^T U = I$, $V \in \mathbb{R}^{n \times n}$ satisfies $V^T V = I$, and $\Sigma$ is a matrix with non-zero elements on the main diagonal $\Sigma = \text{diag}(\sigma_1, ..., \sigma_r) \in \mathbb{R}^{m \times n}$, such that

. . .

$$
\sigma_1 \geq \sigma_2 \geq \ldots \geq \sigma_r > 0. 
$$

. . .

This factorization is called the **singular value decomposition (SVD)** of $A$. The columns of $U$ are called left singular vectors of $A$, the columns of $V$ are right singular vectors, and the numbers $\sigma_i$ are the singular values. The singular value decomposition can be written as

$$
A = \sum_{i=1}^{r} \sigma_i u_i v_i^T,
$$

where $u_i \in \mathbb{R}^m$ are the left singular vectors, and $v_i \in \mathbb{R}^n$ are the right singular vectors.

## Singular value decomposition

::: {.callout-question}
Suppose, matrix $A \in \mathbb{S}^n_{++}$. What can we say about the connection between its eigenvalues and singular values?
:::

. . .

::: {.callout-question}
How do the singular values of a matrix relate to its eigenvalues, especially for a symmetric matrix?
:::

## Skeleton decomposition

:::: {.columns}

::: {.column width="70%"}
Simple, yet very interesting decomposition is Skeleton decomposition, which can be written in two forms:

$$
A = U V^T \quad A = \hat{C}\hat{A}^{-1}\hat{R}
$$

. . .

The latter expression refers to the fun fact: you can randomly choose $r$ linearly independent columns of a matrix and any $r$ linearly independent rows of a matrix and store only them with the ability to reconstruct the whole matrix exactly.

. . .

Use cases for Skeleton decomposition are:

* Model reduction, data compression, and speedup of computations in numerical analysis: given rank-$r$ matrix with $r \ll n, m$ one needs to store $\mathcal{O}((n + m)r) \ll nm$ elements.
* Feature extraction in machine learning, where it is also known as matrix factorization 
* All applications where SVD applies, since Skeleton decomposition can be transformed into truncated SVD form.
:::

::: {.column width="30%"}
![Illustration of Skeleton decomposition](skeleton.pdf){#fig-skeleton}
:::

::::

## Canonical tensor decomposition

One can consider the generalization of Skeleton decomposition to the higher order data structure, like tensors, which implies representing the tensor as a sum of $r$ primitive tensors.

![Illustration of Canonical Polyadic decomposition](cp.pdf){width=40%}

::: {.callout-example} 
Note, that there are many tensor decompositions: Canonical, Tucker, Tensor Train (TT), Tensor Ring (TR), and others. In the tensor case, we do not have a straightforward definition of *rank* for all types of decompositions. For example, for TT decomposition rank is not a scalar, but a vector.
:::

# Problems

## Example. Simple yet important idea on matrix computations.

Suppose, you have the following expression

$$
b = A_1 A_2 A_3 x,
$$

where the $A_1, A_2, A_3 \in \mathbb{R}^{3 \times 3}$ - random square dense matrices and $x \in \mathbb{R}^n$ - vector. You need to compute b.

Which one way is the best to do it?

1. $A_1 A_2 A_3 x$ (from left to right)
2. $\left(A_1 \left(A_2 \left(A_3 x\right)\right)\right)$ (from right to left)
3. It does not matter
4. The results of the first two options will not be the same.

Check the simple [\faPython code snippet](https://colab.research.google.com/github/MerkulovDaniil/optim/blob/master/assets/Notebooks/stupid_important_idea_on_mm.ipynb) after all.

## Problem 1{.t}

Find SVD of the following matrix:

$$A = \begin{bmatrix} 1\\2\\3 \end{bmatrix}$$ 

. . .

:::: {.columns}
::: {.column width="50%"}
**Solution**

1. Compute $A^T A$:
  $$
  A^T A = \begin{bmatrix} 1 & 2 & 3 \end{bmatrix} \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix} = 1^2 + 2^2 + 3^2 = 14.
  $$
  The singular values $\sigma_i$ are the square roots of the eigenvalues of $A^T A$. Since $A^T A$ is a $1 \times 1$ matrix with value 14, the singular value is $\sigma = \sqrt{14}.$
2. Since $V$ is an $n \times n$ orthogonal matrix ($1 \times 1$ in this case), it can be $V = [1]$ (or $V = [-1]$). We choose $V = [1]$.
:::

::: {.column width="50%"}
3. The simplest form of SVD allows us to write:
  $$
  A = U \Sigma V^T = \begin{bmatrix} \frac{1}{\sqrt{14}} \\ \frac{2}{\sqrt{14}} \\ \frac{3}{\sqrt{14}} \end{bmatrix} \begin{bmatrix} \sqrt{14} \end{bmatrix}  \begin{bmatrix} 1 \end{bmatrix} 
  $$
4. However, if you would like to use another form with square singular matrices:
  $$
  A = U \Sigma V^T = \begin{bmatrix}
  \frac{1}{\sqrt{14}} & \frac{1}{\sqrt{3}} & \frac{-5}{\sqrt{42}} \\
  \frac{2}{\sqrt{14}} & \frac{1}{\sqrt{3}} & \frac{4}{\sqrt{42}} \\
  \frac{3}{\sqrt{14}} & \frac{-1}{\sqrt{3}} & \frac{-1}{\sqrt{42}}
  \end{bmatrix} \begin{bmatrix} \sqrt{14} \\ 0 \\ 0 \end{bmatrix}  \begin{bmatrix} 1 \end{bmatrix} 
  $$

:::
::::






## Problem 2{.t}

Find SVD of the following matrix:
$$A = \begin{bmatrix} 1 & 2\\3 & 3\\2 & 1 \end{bmatrix}$$

## Problem 3{.t}

Find $R$ matrix in QR decomposition for matrix $A = ab^T, \text{where } a = [1, 2, 1, 2, 1, 2, 1], b = [1, 2, 3, 4, 5, 6, 7, 8, 9]$

:::: {.columns}
::: {.column width="50%"}
**Solution**
:::

::: {.column width="50%"}
:::
::::
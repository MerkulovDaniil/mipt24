---
title: "Dimensionality reduction"
author: Daniil Merkulov
institute: Introduction to Data Science. Skoltech
format: 
    beamer:
        pdf-engine: pdflatex
        aspectratio: 169
        fontsize: 9pt
        section-titles: true
        incremental: true
        include-in-header: ../files/header.tex  # Custom LaTeX commands and preamble
header-includes:
  - \newcommand{\bgimage}{../files/back_dim.jpeg}
---

# Dimensionality reduction

## General idea

# PCA

## PCA optimization problem

:::: {.columns}

::: {.column width="40%"}

[![](PCA_anim.png)](http://ids.skoltech.ru/PCA_animation.mp4)

:::
::: {.column width="60%"}

The first component should be defined in order to maximize the projection variance. Suppose, we've already normalized the data, i.e. $\sum_i a_i = 0$, then sample variance will become the sum of all squared projections of data points to our vector ${\mathbf{w}}_{(1)}$, which implies the following optimization problem:

. . .

$$
\mathbf{w}_{(1)}={\underset  {\Vert {\mathbf{w}}\Vert =1}{\operatorname{\arg \,max}}}\,\left\{\sum _{i}\left({\mathbf{a}}^{\top}_{(i)}\cdot {\mathbf{w}}\right)^{2}\right\}
$$
or

. . .

$$
\mathbf{w}_{(1)}={\underset {\Vert \mathbf{w} \Vert =1}{\operatorname{\arg \,max} }}\,\{\Vert \mathbf{Aw} \Vert ^{2}\}={\underset {\Vert \mathbf{w} \Vert =1}{\operatorname{\arg \,max} }}\,\left\{\mathbf{w}^{\top}\mathbf{A^{\top}} \mathbf{Aw} \right\}
$$

. . .

since we are looking for the unit vector, we can reformulate the problem:
$$
\mathbf{w} _{(1)}={\operatorname{\arg \,max} }\,\left\{ \frac{\mathbf{w}^{\top}\mathbf{A^{\top}} \mathbf{Aw} }{\mathbf{w}^{\top}\mathbf{w} }\right\}
$$

. . .

It is [known](https://en.wikipedia.org/wiki/Rayleigh_quotient), that for the positive semidefinite matrix $A^\top A$ such vector is nothing else, but an eigenvector of $A^\top A$, which corresponds to the largest eigenvalue.
:::
::::

## Algorithm derivation

:::: {.columns}

::: {.column width="50%"}
So, we can conclude, that the following mapping:
$$
\underset{n \times k}{\Pi} = \underset{n \times d}{A} \cdot \underset{d \times k}{W} 
$$
describes the projection of data onto the $k$ principal components, where $W$ contains first (by the size of eigenvalues) $k$ eigenvectors of $A^\top A$.

Now we'll briefly derive how SVD decomposition could lead us to the PCA.

Firstly, we write down SVD decomposition of our matrix:
$$
A = U \Sigma W^\top
$$
and to its transpose:
$$
\begin{aligned}
A^\top
&= (U \Sigma W^\top)^\top \\
&= (W^\top)^\top \Sigma^\top U^\top \\
&= W \Sigma^\top U^\top \\
&= W \Sigma U^\top
\end{aligned}
$$
:::

::: {.column width="50%"}
Then, consider matrix $A A^\top$:
$$
\begin{aligned}
A^\top A
&= (W \Sigma U^\top)(U \Sigma V^\top)  \\
&= W \Sigma I \Sigma W^\top \\
&= W \Sigma \Sigma W^\top \\
&= W \Sigma^2 W^\top
\end{aligned}
$$
Which corresponds to the eigendecomposition of matrix $A^\top A$, where $W$ stands for the matrix of eigenvectors of $A^\top A$, while $\Sigma^2$ contains eigenvalues of $A^\top A$.

At the end:
$$
\begin{aligned}
\Pi &= A \cdot W =\\
 &= U \Sigma W^\top W = U \Sigma
\end{aligned}
$$
The latter formula provide us with easy way to compute PCA via SVD with any number of principal components:
$$
\Pi_r = U_r \Sigma_r
$$
:::

::::

## Exercise 1

:::: {.columns}

::: {.column width="40%"}

[![](pca_ex1.png)](https://colab.research.google.com/github/MerkulovDaniil/optim/blob/master/assets/Notebooks/Dimensionality_reduction.ipynb#scrollTo=PzL3k9iZJLc2)

:::
::: {.column width="60%"}
What could be wrong with this PCA?
:::
::::


## Exercise 2

:::: {.columns}

::: {.column width="40%"}

[![](pca_ex2.png)](https://colab.research.google.com/github/MerkulovDaniil/optim/blob/master/assets/Notebooks/Dimensionality_reduction.ipynb#scrollTo=PzL3k9iZJLc2)

:::
::: {.column width="60%"}
What could be wrong with this PCA?
:::
::::

## Exercise 3

:::: {.columns}

::: {.column width="40%"}

[![](pca_ex3.png)](https://colab.research.google.com/github/MerkulovDaniil/optim/blob/master/assets/Notebooks/Dimensionality_reduction.ipynb#scrollTo=PzL3k9iZJLc2)

:::
::: {.column width="60%"}
What could be wrong with this PCA?
:::
::::

## Iris dataset variance

![[source](https://sebastianraschka.com/Articles/2015_pca_in_3_steps.html)](iris.png)

## Iris dataset variance

![Iris dataset explained variance](iris_vars.pdf){width=70%}

## Wine dataset variance

![Wine dataset explained variance](wine_vars.pdf){width=70%}

## PCA on MNIST

![PCA on MNIST](mnist_pca.png)

# Other methods

## t-SNE

:::: {.columns}

::: {.column width="50%"}

**t-Distributed Stochastic Neighbor Embedding (t-SNE)** is a nonlinear dimensionality reduction technique particularly well-suited for visualizing high-dimensional data in 2 or 3 dimensions.

- **Key Concepts:**
  - **Pairwise Similarities:** Computes probabilities that pairs of high-dimensional objects are related.
  - **High to Low Dimensional Mapping:** Seeks a low-dimensional embedding where the probability distributions of pairwise similarities are preserved.
  - **Cost Function:** Minimizes the Kullback-Leibler divergence between the high-dimensional and low-dimensional probability distributions.
  - **Student's t-Distribution:** Uses a heavy-tailed distribution in the low-dimensional space to effectively model distant points and mitigate the "crowding problem."

:::

::: {.column width="50%"}

- **Algorithm Steps:**
  1. **Compute High-Dimensional Probabilities:** Use Gaussian distributions to model pairwise similarities.
  2. **Initialize Low-Dimensional Embedding:** Start with random positions.
  3. **Optimize Embedding:** Iteratively update positions to minimize divergence between distributions.
- **Considerations:**
  - **Perplexity Parameter ($\text{Perplexity}$):** Balances attention between local and global aspects of the data.
  - **Computational Complexity:** Can be slow for large datasets due to pairwise computations.
  - **Random Initialization:** Different runs may yield different results; multiple runs can help validate findings.

:::

::::

## t-SNE on MNIST

![t-SNE on MNIST](mnist_tsne.png)

## UMAP

:::: {.columns}

::: {.column width="50%"}

**Uniform Manifold Approximation and Projection (UMAP)** is a nonlinear dimensionality reduction technique that preserves both local and global data structure.

- **Key Concepts:**
  - **Manifold Learning:** Assumes data lies on a manifold in high-dimensional space.
  - **Topological Data Analysis:** Utilizes concepts from topology to model the manifold structure.
  - **Graph Construction:** Builds a weighted graph representing data relationships in high-dimensional space.
  - **Optimization:** Seeks a low-dimensional embedding that has a similar topological structure to the high-dimensional graph.

:::

::: {.column width="50%"}

- **Algorithm Steps:**
  1. **Construct High-Dimensional Graph:** Use k-nearest neighbors to build the graph.
  2. **Compute Fuzzy Simplicial Sets:** Model the probability distribution of data relationships.
  3. **Optimize Low-Dimensional Embedding:** Apply stochastic gradient descent to minimize cross-entropy between high and low-dimensional graphs.
- **Advantages:**
  - **Speed:** Faster than t-SNE, suitable for large datasets.
  - **Preservation of Structure:** Maintains more global structure compared to t-SNE.
  - **Scalability:** Can handle millions of data points efficiently.
- **Parameters:**
  - **Number of Neighbors ($n_{\text{neighbors}}$):** Controls local versus global structure preservation.
  - **Minimum Distance ($\text{min\_dist}$):** Dictates how tightly points are packed in the low-dimensional space.

:::

::::

## UMAP on MNIST

![UMAP on MNIST](mnist_umap.png)
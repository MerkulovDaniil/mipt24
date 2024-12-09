---
title: "Перспективные направления исследований для совместной лаборатории ЦУ x AIRI"
author: Даниил Меркулов
format: 
    beamer:
        pdf-engine: xelatex
        aspectratio: 169
        fontsize: 9pt
        section-titles: true
        incremental: true
        include-in-header: ../files/header_adaio.tex  # Custom LaTeX commands and preamble
header-includes:
  - \newcommand{\bgimage}{../files/backoptml.jpeg}
---

# Optimizers

## Adam работает хуже для CV, чем для LLM? ^[[Linear attention is (maybe) all you need (to understand transformer optimization)](https://arxiv.org/abs/2310.01082)]

:::: {.columns}
::: {.column width="40%"}
![CNNs on MNIST and CIFAR10](cnns.pdf)
:::

::: {.column width="60%"}
![Transformers on PTB, WikiText2, and SQuAD](transformers.pdf)
:::
::::

## Почему Adam работает хуже для CV, чем для LLM? ^[[Linear attention is (maybe) all you need (to understand transformer optimization)](https://arxiv.org/abs/2310.01082)]

### Потому что шум градиентов в языковых моделях имеет тяжелые хвосты?

![](histogram_full.pdf)Х


## Почему Adam работает хуже для CV, чем для LLM? ^[[Heavy-Tailed Class Imbalance and Why Adam Outperforms Gradient Descent on Language Models](https://arxiv.org/abs/2402.19449)]

### Нет! Метки имеют тяжелые хвосты!

:::: {.columns}
::: {.column width="50%"}
В компьютерном зрении датасеты часто сбалансированы: 1000 котиков, 1000 песелей и т.д.

В языковых датасетах почти всегда не так: слово *the* встречается часто, слово *tie* на порядки реже 
:::

::: {.column width="50%"}
![Распределение частоты токенов в PTB](PTB_classes.pdf){width=100%}
:::
::::

## Почему Adam работает хуже для CV, чем для LLM? ^[[Heavy-Tailed Class Imbalance and Why Adam Outperforms Gradient Descent on Language Models](https://arxiv.org/abs/2402.19449)]

### SGD медленно прогрессирует на редких классах

![](sgd_adam_imb.pdf){width=100% align="center"}
![](sgd_adam_imb_leg.pdf){width=100% align="center"}

SGD не добивается прогресса на низкочастотных классах, в то время как Adam добивается. Обучение GPT-2 S на WikiText-103. (a) Распределение классов, отсортированных по частоте встречаемости, разбитых на группы, соответствующие $\approx 10$ % данных. (b) Значение функции потерь при обучении. (c, d) Значение функции потерь при обучении для каждой группы при использовании SGD и Adam. 

## Automatic Gradient Descent ^[[Automatic Gradient Descent: Deep Learning without Hyperparameters](https://arxiv.org/abs/2304.05187)]

![](agd.png){width=90%}

## Automatic Gradient Descent ^[[Automatic Gradient Descent: Deep Learning without Hyperparameters](https://arxiv.org/abs/2304.05187)]

![](agd_res.pdf)

## Prodigy ^[[Prodigy: An Expeditiously Adaptive Parameter-Free Learner](https://arxiv.org/abs/2306.06101)]

![](prodigy.png)

## Muon ^[[Old Optimizer, New Norm: An Anthology](https://arxiv.org/abs/2409.20325)] ^[[Twit with the formula](https://x.com/kellerjordan0/status/1842300918542520360/photo/1)]

![](muon_alg.png)

№

# Scaling Laws

## Cooldown^[[Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations](https://arxiv.org/pdf/2405.18392)] ^[[Scaling Vision Transformers](https://arxiv.org/abs/2106.04560v2)]

:::: {.columns}

::: {.column width="50%"}

\vspace{12pt}

![](lr_schedule.pdf)

:::

::: {.column width="50%"}

![](scaling_360M_val_perplexity.pdf)

:::
::::



## NanoGPT speedrun

![[\faLink\ Источник](https://github.com/KellerJordan/modded-nanogpt)](nanogpt_speedrun.pdf){width=96%}

## Работают ли трюки, если увеличить размер модели?

![[\faLink\ Источник](https://github.com/KellerJordan/modded-nanogpt/blob/master/img/nanogpt_speedrun51.png)](nanogpt_speedrun_scale.png){width=75%}

## Работают ли трюки, если увеличить размер модели?

![[\faLink\ Источник](https://github.com/KellerJordan/modded-nanogpt/blob/master/img/nanogpt_speedrun52.png)](nanogpt_speedrun_tokens.png){width=65%}



# Обучение как задача оптимизации

## Градиентный спуск для линейной регрессии

[![](gd_2d.pdf)](https://fmin.xyz/docs/visualizations/gd_lls.mp4)


## Методы оптимизации

$$
\min_{x \in \mathbb{R}^n} f(x)
$$

:::: {.columns}
::: {.column width="33%"}

### Методы нулевого порядка

::: {.nonincremental}
* Метод Нелдера - Мида
* Эволюционные методы
* Генетические алгоритмы
* Безградиентные методы
* и другие...
:::

:::

. . .

::: {.column width="33%"}

### Методы первого порядка

$$
x_{k+1} = x_k - \alpha_k \nabla f(x_k) \hphantom{\left[\nabla^2 f(x_k)\right]^{-1}}
$$

:::

. . .

::: {.column width="33%"}

### Методы второго порядка

$$
x_{k+1} = x_k - \alpha_k \left[\nabla^2 f(x_k)\right]^{-1}\nabla f(x_k)
$$

:::

::::

## Проклятие размерности методов нулевого порядка

$$
\min_{x \in \mathbb{R}^n} f(x)
$$

. . .

$$
\text{GD: } x_{k+1} = x_k - \alpha_k \nabla f(x_k) \qquad \qquad \text{Zero order GD: } x_{k+1} = x_k - \alpha_k G,
$$

где $G$ - двухточеченая или многоточечная оценка градиента по значениям функции

. . .

|  | $f(x)$ - гладкая | $f(x)$ - гладкая и выпуклая | $f(x)$ - гладкая и сильно выпуклая |
|:-:|:---:|:----:|:-------:|
| GD | $\|\nabla f(x_k)\|^2 \approx \mathcal{O} \left( \dfrac{1}{k} \right)$ | $f(x_k) - f^* \approx  \mathcal{O} \left( \dfrac{1}{k} \right)$ | $\|x_k - x^*\|^2 \approx \mathcal{O} \left( \left(1 - \dfrac{\mu}{L}\right)^k \right)$ |
| Zero order GD | $\|\nabla f(x_k)\|^2 \approx \mathcal{O} \left( \dfrac{n}{k} \right)$ | $f(x_k) - f^* \approx  \mathcal{O} \left( \dfrac{n}{k} \right)$ | $\|x_k - x^*\|^2 \approx \mathcal{O} \left( \left(1 - \dfrac{\mu}{n L}\right)^k \right)$ |




## Пример: задача многомерного шкалирования

Пусть у нас есть матрица попарных расстояний для $N$ $d$-мерных объектов $D \in \mathbb{R}^{N \times N}$. По этой матрице мы должны восстановить начальные координаты $W_i \in \mathbb{R}^d, \; i = 1, \ldots, N$.

. . .

$$
L(W) = \sum_{i, j = 1}^N \left(\|W_i - W_j\|^2_2 - D_{i,j}\right)^2 \to \min_{W \in \mathbb{R}^{N \times d}}
$$

## Пример: задача многомерного шкалирования

[![](mds.png){width=50% fig-align="center"}](https://fmin.xyz/docs/visualizations/gd_lls.mp4)

## В машинном обучении задача оптимизации часто имеет особенный вид

$$
\min_{x \in \mathbb{R}^n} f(x) = \min_{x \in \mathbb{R}^n}\frac{1}{N} \sum_{i=1}^N f_i(x)
$$

* $f_i(x)$ - значение функции потерь модели при весах $x$ на $i$-ом объекте обучающей выборки
* $n$ - число обучаемых параметров модели ($175 \cdot 10^9$ для GPT-3.5, $405 \cdot 10^9$ для Llama 3.2)
* $N$ - размер обучающей выборки (для ImageNet $\approx 1.4 \cdot 10^7$, для WikiText $\approx 10^8$, для FineWeb-Edu $\approx 1.3 \cdot 10^{12}$).

. . .

$$
\nabla f(x_k) = \frac{1}{N} \sum_{i=1}^N \nabla f_i(x)
$$

. . .

$$
\tag{GD}
x_{k+1} = x_k - \frac{\alpha_k}{N} \sum_{i=1}^N \nabla f_i(x)
$$

Тяжело считать при больших $N$!

## В машинном обучении задача оптимизации часто имеет особенный вид{.noframebumbering}

$$
\min_{x \in \mathbb{R}^n} f(x) = \min_{x \in \mathbb{R}^n}\frac{1}{N} \sum_{i=1}^N f_i(x)
$$

::: {.nonincremental}
* $f_i(x)$ - значение функции потерь модели при весах $x$ на $i$-ом объекте обучающей выборки
* $n$ - число обучаемых параметров модели ($175 \cdot 10^9$ для GPT-3.5, $405 \cdot 10^9$ для Llama 3.2)
* $N$ - размер обучающей выборки (для ImageNet $\approx 1.4 \cdot 10^7$, для WikiText $\approx 10^8$, для FineWeb-Edu $\approx 1.3 \cdot 10^{12}$).
:::

$$
\nabla f(x_k) \approx \frac{1}{b} \sum_{i=1}^b \nabla f_{j_i}(x)
$$

$$
\tag{SGD}
x_{k+1} = x_k - \frac{\alpha_k}{b} \sum_{i=1}^b \nabla f_{j_i}(x)
$$

Можно считать при больших $N$!

## Идея SGD и батчей

![](batches_1.pdf)

## Идея SGD и батчей {.noframenumbering}

![](batches_2.pdf)

## Идея SGD и батчей {.noframenumbering}

![](batches_3.pdf)

## Идея SGD и батчей {.noframenumbering}

![](batches_4.pdf)

## Идея SGD и батчей {.noframenumbering}

![](batches_5.pdf)

## 

[![](gd_scalar_convergence.pdf)](https://fmin.xyz/docs/visualizations/sgd_3.mp4)

## 

[![](gd_scalar_convergence_to_local_minimum.pdf)](https://fmin.xyz/docs/visualizations/sgd_4.mp4)

## 

[![](sgd_escape.pdf)](https://fmin.xyz/docs/visualizations/sgd_5.mp4)

## SGD не сходится с постоянным шагом для выпуклой функции

[![](sgd_2d.pdf){width=95%}](https://fmin.xyz/docs/visualizations/sgd_divergence.mp4)

## Основные результаты сходимости SGD

:::{.callout-note appearance="simple"}
Пусть $f$ - $L$-гладкая $\mu$-сильно выпуклая функция, а дисперсия стохастического градиента конечна ($\mathbb{E}[\|\nabla f_i(x_k)\|^2] \leq \sigma^2$). Тогда траектория стохастического градиентного спуска с постоянным шагом $\alpha < \frac{1}{2\mu}$ будет гарантировать:

$$
\mathbb{E}[f(x_{k+1}) - f^*] \leq (1 - 2\alpha \mu)^k[f(x_{0}) - f^*]  + \frac{L \sigma^2 \alpha }{ 4 \mu}.
$$
:::

. . .

:::{.callout-note appearance="simple"}
Пусть $f$ - $L$-гладкая $\mu$-сильно выпуклая функция, а дисперсия стохастического градиента конечна ($\mathbb{E}[\|\nabla f_i(x_k)\|^2] \leq \sigma^2$). Тогда стохастический градиентный шум с уменьшающимся шагом $\alpha_k = \frac{2k + 1 }{ 2\mu(k+1)^2}$ будет сходиться сублинейно:

$$
\mathbb{E}[f(x_{k+1}) - f^*] \leq \frac{L \sigma^2}{ 2 \mu^2 (k+1)}
$$
:::
## Сходимость в зависимости от размера батча

$$
f(x) = \frac{\mu}{2} \|x\|_2^2 + \frac1m \sum_{i=1}^m \log (1 + \exp(- y_i \langle a_i, x \rangle)) \to \min_{x \in \mathbb{R}^n}
$$

![](sgd_problems.pdf)

# Эта задача оптимизации даже сложнее, чем кажется

## Улучшаем SGD - адаптивные методы (Adam) ^[[Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)] ^[[On the Convergence of Adam and Beyond](https://arxiv.org/abs/1904.09237)]

:::: {.columns}
::: {.column width="50%"}
* Одна из самых цитируемых научных работ в мире
* В 2018-2019 годах вышли статьи, указывающие на ошибку в оригинальной статье
* Не сходится для некоторых простых задач (даже выпуклых)
* Почему-то очень хорошо работает для некоторых сложных задач
* Гораздо лучше работает для языковых моделей, чем для задач компьютерного зрения - почему?
:::
::: {.column width="50%"}
$$
m_j^{(k)} = \beta_1 m_j^{(k-1)} + (1-\beta_1) g_j^{(k)}
$$

$$
v_j^{(k)} = \beta_2 v_j^{(k-1)} + (1-\beta_2) (g_j^{(k)})^2
$$

$$
\hat{m}_j = \frac{m_j^{(k)}}{1-\beta_1^k}, \quad \hat{v}_j = \frac{v_j^{(k)} }{1-\beta_2^k}
$$

$$
x_j^{(k)} = x_j^{(k-1)} - \alpha \frac{\hat{m}_j}{\sqrt{\hat{v}_j} + \epsilon}
$$
:::
::::



## NAG-GS ^[[NAG-GS: Semi-Implicit, Accelerated and Robust Stochastic Optimizer](https://arxiv.org/abs/2209.14937)]

:::: {.columns}
::: {.column width="50%"}
* Требует хранения одного дополнительного вектора, вместо двух, как в Adam.

* Качество в ряде задач сопоставимо с AdamW
:::

::: {.column width="50%"}
[![](nag_gs.pdf){width=100%}](https://fmin.xyz/docs/visualizations/nag_gs.mp4)
:::
::::




## Визуализация с помощью проекции на прямую

* Обозначим начальную точку как $w_0$, представляющую собой веса нейронной сети при инициализации. Веса, полученные после обучения, обозначим как $\hat{w}$.

* Генерируем случайный вектор такой же размерности и нормы $w_1 \in \mathbb{R}^p$, затем вычисляем значение функции потерь вдоль этого вектора:

$$
L (\alpha) = L (w_0 + \alpha w_1), \text{ where } \alpha \in [-b, b].
$$

## Проекция функции потерь нейронной сети на прямую

![[\faPython Open in colab](https://colab.research.google.com/github/MerkulovDaniil/optim/blob/master/assets/Notebooks/NN_Surface_Visualization.ipynb)](Line_projection_No Dropout.pdf)

## Проекция функции потерь нейронной сети на прямую {.noframenumbering}

![[\faPython Open in colab](https://colab.research.google.com/github/MerkulovDaniil/optim/blob/master/assets/Notebooks/NN_Surface_Visualization.ipynb)](Line_projection_Dropout 0.pdf)

## Проекция функции потерь нейронной сети на плоскость

* Мы можем расширить эту идею и построить проекцию поверхности потерь на плоскость, которая задается 2 случайными векторами. 

* Два случайных гауссовых вектора в пространстве большой размерности с высокой вероятностью ортогональны. 

$$
L (\alpha, \beta) = L (w_0 + \alpha w_1 + \beta w_2), \text{ where } \alpha, \beta \in [-b, b]^2.
$$

![[\faPython Open in colab](https://colab.research.google.com/github/MerkulovDaniil/optim/blob/master/assets/Notebooks/NN_Surface_Visualization.ipynb)](plane_projection.jpeg){width=70%}

## Может ли быть полезно изучение таких проекций? ^[[Visualizing the Loss Landscape of Neural Nets, Hao Li, Zheng Xu, Gavin Taylor, Christoph Studer, Tom Goldstein](https://arxiv.org/abs/1712.09913)]

:::: {.columns}
::: {.column width="35%"}
![The loss surface of ResNet-56 without skip connections](noshortLog.png)
:::

::: {.column width="65%"}
![The loss surface of ResNet-56 with skip connections](shortHighResLog.png)
:::

::::

## Может ли быть полезно изучение таких проекций, если серьезно? ^[[Loss Landscape Sightseeing with Multi-Point Optimization, Ivan Skorokhodov, Mikhail Burtsev](https://arxiv.org/abs/1910.03867)]

![Examples of a loss landscape of a typical CNN model on FashionMNIST and CIFAR10 datasets found with MPO. Loss values are color-coded according to a logarithmic scale](icons-grid.png)


## Ширина локальных минимумов

![](sam_a.pdf)

## Ширина локальных минимумов{.noframenumbering}

![](sam_b.pdf)

## Ширина локальных минимумов{.noframenumbering}

![](sam_c.pdf)

## 

[![](gd_local_convergence.pdf)](https://fmin.xyz/docs/visualizations/sgd_1.mp4)

## 

[![](sgd_local_divergence.pdf)](https://fmin.xyz/docs/visualizations/sgd_2.mp4)

## Модели не сходятся к стационарным точкам, но это не страшно ^[[NN Weights Do Not Converge to Stationary Points](https://arxiv.org/pdf/2110.06256)]

:::: {.columns}
::: {.column width="33%"}
![](imagenet-baseline-val-accu.pdf)
:::

::: {.column width="33%"}
![](imagenet-baseline-train-loss-nonsmooth.pdf)
:::

::: {.column width="33%"}
![](imagenet-baseline-train-gradnorm.pdf)
:::
::::

## Grokking ^[[Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets,   Alethea Power, Yuri Burda, Harri Edwards, Igor Babuschkin, Vedant Misra](https://arxiv.org/abs/2201.02177)]

![Training transformer with 2 layers, width 128, and 4 attention heads, with a total of about $4 \cdot 10^5$ non-embedding parameters. Reproduction of experiments (\~ half an hour) is available [here](https://colab.research.google.com/drive/1r3Wg84XECq57fT2B1dvHLSJrJ2sjIDCJ?usp=sharing)](grokking.png){width=55%}

## Double Descent ^[[Reconciling modern machine learning practice and the bias-variance trade-off, Mikhail Belkin, Daniel Hsu, Siyuan Ma, Soumik Mandal](https://arxiv.org/abs/1812.11118)]

[![](dd.pdf)](https://fmin.xyz/docs/visualizations/double_descent.mp4)




# Обучение больших моделей

## Потребление памяти при обучении GPT-2 ^[[ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)]

![](gpt2_memory_hor.pdf)

* Размер модели 1.5 B. Веса модели в fp16 занимают всего 3 GB, однако, для наивного обучения не хватит GPU даже на 32 GB 
* Для использования Adam в режиме mixed precision необходимо хранить 3 (!) копии модели в fp32.
* Активации в наивном режиме могут занимать гораздо больше памяти: для длины последовательности 1K и размера батча 32 нужно 60 GB для хранения всех промежуточных активаций. Чекпоинтинг активаций позволяет сократить потребление до 8 GB за счёт их перевычисления (33% computational overhead)

## Large batch training ^[[Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677)]

![](time.pdf){width=90%}

## Large batch training ^[[An Empirical Model of Large-Batch Training](https://arxiv.org/abs/1812.06162)]

![](basic-scaling.png)

## Large batch training ^[[Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677)]

![](batchsize.pdf){width=85%}


## Linear and square root scaling rules

When training with large batches, the learning rate must be adjusted to maintain convergence speed and stability. The **linear scaling rule**^[[Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677)] suggests multiplying the learning rate by the same factor as the increase in batch size:
$$
\alpha_{\text{new}} = \alpha_{\text{base}} \cdot \frac{\text{Batch Size}_{\text{new}}}{\text{Batch Size}_{\text{base}}}
$$
The **square root scaling rule**^[[Learning Rates as a Function of Batch Size: A Random Matrix Theory Approach to Neural Network Training](https://arxiv.org/abs/2006.09092)] proposes scaling the learning rate with the square root of the batch size increase:
$$
\alpha_{\text{new}} = \alpha_{\text{base}} \cdot \sqrt{\frac{\text{Batch Size}_{\text{new}}}{\text{Batch Size}_{\text{base}}}}
$$
Authors claimed, that it suits for adaptive optimizers like Adam, RMSProp and etc. while linear scaling rule serves well for SGD.

## Gradual warmup ^[[Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677)]

Gradual warmup helps to avoid instability when starting with large learning rates by slowly increasing the learning rate from a small value to the target value over a few epochs. This is defined as:
$$
\alpha_t = \alpha_{\text{max}} \cdot \frac{t}{T_w}
$$
where $t$ is the current iteration and $T_w$ is the warmup duration in iterations. In the original paper, authors used first 5 epochs for gradual warmup.

:::: {.columns}
::: {.column width="36%"}
![no warmup](distr-warmup-none.pdf)
:::

::: {.column width="32%"}
![constant warmup](distr-warmup-constant.pdf)
:::

::: {.column width="32%"}
![gradual warmup](distr-warmup-gradual.pdf)
:::

::::



## Спаcибо за внимание!

![Мои контакты](fmin_qr.png){width=45%}

# Запас

## Влияние инициализации весов нейронной сети на сходимость методов ^[[Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification, Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun](https://arxiv.org/abs/1502.01852)]

:::: {.columns}
::: {.column width="50%"}
![22-layer ReLU net: good init converges faster](converge_22layers.pdf)
:::

::: {.column width="50%"}
![30-layer ReLU net: good init is able to converge](converge_30layers.pdf)
:::

::::

## Методы уменьшения дисперсии? Не всё так просто на практике ^[[On the Ineffectiveness of Variance Reduced Optimization for Deep Learning](https://arxiv.org/abs/1812.04529)]
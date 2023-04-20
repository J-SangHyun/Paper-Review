[<- Back](../README.md)

[🏠Home](../../../README.md) > [📖Tech Blog](../../README.md) > [📷Computer Vision](../README.md) > \[Paper Review\] Restormer: Efficient Transformer for High-Resolution Image Restoration

### \[Paper Review\]
# Restormer: Efficient Transformer for High-Resolution Image Restoration
2023-04-12

-----

[Paper](https://arxiv.org/pdf/2111.09881.pdf) | [Code](https://github.com/swz30/Restormer)

-----

## 목차
1. [Diffusion Model](#1-diffusion-model)
2. [Range-Null Space Decomposition](#2-range-null-space-decomposition)
3. [Denoising Diffusion Null-Space Model](#3-denoising-diffusion-null-space-model)
4. [DDNM+](#4-ddnm)

-----

Convolution은 local connectivity와 translation equivariance
- limited receptive field -> preventing it from modeling long-range pixel dependencies
- static weights at inference -> cannot flexibly adapt to the input content
이걸 해결하기 위해 self-attention(SA) 메커니즘 사용

## 1. Diffusion Model
### 1.1. Forward Process
DDPM은 $T$-step의 forward process와 $T$-step의 reverse process로 정의된다. Forward process는 천천히 random noise를 데이터에 더하는 과정이며, reverse process는 noise로부터 data sample을 복원하는 과정이다.

Scale factor인 $\beta_t$와 이전 상태인 $\boldsymbol{x}_{t-1}$에 대하여, 현재 상태인 $\boldsymbol{x}_t$로의 forward process는 다음과 같다.
$$
q(\boldsymbol{x}_t|\boldsymbol{x}_{t-1})
=\mathcal{N}(\boldsymbol{x}_t;\sqrt{1-\beta_t}\boldsymbol{x}_{t-1}, \beta_t\boldsymbol{I})
$$
이를 현재 상태인 $\boldsymbol{x_t}$의 관점에서 표현하면 다음과 같다.
$$
\boldsymbol{x_t}
=\sqrt{1-\beta_t}\boldsymbol{x}_{t-1}+\sqrt{\beta_t}\boldsymbol{\epsilon},\quad\boldsymbol{\epsilon}\sim\mathcal{N}(0,\boldsymbol{I})
$$
여기서 reparametrization trick을 사용하면,
$$
q(\boldsymbol{x}_t|\boldsymbol{x}_0)
=\mathcal{N}(\boldsymbol{x}_t;\sqrt{\bar{\alpha}_t}\boldsymbol{x}_0,(1-\bar{\alpha}_t)\boldsymbol{I})\\
{\rm with}\quad\alpha_t=1-\beta_t,\quad\bar{\alpha}_t=\prod_{i=1}^{t}\alpha_i
$$
가 됨을 보일 수 있다.

### 1.2. Reverse Process
Reverse process는 현재 상태 $\boldsymbol{x}_t$로부터 이전 상태인 $\boldsymbol{x}_{t-1}$을 얻기 위하여 posterior distribution $p(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t, \boldsymbol{x}_0)$을 구하는 과정이다. Forward process의 수식과 Bayes 정리를 이용하면,
$$
p(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t,\boldsymbol{x}_0)
=q(\boldsymbol{x}_t|\boldsymbol{x}_{t-1})\frac{q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_0)}{q(\boldsymbol{x}_t|\boldsymbol{x}_0)}
=\mathcal{N}(\boldsymbol{x}_{t-1};\boldsymbol{\mu}_t(\boldsymbol{x}_t,\boldsymbol{x}_0), \sigma_t^2\boldsymbol{I})\\
{\rm with}\quad\boldsymbol{\mu}_t(\boldsymbol{x}_t,\boldsymbol{x}_0)=\frac{1}{\sqrt{\alpha_t}}\left(\boldsymbol{x}_t-\boldsymbol{\epsilon}\frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\right),\quad
\sigma_t^2=\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t
$$
를 얻을 수 있다.

DDPM은 neural network $\mathcal{Z}_{\boldsymbol{\theta}}$를 이용하여 time-step $t$의 noise $\boldsymbol{\epsilon}$을 예측하며, 예측한 noise는 $\boldsymbol{\epsilon}_t=\mathcal{Z}_{\boldsymbol{\theta}}(\boldsymbol{x}_t,t)$로 표기한다. DDPM에서는 데이터셋에서 clean한 이미지 $\boldsymbol{x}_0$과 noise $\boldsymbol{\epsilon}\sim\mathcal{N}(0,\boldsymbol{I})$, time-step $t$를 랜덤하게 뽑아 neural network $\mathcal{Z}_{\boldsymbol{\theta}}$의 parameter인 $\boldsymbol{\theta}$를 다음과 같은 gradient step으로 학습시킨다.
$$
\nabla_{\boldsymbol{\theta}}||\boldsymbol{\epsilon}-\mathcal{Z}_{\boldsymbol{\theta}}(\sqrt{\bar{\alpha}_t}\boldsymbol{x}_0+\boldsymbol{\epsilon}\sqrt{1-\bar{\alpha}_t},t)||_2^2
$$
DDPM에서는 학습된 neural network $\mathcal{Z}_{\boldsymbol{\theta}}$를 이용하여 random noises $\boldsymbol{x}_T\sim\mathcal{N}(\boldsymbol{0},\boldsymbol{I})$로부터, $p(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t,\boldsymbol{x}_0)$를 이용하여 $\boldsymbol{x}_{t-1}$을 반복적으로 sampling하게 되면 clean images $\boldsymbol{x}_0\sim q(\boldsymbol{x})$를 얻을 수 있다.

-----

## 2. Range-Null Space Decomposition
> **Pseudo-inverse**란, 주어진 linear operator $\boldsymbol{A}\in\mathbb{R}^{d\times D}$에 대하여, $\boldsymbol{A}\boldsymbol{A}^\dagger \boldsymbol{A}\equiv\boldsymbol{A}$를 만족하는 $\boldsymbol{A}^\dagger\in\mathbb{R}^{D\times d}$를 말한다.

$\boldsymbol{A}\boldsymbol{A}^\dagger \boldsymbol{A}\boldsymbol{x}\equiv \boldsymbol{A}\boldsymbol{x}$이므로 $\boldsymbol{A}^\dagger \boldsymbol{A}$는 sample $\boldsymbol{x}\in\mathbb{R}^{D\times 1}$를 $\boldsymbol{A}$의 range-space로 projection하는 operator로 볼 수 있다. 반대로 $(\boldsymbol{I}-\boldsymbol{A}^\dagger\boldsymbol{A})$는 sample $\boldsymbol{x}$를 $\boldsymbol{A}$의 null-space로 projection하는 operator가 된다.

따라서, 임의의 sample $\boldsymbol{x}$는 다음과 같이 $\boldsymbol{A}$의 range-space와 null-space의 두 가지 성분으로 decomposition될 수 있다.
$$
\boldsymbol{x}\equiv\boldsymbol{A}^\dagger\boldsymbol{A}\boldsymbol{x}+(\boldsymbol{I}-\boldsymbol{A}^\dagger\boldsymbol{A})\boldsymbol{x}
$$

-----

## 3. Denoising Diffusion Null-Space Model
### 3.1. Image Restoration & Null-Space
먼저, ground-truth 이미지 $\boldsymbol{x}\in\mathbb{R}^{D\times 1}$와 linear degradation operator $\boldsymbol{A}\in\mathbb{R}^{d\times D}$, degraded 이미지 $\boldsymbol{y}\in\mathbb{R}^{d\times 1}$의 관계를 다음과 같이 표현할 수 있다.
$$
\boldsymbol{y}=\boldsymbol{A}\boldsymbol{x}
$$
주어진 ground-truth 이미지 $\boldsymbol{y}$에 대하여, noise-free image restoration 문제는 아래의 constraints를 갖는 이미지 $\hat{\boldsymbol{x}}\in\mathbb{R}^{D\times 1}$을 찾는 문제이다.
$$
{\rm Consistency:}\quad \boldsymbol{A}\hat{\boldsymbol{x}}\equiv\boldsymbol{y},\quad
{\rm Realness:}\quad \hat{\boldsymbol{x}}\sim q(\boldsymbol{x})
$$

Consistency constraint $\boldsymbol{A}\hat{\boldsymbol{x}}\equiv\boldsymbol{y}$를 만족하는 general solution $\hat{\boldsymbol{x}}$는 $\hat{\boldsymbol{x}}=\boldsymbol{A}^\dagger\boldsymbol{y}+(\boldsymbol{I}-\boldsymbol{A}^\dagger\boldsymbol{A})\bar{\boldsymbol{x}}$의 형태가 된다. 이 때, $\boldsymbol{\bar{x}}$가 무엇이든 consistency constraint는 만족하므로 realness constraint $\hat{\boldsymbol{x}}\sim q(\boldsymbol{x})$를 만족하게 하는 적절한 $\bar{\boldsymbol{x}}$를 찾아야한다.

### 3.2. Refine Null-Space Iteratively
Reverse diffusion process는 
$$
\boldsymbol{\mu}_t(\boldsymbol{x}_t, \boldsymbol{x}_0)=\frac{\sqrt{\bar{\alpha}_{t-1}\beta_t}}{1-\bar{\alpha}_t}\boldsymbol{x}_0+\frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}\boldsymbol{x}_t,\quad
\sigma_t^2=\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t
$$

$$
\boldsymbol{x}_{0|t}=\frac{1}{\sqrt{\bar{\alpha}_t}}(\boldsymbol{x}_t-\mathcal{Z}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t)\sqrt{1-\bar{\alpha}_t})
$$

> **Algorithm** | Sampling of DDNM
> 1. $\boldsymbol{x}_T\sim\mathcal{N}(\boldsymbol{0},\boldsymbol{I})$
> 2. **for** $t=T, \dots, 1$ **do**
> 3. $\qquad\boldsymbol{x}_{0|t} = \frac{1}{\sqrt{\bar{\alpha}}_t}\left(\boldsymbol{x}_t-\mathcal{Z}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t)\sqrt{1-\bar{\alpha}_t}\right)$
> 4. $\qquad\hat{\boldsymbol{x}}_{0|t}=\boldsymbol{A}^\dagger\boldsymbol{y}+(\boldsymbol{I}-\boldsymbol{A}^\dagger\boldsymbol{A})\boldsymbol{x}_{0|t}$
> 5. $\qquad\boldsymbol{x}_{t-1}\sim p(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t,\hat{\boldsymbol{x}}_{0|t})$
> 6. **return** $\boldsymbol{x}_0$

### 3.3. Examples of $\boldsymbol{A}$ and $\boldsymbol{A}^\dagger$
- Inpainting: $\boldsymbol{A}$는 간단하게 mask operator가 된다.
- Colorization: $\boldsymbol{A}$를 pixel-wise operator라고 생각하면 각각의 RGB channel pixel $[r\quad g\quad b]^\top$에 대하여 grayscale 값은 $\left[\frac{r}{3}+\frac{g}{3}+\frac{b}{3}\right]$이므로 $\boldsymbol{A}=\left[\frac{1}{3}\quad\frac{1}{3}\quad\frac{1}{3}\right]$이 되며, $\boldsymbol{A}^\dagger=[1\quad 1\quad 1]^\top$이 된다.
- Super-resolution: scale이 $n$이라고 할 때, 각각의 $n\times n$ 크기의 패치에 대하여 $\boldsymbol{A}\in\mathbb{R}^{1\times n^2}$은 average-pooling operator가 되어야하므로 $\boldsymbol{A}=\left[\frac{1}{n^2}\quad\dots\quad\frac{1}{n^2}\right]$와 $\boldsymbol{A}^\dagger\in\mathbb{R}^{n^2\times 1}=[1\quad\dots\quad 1]^\top$가 된다.

$\boldsymbol{A}$가 여러 sub-operation들의 조합으로 구성되어 있다면, $\boldsymbol{A}=\boldsymbol{A}_1\dots\boldsymbol{A}_n$일 때, $\boldsymbol{A}$의 pseudo-inverse $\boldsymbol{A}^\dagger=\boldsymbol{A}_n^\dagger\dots\boldsymbol{A}_1^\dagger$가 된다.

-----

## 4. DDNM+
### 4.1. Scaling Range-Space Correction
Noisy image restoration는 additive Gaussian noise $\boldsymbol{n}\in\mathbb{R}^{d\times 1}\sim\mathcal{N}(\boldsymbol{0},\sigma_{\boldsymbol{y}}^2\boldsymbol{I})$에 대하여 $\boldsymbol{y}=\boldsymbol{A}\boldsymbol{x}+\boldsymbol{n}$으로 나타낼 수 있다. 여기에 DDNM을 직접 적용하게 되면
$$
\hat{\boldsymbol{x}}_{0|t}=\boldsymbol{A}^\dagger\boldsymbol{y}+(\boldsymbol{I}-\boldsymbol{A}^\dagger\boldsymbol{A})\boldsymbol{x}_{0|t}
=\boldsymbol{x}_{0|t}-\boldsymbol{A}^\dagger(\boldsymbol{A}\boldsymbol{x}_{0|t}-\boldsymbol{A}\boldsymbol{x})+\boldsymbol{A}^\dagger\boldsymbol{n}
$$
이 되며, $\boldsymbol{A}^\dagger\boldsymbol{n}\in\mathbb{R}^{D\times 1}$은 extra noise가 된다.


### 4.2. Time-Travel Trick

> **Algorithm** | Sampling of DDNM+
> 1. $\boldsymbol{x}_T\sim\mathcal{N}(\boldsymbol{0},\boldsymbol{I})$
> 2. **for** $t=T, \dots, 1$ **do**
> 3. $\qquad L=\min\left\{T-t,l\right\}$
> 4. $\qquad\boldsymbol{x}_{t+L}\sim q(\boldsymbol{x}_{t+L}|\boldsymbol{x}_t)$
> 5. $\qquad$**for** $j=L,\dots,0$ **do**
> 6. $\qquad\qquad\boldsymbol{x}_{0|t+j} = \frac{1}{\sqrt{\bar{\alpha}}_{t+j}}\left(\boldsymbol{x}_{t+j}-\mathcal{Z}_{\boldsymbol {\theta}}(\boldsymbol{x}_{t+j}, t+j)\sqrt{1-\bar{\alpha}_{t+j}}\right)$
> 7. $\qquad\qquad\hat{\boldsymbol{x}}_{0|t+j}=\boldsymbol{x}_{0|t+j}-\boldsymbol{\Sigma}_{t+j}\boldsymbol{A}^\dagger(\boldsymbol{A}\boldsymbol{x}_{0|t+j}-\boldsymbol{y})$
> 8. $\qquad\qquad\boldsymbol{x}_{t+j-1}\sim p(\boldsymbol{x}_{t+j-1}|\boldsymbol{x}_{t+j},\hat{\boldsymbol{x}}_{0|t+j})$
> 9. **return** $\boldsymbol{x}_0$

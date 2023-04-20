### Paper Review
# Zero-Shot Image Restoration Using Denoising Diffusion Null-Space Model (2022)
2022 | [Paper](https://arxiv.org/pdf/2212.00490.pdf) | [Code](https://github.com/wyhuai/DDNM)

-----



-----

## 1. Diffusion Model
### 1.1. Forward Process
DDPM은 $T$-step의 forward process와 $T$-step의 reverse process로 정의된다. Forward process는 천천히 random noise를 데이터에 더하는 과정이며, reverse process는 noise로부터 data sample을 복원하는 과정이다.

Scale factor인 $\beta_t$와 이전 상태인 ${\bf x}_{t-1}$에 대하여, 현재 상태인 ${\bf x}_t$로의 forward process는 다음과 같다.
$$
q({\bf x}_t|{\bf x}_{t-1})
=\mathcal{N}({\bf x}_t;\sqrt{1-\beta_t}{\bf x}_{t-1}, \beta_t{\bf I})
$$
이를 현재 상태인 ${\bf x_t}$의 관점에서 표현하면 다음과 같다.
$$
{\bf x_t}
=\sqrt{1-\beta_t}{\bf x}_{t-1}+\sqrt{\beta_t}{\bf \epsilon},\quad{\bf \epsilon}\sim\mathcal{N}(0,{\bf I})
$$
여기서 reparametrization trick을 사용하면,
$$
q({\bf x}_t|{\bf x}_0)
=\mathcal{N}({\bf x}_t;\sqrt{\bar{\alpha}_t}{\bf x}_0,(1-\bar{\alpha}_t){\bf I})\\
{\rm with}\quad\alpha_t=1-\beta_t,\quad\bar{\alpha}_t=\prod_{i=1}^{t}\alpha_i
$$
가 됨을 보일 수 있다.

### 1.2. Reverse Process
Reverse process는 현재 상태 ${\bf x}_t$로부터 이전 상태인 ${\bf x}_{t-1}$을 얻기 위하여 posterior distribution $p({\bf x}_{t-1}|{\bf x}_t, {\bf x}_0)$을 구하는 과정이다. Forward process의 수식과 Bayes 정리를 이용하면,
$$
p({\bf x}_{t-1}|{\bf x}_t,{\bf x}_0)
=q({\bf x}_t|{\bf x}_{t-1})\frac{q({\bf x}_{t-1}|{\bf x}_0)}{q({\bf x}_t|{\bf x}_0)}
=\mathcal{N}({\bf x}_{t-1};{\bf \mu}_t({\bf x}_t,{\bf x}_0), \sigma_t^2{\bf I})\\
{\rm with}\quad{\bf \mu}_t({\bf x}_t,{\bf x}_0)=\frac{1}{\sqrt{\alpha_t}}\left({\bf x}_t-{\bf \epsilon}\frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\right),\quad
\sigma_t^2=\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t
$$
를 얻을 수 있다.

DDPM은 neural network $\mathcal{Z}_{{\bf \theta}}$를 이용하여 time-step $t$의 noise ${\bf \epsilon}$을 예측하며, 예측한 noise는 ${\bf \epsilon}_t=\mathcal{Z}_{{\bf \theta}}({\bf x}_t,t)$로 표기한다. DDPM에서는 데이터셋에서 clean한 이미지 ${\bf x}_0$과 noise ${\bf \epsilon}\sim\mathcal{N}(0,{\bf I})$, time-step $t$를 랜덤하게 뽑아 neural network $\mathcal{Z}_{{\bf \theta}}$의 parameter인 ${\bf \theta}$를 다음과 같은 gradient step으로 학습시킨다.
$$
\nabla_{{\bf \theta}}||{\bf \epsilon}-\mathcal{Z}_{{\bf \theta}}(\sqrt{\bar{\alpha}_t}{\bf x}_0+{\bf \epsilon}\sqrt{1-\bar{\alpha}_t},t)||_2^2
$$
DDPM에서는 학습된 neural network $\mathcal{Z}_{{\bf \theta}}$를 이용하여 random noises ${\bf x}_T\sim\mathcal{N}({\bf 0},{\bf I})$로부터, $p({\bf x}_{t-1}|{\bf x}_t,{\bf x}_0)$를 이용하여 ${\bf x}_{t-1}$을 반복적으로 sampling하게 되면 clean images ${\bf x}_0\sim q({\bf x})$를 얻을 수 있다.

-----

## 2. Range-Null Space Decomposition
> **Pseudo-inverse**란, 주어진 linear operator ${\bf A}\in\mathbb{R}^{d\times D}$에 대하여, ${\bf A}{\bf A}^\dagger {\bf A}\equiv{\bf A}$를 만족하는 ${\bf A}^\dagger\in\mathbb{R}^{D\times d}$를 말한다.

${\bf A}{\bf A}^\dagger {\bf A}{\bf x}\equiv {\bf A}{\bf x}$이므로 ${\bf A}^\dagger {\bf A}$는 sample ${\bf x}\in\mathbb{R}^{D\times 1}$를 ${\bf A}$의 range-space로 projection하는 operator로 볼 수 있다. 반대로 $({\bf I}-{\bf A}^\dagger{\bf A})$는 sample ${\bf x}$를 ${\bf A}$의 null-space로 projection하는 operator가 된다.

따라서, 임의의 sample ${\bf x}$는 다음과 같이 ${\bf A}$의 range-space와 null-space의 두 가지 성분으로 decomposition될 수 있다.
$$
{\bf x}\equiv{\bf A}^\dagger{\bf A}{\bf x}+({\bf I}-{\bf A}^\dagger{\bf A}){\bf x}
$$

-----

## 3. Denoising Diffusion Null-Space Model
### 3.1. Image Restoration & Null-Space
먼저, ground-truth 이미지 ${\bf x}\in\mathbb{R}^{D\times 1}$와 linear degradation operator ${\bf A}\in\mathbb{R}^{d\times D}$, degraded 이미지 ${\bf y}\in\mathbb{R}^{d\times 1}$의 관계를 다음과 같이 표현할 수 있다.
$$
{\bf y}={\bf A}{\bf x}
$$
주어진 ground-truth 이미지 ${\bf y}$에 대하여, noise-free image restoration 문제는 아래의 constraints를 갖는 이미지 $\hat{{\bf x}}\in\mathbb{R}^{D\times 1}$을 찾는 문제이다.
$$
{\rm Consistency:}\quad {\bf A}\hat{{\bf x}}\equiv{\bf y},\quad
{\rm Realness:}\quad \hat{{\bf x}}\sim q({\bf x})
$$

Consistency constraint ${\bf A}\hat{{\bf x}}\equiv{\bf y}$를 만족하는 general solution $\hat{{\bf x}}$는 $\hat{{\bf x}}={\bf A}^\dagger{\bf y}+({\bf I}-{\bf A}^\dagger{\bf A})\bar{{\bf x}}$의 형태가 된다. 이 때, ${\bf \bar{x}}$가 무엇이든 consistency constraint는 만족하므로 realness constraint $\hat{{\bf x}}\sim q({\bf x})$를 만족하게 하는 적절한 $\bar{{\bf x}}$를 찾아야한다.

### 3.2. Refine Null-Space Iteratively
Reverse diffusion process는 
$$
{\bf \mu}_t({\bf x}_t, {\bf x}_0)=\frac{\sqrt{\bar{\alpha}_{t-1}\beta_t}}{1-\bar{\alpha}_t}{\bf x}_0+\frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}{\bf x}_t,\quad
\sigma_t^2=\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t
$$

$$
{\bf x}_{0|t}=\frac{1}{\sqrt{\bar{\alpha}_t}}({\bf x}_t-\mathcal{Z}_{{\bf \theta}}({\bf x}_t, t)\sqrt{1-\bar{\alpha}_t})
$$

> **Algorithm** | Sampling of DDNM
> 1. ${\bf x}_T\sim\mathcal{N}({\bf 0},{\bf I})$
> 2. **for** $t=T, \dots, 1$ **do**
> 3. $\qquad{\bf x}_{0|t} = \frac{1}{\sqrt{\bar{\alpha}}_t}\left({\bf x}_t-\mathcal{Z}_{{\bf \theta}}({\bf x}_t, t)\sqrt{1-\bar{\alpha}_t}\right)$
> 4. $\qquad\hat{{\bf x}}_{0|t}={\bf A}^\dagger{\bf y}+({\bf I}-{\bf A}^\dagger{\bf A}){\bf x}_{0|t}$
> 5. $\qquad{\bf x}_{t-1}\sim p({\bf x}_{t-1}|{\bf x}_t,\hat{{\bf x}}_{0|t})$
> 6. **return** ${\bf x}_0$

### 3.3. Examples of ${\bf A}$ and ${\bf A}^\dagger$
- Inpainting: ${\bf A}$는 간단하게 mask operator가 된다.
- Colorization: ${\bf A}$를 pixel-wise operator라고 생각하면 각각의 RGB channel pixel $[r\quad g\quad b]^\top$에 대하여 grayscale 값은 $\left[\frac{r}{3}+\frac{g}{3}+\frac{b}{3}\right]$이므로 ${\bf A}=\left[\frac{1}{3}\quad\frac{1}{3}\quad\frac{1}{3}\right]$이 되며, ${\bf A}^\dagger=[1\quad 1\quad 1]^\top$이 된다.
- Super-resolution: scale이 $n$이라고 할 때, 각각의 $n\times n$ 크기의 패치에 대하여 ${\bf A}\in\mathbb{R}^{1\times n^2}$은 average-pooling operator가 되어야하므로 ${\bf A}=\left[\frac{1}{n^2}\quad\dots\quad\frac{1}{n^2}\right]$와 ${\bf A}^\dagger\in\mathbb{R}^{n^2\times 1}=[1\quad\dots\quad 1]^\top$가 된다.

${\bf A}$가 여러 sub-operation들의 조합으로 구성되어 있다면, ${\bf A}={\bf A}_1\dots{\bf A}_n$일 때, ${\bf A}$의 pseudo-inverse ${\bf A}^\dagger={\bf A}_n^\dagger\dots{\bf A}_1^\dagger$가 된다.

-----

## 4. DDNM+
### 4.1. Scaling Range-Space Correction
Noisy image restoration는 additive Gaussian noise ${\bf n}\in\mathbb{R}^{d\times 1}\sim\mathcal{N}({\bf 0},\sigma_{{\bf y}}^2{\bf I})$에 대하여 ${\bf y}={\bf A}{\bf x}+{\bf n}$으로 나타낼 수 있다. 여기에 DDNM을 직접 적용하게 되면
$$
\hat{{\bf x}}_{0|t}={\bf A}^\dagger{\bf y}+({\bf I}-{\bf A}^\dagger{\bf A}){\bf x}_{0|t}
={\bf x}_{0|t}-{\bf A}^\dagger({\bf A}{\bf x}_{0|t}-{\bf A}{\bf x})+{\bf A}^\dagger{\bf n}
$$
이 되며, ${\bf A}^\dagger{\bf n}\in\mathbb{R}^{D\times 1}$은 extra noise가 된다.


### 4.2. Time-Travel Trick

> **Algorithm** | Sampling of DDNM+
> 1. ${\bf x}_T\sim\mathcal{N}({\bf 0},{\bf I})$
> 2. **for** $t=T, \dots, 1$ **do**
> 3. $\qquad L=\min\left\{T-t,l\right\}$
> 4. $\qquad{\bf x}_{t+L}\sim q({\bf x}_{t+L}|{\bf x}_t)$
> 5. $\qquad$**for** $j=L,\dots,0$ **do**
> 6. $\qquad\qquad{\bf x}_{0|t+j} = \frac{1}{\sqrt{\bar{\alpha}}_{t+j}}\left({\bf x}_{t+j}-\mathcal{Z}_{\bf {\theta}}({\bf x}_{t+j}, t+j)\sqrt{1-\bar{\alpha}_{t+j}}\right)$
> 7. $\qquad\qquad\hat{{\bf x}}_{0|t+j}={\bf x}_{0|t+j}-{\bf \Sigma}_{t+j}{\bf A}^\dagger({\bf A}{\bf x}_{0|t+j}-{\bf y})$
> 8. $\qquad\qquad{\bf x}_{t+j-1}\sim p({\bf x}_{t+j-1}|{\bf x}_{t+j},\hat{{\bf x}}_{0|t+j})$
> 9. **return** ${\bf x}_0$

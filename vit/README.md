### Paper Review
# An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (2020)
[Paper](https://arxiv.org/pdf/2010.11929.pdf) | [Code](https://github.com/google-research/vision_transformer)

## 1. Introduction
Transformer와 같은 self-attention 기반의 모델들이 NLP 분야에서 많이 사용되고 있다. Transformer는 computational efficiency와 scalability가 좋기 때문에, 큰 모델을 학습하는 것이 가능하며, dataset과 model의 크기가 점점 커지고 있는 현재에도 여전히 좋은 성능을 보이고 있다.

하지만 computer vision 분야에서는 convolution 구조가 대부분이지만, CNN과 비슷한 구조에 self-attention을 적용하거나 convolution을 완전히 self-attention으로 대체하려는 시도들이 있었다. 하지만 이러한 모델들은 이론적으로는 효율적인 구조일 수 있어도, 특수한 attention 패턴들을 사용하기 때문에 modern hardware accelerator에서는 효율적으로 작동하기 힘들었다. 따라서 이미지 인식 분야에서는 여전히 ResNet 류의 고전 모델들이 state of the art 모델이었다.

ImageNet과 같은 중간 크기의 데이터셋에서 Transformer는 CNN보다 inductive bias가 좋지 않기 때문에 데이터가 부족하여 generalization에 한계가 있었다. 하지만 큰 크기의 데이터셋에서는 이러한 낮은 inductive bias 문제를 뛰어넘어 large scale training을 통해 기존 모델들보다 좋은 성능을 보이게 된다.

## 2. Vision Transformer (ViT)
![Figure 1](./assets/vit_figure.png)

### 2.1. Patch + Position Embedding
일반적인 Transformer는 token embedding으로 구성된 1D sequence를 input으로 받는데, 2D 이미지를 적용하기 위해서 이미지 ${\bf x}\in\mathbb{R}^{H\times W\times C}$를 2D patch들의 sequence 형태인 ${\bf x}_p\in\mathbb{R}^{N\times (P^2\cdot C)}$로 reshape한다. $(P,P)$는 image patch의 resolution이며, $N=\frac{HW}{P^2}$는 patch의 개수가 된다. Transformer는 layer에서 constant latent vector size $D$를 사용하므로 patch들을 flatten하고 $D$ dimension으로 trainable linear projection을 이용하여 mapping해야 한다. 이렇게 나온 결과가 patch embedding이다.

BERT의 ```[class]``` token처럼 embedded patch들의 sequence 앞에 learnable embedding ${\bf z}_0^0={\bf x}_{\rm class}$를 붙여준다. Transformer encoder의 output의 class token ${\bf z}_L^0$가 classification을 위한 image representation ${\bf y}$가 된다. Classification head는 pre-training시에는 하나의 hidden layer로, fine-tuning시에는 single linear layer로 구성된 MLP를 사용한다.

또한, 위치 정보를 전달하기 위해서 patch embedding에 position embedding을 더한다. 2D-aware position embedding을 사용해도 일반적인 learnable 1D position embedding과 비슷하므로 1D position embedding을 사용하였다. 이렇게 만들어진 embedding vector들의 sequence는 encoder의 input으로 사용된다.

### 2.2. Transformer Encoder
Transformer encoder는 multiheaded self-attention(MSA)과 MLP block들이 번갈아가며 쌓여있는 구조이다. 그리고 모든 block 전에는 layernorm(LN)이, 모든 block 후에는 residual connection이 적용되어 있다. 또한 MLP는 GELU non-linearity와 두 개의 layer로 구성되어 있다.

$$
{\bf z}_0=[{\bf x}_{\rm class};{\bf x}_p^1{\bf E};{\bf x}_p^2{\bf E};\dots;{\bf x}_p^N{\bf E}]+{\bf E}_{\rm pos}, \quad{\bf E}\in\mathbb{R}^{(P^2\cdot C)\times D}, {\bf E}_{\rm pos}\in\mathbb{R}^{(N+1)\times D}
$$

$$
{\bf z}_l'= {\rm MSA}({\rm LN}({\bf z}_{l-1}))+{\bf z}_{l-1}, \quad l=1\dots L
$$

$$
{\bf z}_l={\rm MLP}({\rm LN}({\bf z}_l'))+{\bf z}_l', \quad l=1\dots L
$$

$$
{\bf y}={\rm LN}({\bf z}_L^0)
$$

### 2.3. Vision Transformer & CNN based Model
- **Inductive Bias** | CNN은 locality와 two-dimensional neighborhood structure, translation equivariance가 고려된 layer이다. 반면 ViT는 self-attention layer들이 global한 정보만을 학습하므로 local & translation equivariant는 오직 MLP layer에서만 학습될 수 있다. 따라서 ViT는 CNN으로 구성된 모델보다는 image-specific inductive bias가 낮다.
- **Hybrid Architecture** | Hybrid model에서는 raw image patch들을 사용하는 것이 아니라 CNN feature map으로부터 추출된 patch들을 사용한다. 추출된 patch들은 flatten하고 Transformer dimension으로 projection하여 patch embedding이 된다.

### 2.4. Model Variants
| Model | Layers | Hidden size $D$ | MLP size | Heads | Params |
|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| ViT-Base | 12 | 768 | 3072 | 12 | 86M |
| ViT-Large | 24 | 1024 | 4096 | 16 | 307M |
| ViT-Huge | 32 | 1280 | 5120 | 16 | 632M |

# Adversarial Member vs Generated Inference (MGI) | CISPA Hackathon

Adversarial pipeline developed for the **European Championship in Trustworthy AI** organized by the **CISPA Helmholtz Center for Information Security**.

> 🏆 **Achievement:** Qualified as **2nd Place Finalists in the Barcelona Stage** to compete at the European Grand Final in Germany.

---

## Task Overview: Member vs Generated Inference (MGI)
Modern Image Generative Models (IGMs) suffer from blurred boundaries between natural training data, non-members, and model-generated samples. 

In this challenge, an unknown target detector classified samples into:
- **M**: Training Member samples.
- **N**: Non-member distribution samples.
- **G**: Model-generated samples (via RAR).

The objective was to craft adversarial images that force the detector into targeted misclassifications ($M \rightarrow N$, $M \rightarrow G$, $N \rightarrow M$, $N \rightarrow G$, $G \rightarrow M$, $G \rightarrow N$) while minimizing visual distortion.

### Evaluation Metric
$$\text{Score} = \frac{1}{n}\sum_{i=1}^{n} \text{DetectScore}_i \times (1 - \text{MSE}_i)$$

- $\text{DetectScore} = 1$ if the target misclassification is achieved, $0$ otherwise.
- $\text{MSE}$ measures normalized pixel divergence against original inputs.

---

## Solution Strategy (Best Score: `0.6445`)

Our strategy balanced the strict binary classifier requirement against pixel distortion:

1. **Target Swapping with Alpha Blending ($M \leftrightarrow N$, $G \rightarrow M/N$):**
   - For distribution-shift transfers, we implemented a nearest-neighbor MSE scan across target subsets.
   - We applied a convex blend ($\alpha = 0.95$ target, $0.05$ source) to guarantee the detector triggers $1.0$ while shaving MSE points.

2. **VQGAN Latent Reconstruction ($M/N \rightarrow G$):**
   - To fool the detector into flagging natural images as generative ($G$), samples were encoded and decoded through the pretrained VQGAN/TiTok tokenizer.
   - This introduced the latent artifacts expected by generative attribution models while maintaining high structural similarity to the source.

---

## 🛠️ Stack & Pretrained Models
- **Language:** Python 3.10+
- **Frameworks:** PyTorch, Hugging Face Hub, NumPy, PIL
- **Generative Models:** [RAR (Randomized Autoregressive Visual Generation)](https://github.com/yucornetto/RAR) & [TiTok / MaskGIT Tokenizer](https://huggingface.co/fun-research/TiTok)

---

## 🚀 Usage

1. Install requirements:
   ```bash
   pip install -r requirements.txt

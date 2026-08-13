# Rough VS Multifractal Hypothesis Testing

This repository contains the Python implementation and numerical experiments associated with a statistical hypothesis-testing study for the Hurst exponent of a log-volatility field. It is mainly based on the following papers:

**"From Rough to Multifractal volatility: the Log S-fBM model"**  
Peng Wu, Jean-François Muzy, Emmanuel Bacry (2022)  
arXiv: [https://arxiv.org/abs/2201.09516](https://arxiv.org/abs/2201.09516)

**"From rough to multifractal multidimensional volatility: A multidimensional Log S-fBM model"**  
Othmane Zarhali, Emmanuel Bacry, Jean-François Muzy (2026)  
arXiv: [https://arxiv.org/abs/2601.10517](https://arxiv.org/abs/2601.10517)

**"A GMM approach to estimate the roughness of stochastic volatility"**  
Anine E. Bolko, Kim Christensen, Mikko S. Pakkanen, Bezirgen Veliyev (2020)  
arXiv: [https://arxiv.org/abs/2010.04610](https://arxiv.org/abs/2010.04610)

The code allows the simulation of the log-volatility field and the numerical study of two statistical tests for **$H = 0$ vs. $H \neq 0$**, i.e. for distinguishing the *rough volatility* regime from its *multifractal* / log-correlated boundary case, together with a standalone Central Limit Theorem illustration used as a pedagogical warm-up.

---

## Overview

The log-volatility field $\omega(t)$ studied throughout this repository is a centred stationary Gaussian process with autocovariance

$$ c_H(x) =
\lambda^2
\log\left(\frac{T}{x}\right)
\mathbf{1}_{\{H=0\}}
\mathbf{1}_{\{0<x<T\}}
+
\frac{\lambda^2}{2H(1-2H)}
\left(T^{2H}-x^{2H}\right)
\mathbf{1}_{\{H\neq0\}}
\mathbf{1}_{\{0<x<T\}}.
$$



Thus, the same covariance family interpolates between:

- **Rough volatility regime:** $0 < H < \frac{1}{2}$
- **Multifractal regime:** $H = 0$, the logarithmic / log-correlated boundary case

The main parameters are:

- $H$: Hurst exponent controlling roughness
- $T$: correlation (memory) scale
- $\lambda^2$: volatility-of-volatility (intermittency) coefficient
- $n, \Delta$: number and length of the aggregation windows used to build the test statistics

On each window $j$, two block quantities are defined:

$$
M_j
:=
\int_{\text{window }j} e^{\omega(t)} dt,
$$

The centred log-mass statistic

$$
\Sigma_n
:=
\sum_{j=1}^{n}
\left(
\log M_j-\mathbb{E}[\log M_j]
\right)
$$

satisfies, under both the null and the alternative,

$$
\frac{\Sigma_n}{\lambda}
\xrightarrow{\ d\ }
\mathcal{N}\left(0,V_n(H)\right),
$$

for a closed-form asymptotic variance $V_n(H)$.

Two test statistic for

$$
H_0:H=0
\qquad\text{vs.}\qquad
H_1:H\neq0
$$

is studied: 

a **direct log-mass statistic**
  $$
  Z_n
  :=
  \frac{\Sigma_n}
  {\lambda\sqrt{V_n(0)}},
  $$
  which shares the same asymptotic power as $T_n$ but shows noticeably better finite-sample power and requires no threshold or bandwidth selection.

---

## Usage

### CLT illustration

```bash
cd clt_demo
python clt_illustration.py

# Online Softmax Correction Derivation

Here, assume we have a GEMV problem: 

P @ V = O

with shape: 1xN @ NxD = 1xD

given sm(S) = P

First we split P into 2 halves. Each half has its own local max and local sum. Global max is called m, and local max is called m0 and m1 for split 0 and split 1 respectively. There's no better way to understand Online softmax than deriving the formula yourself on a piece of paper, try to think on it until it clicks.

# Online Softmax Correction Derivation

Given the attention output for a single output dimension `d`:

$$
O_d = \sum_{n=0}^{N-1} p_n \cdot V_{nd}
$$

where `p_n` is the softmax probability computed from the score `S_n`:

$$
p_n =
\frac{e^{S_n - m}}
{\sum_{j=0}^{N-1} e^{S_j - m}}
$$

and `m` is the global maximum score:

$$
m = \max_{0 \leq j < N} S_j
$$

Substituting the softmax expression into the output:

$$
O_d =
\sum_{n=0}^{N-1}
\frac{e^{S_n - m}}
{\sum_{j=0}^{N-1} e^{S_j - m}}
\cdot V_{nd}
$$

Since the denominator does not depend on `n`, it can be factored out:

$$
O_d =
\frac{1}
{\sum_{j=0}^{N-1} e^{S_j - m}}
\sum_{n=0}^{N-1}
e^{S_n - m} \cdot V_{nd}
$$

---

## 1. Denominator: Row Sum with Correction Term

Split the row sum into two blocks:

$$
\sum_{j=0}^{N-1} e^{S_j - m}
=
\sum_{j=0}^{n_0} e^{S_j - m}
+
\sum_{j=n_0+1}^{N-1} e^{S_j - m}
$$

Assume each block has its own local maximum:

- `m_0`: maximum score of the first block
- `m_1`: maximum score of the second block
- `m`: global maximum score across both blocks

where:

$$
m_0 = \max_{0 \leq j \leq n_0} S_j
$$

$$
m_1 = \max_{n_0 + 1 \leq j < N} S_j
$$

$$
m = \max(m_0, m_1)
$$

Rewrite each block using its local maximum:

$$
\sum_{j=0}^{N-1} e^{S_j - m}
=
e^{m_0 - m}
\sum_{j=0}^{n_0} e^{S_j - m_0}
+
e^{m_1 - m}
\sum_{j=n_0+1}^{N-1} e^{S_j - m_1}
$$

Define the local softmax denominators:

$$
l_0 =
\sum_{j=0}^{n_0} e^{S_j - m_0}
$$

$$
l_1 =
\sum_{j=n_0+1}^{N-1} e^{S_j - m_1}
$$

Therefore, the corrected denominator is:

$$
l =
e^{m_0 - m} l_0
+
e^{m_1 - m} l_1
$$

---

## 2. Numerator with Correction Term

The numerator is:

$$
\sum_{n=0}^{N-1} e^{S_n - m} \cdot V_{nd}
$$

Split it into two blocks:

$$
\sum_{n=0}^{N-1} e^{S_n - m} \cdot V_{nd}
=
\sum_{n=0}^{n_0} e^{S_n - m} \cdot V_{nd}
+
\sum_{n=n_0+1}^{N-1} e^{S_n - m} \cdot V_{nd}
$$

Apply the local-max correction terms:

$$
\sum_{n=0}^{N-1} e^{S_n - m} \cdot V_{nd}
=
e^{m_0 - m}
\sum_{n=0}^{n_0}
e^{S_n - m_0} \cdot V_{nd}
+
e^{m_1 - m}
\sum_{n=n_0+1}^{N-1}
e^{S_n - m_1} \cdot V_{nd}
$$

Define the local unnormalized output accumulators:

$$
O_{0d}^{num} =
\sum_{n=0}^{n_0}
e^{S_n - m_0} \cdot V_{nd}
$$

$$
O_{1d}^{num} =
\sum_{n=n_0+1}^{N-1}
e^{S_n - m_1} \cdot V_{nd}
$$

Therefore, the corrected numerator is:

$$
O_d^{num} =
e^{m_0 - m} O_{0d}^{num}
+
e^{m_1 - m} O_{1d}^{num}
$$

---

## 3. Final Corrected Output

The final softmax output is the corrected numerator divided by the corrected denominator:

$$
O_d =
\frac{O_d^{num}}{l}
$$

Substituting the corrected numerator and denominator:

$$
O_d =
\frac{
e^{m_0 - m} O_{0d}^{num}
+
e^{m_1 - m} O_{1d}^{num}
}{
e^{m_0 - m} l_0
+
e^{m_1 - m} l_1
}
$$

This is the online softmax merge rule for two blocks.

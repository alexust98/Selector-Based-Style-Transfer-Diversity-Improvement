# Selector Based Style Transfer Diversity Improvement
A simple method that conditions a MultiStyle Transformer model to produce diverse stylizations.

In the beggining of each training step we propose to sample a random vector  $`\mathbf{s} \in \{\mathbf{S} \in \mathbb{R}^{\mathrm{C}} \ \vert \sum\limits_{i=1}^{\mathrm{C}} \mathbf{S}_{i} = \mathrm{C}, \mathbf{S}_{i} \ge 0\}`$. This vector is later used as a condition input to MultiStyle transformer model that rescales channels of intermediate feature maps. The desired outcome of proposed conditioning is guided by updated style loss:
```math
\begin{gather}
    \hat{\mathcal{L}}(x_{style}, x_{o}) = \sum\limits_{l=1}^{L} \dfrac{1}{\mathrm{C}^{2}_{l}}\Arrowvert \hat{G}_{l}(x_{style}) - \hat{G}_{l}(x_{o}) \Arrowvert^{2}_{F}\\
    \hat{\mathrm{G}}_{l}(x) = \frac{1}{\mathrm{W_{l}H_{l}}} \left(interp(\mathbf{s}, \mathrm{C}_{l}) \odot \mathrm{F}_{l}(x)\right)\left(interp(\mathbf{s}, \mathrm{C}_{l}) \odot\mathrm{F}_{l}(x)\right)^{\top},
\end{gather}
```
This way we are able to control stylization intensities for each encoded style feature. To produce various results we just randomize the condition vector!
Here are some of the results with different sampling parameter $`\sigma`$:

![results](images/RandomSelector2.png)

# Apply Shannon Entropy Loss on gaussians of which a ray pass through.
Based on what InfoNerf(https://browse.arxiv.org/pdf/2112.15399.pdf) did, we can apply Shannon Entropy Loss on gaussians of which a ray pass through. Due to the slight difference between InfoNerf and 3D Gaussian Splatting, we need to modify the original loss function a little bit. The original loss function is as follows:

$$
p(r_i) = \frac{\alpha_i}{\sum_{j=1}^N \alpha_j}=\frac{1-exp(-\sigma_i \delta_i)}{\sum_{j=1}^N 1-exp(-\sigma_j \delta_j)}
$$

$$
H(r) = -\sum_{i=1}^N p(r_i)log(p(r_i))
$$

$$
M(r) = \begin{cases}
1, & \text{if } \sum_{i=1}^N{1-exp(-\sigma_i \delta_i)} \geq \epsilon \\ 
0, & \text{otherwise}
\end{cases}
$$

$$
L_{entropy} = M(r)H(r)
$$

and the ray are both sampled from the training data and randomly generated. However, in 3D Gaussian Splatting, we don't have the $\sigma_i$ and $\delta_i$ of each gaussian. 

So we need to modify the loss function to make it suitable for 3D Gaussian Splatting. The Shannon Entropy can come from the probability distribution of gaussian mean or the probability distribution of color weight. For the latter one, the loss function is as follows:

$$
o_i = \alpha_i \prod_{j=1}^{i-1}(1-\alpha_j)
$$

$$
L_{entropy_{r}} = -\sum_{i=1}^N o_i log(o_i)\\
= -\sum_{i=1}^N \alpha_i \prod_{j=1}^{i-1}(1-\alpha_j) log(\alpha_i \prod_{j=1}^{i-1}(1-\alpha_j))
$$

$$
\partial L_{entropy_{r}} / \partial \alpha_i = -\frac{\partial {(\alpha_i \prod_{j=1}^{i-1}(1-\alpha_j) log(\alpha_i \prod_{j=1}^{i-1}(1-\alpha_j)))}}{\partial \alpha_i} - \frac{\sum_{j=i+1}^{N} \partial {(\alpha_j \prod_{k=1}^{j-1}(1-\alpha_k) log(\alpha_j \prod_{k=1}^{j-1}(1-\alpha_k)))}}{\partial \alpha_i}
$$

first term:

$$
\frac{\partial {(\alpha_i \prod_{j=1}^{i-1}(1-\alpha_j) log(\alpha_i \prod_{j=1}^{i-1}(1-\alpha_j)))}}{\partial \alpha_i}=\frac{\partial {(\alpha_i T_i \ log(\alpha_i T_i))}}{\partial \alpha_i}=T_i+T_i log(\alpha_i T_i)
$$

second term:

$$
\frac{\sum_{j=i+1}^{N} \partial {(\alpha_j \prod_{k=1}^{j-1}(1-\alpha_k) log(\alpha_j \prod_{k=1}^{j-1}(1-\alpha_k)))}}{\partial \alpha_i}=\frac{\sum_{j=i+1}^{N} \partial {(\alpha_j T_j \ log(\alpha_j T_j))}}{\partial \alpha_i}\\
=\frac{\sum_{j=i+1}^{N} \partial {\alpha_j \frac{T_j(1-\alpha_i)}{1-\alpha_i} \ log(\alpha_j \frac{T_j(1-\alpha_i)}{1-\alpha_i})}}{\partial \alpha_i}\\
=\sum_{j=i+1}^{N}(-c-c\ln(c-c\alpha_i))
$$
in which $c=\frac{T_j}{1-\alpha_i}$

but 
$$
\sum_{j=i+1}^{N}(-c-c\ln(c-c\alpha_i))
$$
is difficult to calculate during back propagation, so we only use the first term to calculate the gradient.

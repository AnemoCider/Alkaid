# PRB × IBL

## Starting from the Rendering Equation

$$L_o(p, \omega_o) = \int_{\Omega^+}L_i(p, \omega_i)f_r(p, \omega_i, \omega_o)\cos\theta_iV(p, \omega_i)d\omega_i$$ Since we are not going to consider shadowing, the visibility term can be omitted.

$$L_o(p, \omega_o) = \int_{\Omega^+}L_i(p, \omega_i)f_r(p, \omega_i, \omega_o)\cos\theta_id\omega_i$$
We call the term $L_i(p, \omega_i)$  `lighting`, and $f_r(p, \omega_i, \omega_o)\cos\theta_id\omega_i$ `light transport`

### Split-Sum Approximation

We split the `lighting` term outside of the integral:  
$$\int_{\Omega^+}L_i(p, \omega_i)f_r(p, \omega_i, \omega_o)\cos\theta_id\omega_i \approx \frac{\int_{\Omega_f}L_id\omega_i}{\int_{\Omega_f} d\omega_i}\int_{\Omega^+}f_r\cos\theta_id\omega_i$$

Intuitively, it approximates the `light transport` term as a constant over its support, and averages the `lighting` term. Therefore, it is exact when`light transport` is indeed a constant function, corresponding to a diffuse material.

Now, using the PBR BRDF, our target is to calculate $$\frac{\int_{\Omega_f}L_id\omega_i}{\int_{\Omega_f} d\omega_i}\int_{\Omega^+}\frac{D(\textbf{h})F(\textbf{h}, \textbf{v})G(\textbf{l}, \textbf{v}, \textbf{h})}{4(\textbf{n}\cdot\textbf{l})(\textbf{n}\cdot\textbf{v})}\cos\theta_id\omega_i$$

## Mathematical Calculation

### Lighting


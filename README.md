# NumericalSolver 

This software package provides the implementation of a collection of different numerical solvers developed in support of various research-focused toolkits within the [GIFT-Surg](http://www.gift-surg.ac.uk/) project.

Implemented solvers include
* **Primal-Dual Methods** as described in [[Chambolle and Pock, 2010]](https://link.springer.com/article/10.1007/s10851-010-0251-1)
* **Alternating Direction Method of Multipliers (ADMM)** as described in, e.g., [[Diamond and Boyd, 2015]](http://stanford.edu/~boyd/papers/admm_distr_stats.html) 

to solve
* **L1- and L2-denoising** problems, i.e.
```math
\vec{x}^*:=\text{argmin}_{\vec{x}} \Big[\Vert \vec{x} - \vec{x}_0 \Vert_{\ell^p}^p + \alpha\,\text{Reg}(\vec{x})\Big],\quad\text{for}\quad p\in\{1,\,2\},
```
and

* **robust L2-deconvolution** problems, i.e.
```math
\vec{x}^*:=\text{argmin}_{\vec{x}} \Big[\sum_{i=1}^N \varrho\big( (A\vec{x} - \vec{b})_i^2 )  + \alpha\,\text{Reg}(\vec{x})\Big],
```

in 1D, 2D or 3D for a variety of different regularizers $`\text{Reg}`$ and data loss functions $`\varrho`$. 

If you have any questions or comments (or find bugs), please drop me an email to `michael.ebner.14@ucl.ac.uk`.

## Features

The **available regularizers**, depending on the minimization problem, include
* Zeroth-order Tikhonov (TK0): $`\text{Reg}(\vec{x}) = \frac{1}{2}\Vert \vec{x} \Vert_{\ell^2}^2`$
* First-order Tikhonov (TK1): $`\text{Reg}(\vec{x}) = \frac{1}{2}\Vert \nabla \vec{x} \Vert_{\ell^2}^2`$
* Isotropic Total Variation (TV): $`\text{Reg}(\vec{x}) = \text{TV}_\text{iso}(\vec{x}) = \big\Vert |\nabla \vec{x}| \big\Vert_{\ell^1}`$
* Huber Function: $`\text{Reg}(\vec{x}) = \frac{1}{2\gamma} \big| |\nabla \vec{x}| \big|_{\gamma}`$

---

**Data loss functions** $`\varrho`$ are motivated by [SciPy](https://docs.scipy.org/doc/scipy-0.19.0/reference/generated/scipy.optimize.least_squares.html) and allow for robust outlier rejection. Implemented data loss functions are:
* `linear`: $`\varrho(e) = e `$ 
* `soft_l1`: $`\varrho(e) = 2 (\sqrt{1+e} - 1)`$ 
* `huber`: $`\varrho(e) = |e|_\gamma = \begin{cases} e, & e < \gamma^2 \\ 2\gamma\sqrt{e} - \gamma^2, & e\ge \gamma^2\end{cases}`$
* `arctan`: $`\varrho(e) = \arctan(e)`$
* `cauchy`: $`\varrho(e) = \ln(1 + e)`$

---

Additionally, the choice of finding optimal reconstruction parameters is facilitated by providing several evaluation methods including
* **L-curve studies**, and 
* the **evaluation of similarity measures** (in case a reference image is available) 

in the course of **parameter studies**. Implemented similarity measures are

* Sum of Squared Differences (SSD)
* Mean Square Error (MSE)
* Root Mean Square Error (RMSE)
* Peak-Signal-to-Noise-Ratio (PSNR)
* Mutual Information (MI)
* Normalized Mutual Information (NMI)
* Structural Similarity (SSIM)
* Normalized Cross Correlation (NCC)

## Installation

Required dependencies can be installed using `pip` by running
* `pip install -r requirements.txt`
* `pip install -e .`

In addition, you will need to install `itk` for Python. In case you want to make use of the [Volumetric MRI Reconstruction from Motion Corrupted 2D Slices](https://cmiclab.cs.ucl.ac.uk/mebner/VolumetricReconstruction) tool or any of its dependencies, please install the ITK version as described there. Otherwise, simply run
* `pip install itk`

In order to run the provided unit tests, please execute
* `python test/runTests.py`

## Usage

### Denoising

TVL1/TVL2/HuberL2/HuberL1 Denoising can be run via

* `python runDenoising.py \
--observation path-to-observation-png-nii-mat \
--result path-to-denoised-result-png-nii-mat \
--reconstruction-type TVL1 \
--alpha 0.05 \
--iterations 50
`
* `python runDenoising.py \
--observation path-to-observation-png-nii-mat \
--reference path-to-reference-png-nii-mat \
--result path-to-denoised-result-png-nii-mat \
--reconstruction-type TVL1 \
--alpha 0.05 \
--iterations 50
`

### Deconvolution
Examples for TK0L2/TK1L2/TVL2/HuberL2 deconvolution calls are

* `python runDeconvolution.py \
--observation path-to-observation-png-nii-mat \
--result path-to-denoised-result-png-nii-mat \
--reconstruction-type HuberL2 \
--blur 1.2 \
--alpha 0.05 \
--iterations 50
`
* `python runDeconvolution.py \
--observation path-to-observation-png-nii-mat \
--reference path-to-reference-png-nii-mat \
--result path-to-denoised-result-png-nii-mat \
--reconstruction-type HuberL2 \
--alpha 0.05 \
--blur 1.2 \
--iterations 50 \
--data-loss soft_l1 \
--minimizer L-BFGS-B \
`

### Parameter Studies
Parameter studies for the denoising problem (and, similarly, for deconvolution problem) can be performed by, e.g.,
* `python runDenoisingStudy.py \
--observation path-to-observation-png-nii-mat \
--dir-output path-to-parameter-study \
--reference path-to-reference-png-nii-mat \
--reconstruction-type TVL2 \
--study-name TVL2-Denoising
`

The results can be visualized by
* `python showParameterStudy.py \
--dir-input path-to-parameter-study \
--study-name TVL2-Denoising
`

## License
This framework is licensed under the [MIT license ![MIT](https://raw.githubusercontent.com/legacy-icons/license-icons/master/dist/32x32/mit.png)](http://opensource.org/licenses/MIT)
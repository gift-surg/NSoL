# Numerical Solver Library 

The Numerical Solver Library (NSoL) is a Python-based open-source toolkit for research developed within the [GIFT-Surg][giftsurg] project and contains several implementations of denoising and deconvolution algorithms.

The algorithm and software were developed by [Michael Ebner][mebner] at the [Translational Imaging Group][tig] in the [Centre for Medical Image Computing][cmic] at [University College London (UCL)][ucl].

If you have any questions or comments, please drop an email to `michael.ebner.14@ucl.ac.uk`.

## Features

Implemented solvers include
* **Primal-Dual Methods** as described in [[Chambolle and Pock, 2010]](https://link.springer.com/article/10.1007/s10851-010-0251-1)
* **Alternating Direction Method of Multipliers (ADMM)** as described in, e.g., [[Diamond and Boyd, 2015]](http://stanford.edu/~boyd/papers/admm_distr_stats.html) 

to solve
* **L1- and L2-denoising** problems, i.e.
<!--https://www.url-encode-decode.com/-->
<!--```math-->
<!--\vec{x}^*:=\text{argmin}_{\vec{x}}\Big[\Vert\vec{x}-\vec{x}_0\Vert_{\ell^p}^p+\alpha\,\text{Reg}(\vec{x})\Big],\quad\text{for}{\quad}p\in\{1,\,2\},-->
<!--```-->
<p align="center">
<img src="http://latex.codecogs.com/svg.latex?%5Cvec%7Bx%7D%5E%2A%3A%3D%5Ctext%7Bargmin%7D_%7B%5Cvec%7Bx%7D%7D%5CBig%5B%5CVert%5Cvec%7Bx%7D-%5Cvec%7Bx%7D_0%5CVert_%7B%5Cell%5Ep%7D%5Ep%2B%5Calpha%5C%2C%5Ctext%7BReg%7D%28%5Cvec%7Bx%7D%29%5CBig%5D%2C%5Cquad%5Ctext%7Bfor%7D%7B%5Cquad%7Dp%5Cin%5C%7B1%2C%5C%2C2%5C%7D%2C">
</p>
and

* **robust L2-deconvolution** problems, i.e.
<!--https://www.url-encode-decode.com/-->
<!--```math-->
<!--\vec{x}^*:=\text{argmin}_{\vec{x}}\Big[\sum_{i=1}^N\varrho\big((A\vec{x}-\vec{b})_i^2)+\alpha\,\text{Reg}(\vec{x})\Big],-->
<!--```-->
<p align="center">
<img src="http://latex.codecogs.com/svg.latex?%5Cvec%7Bx%7D%5E%2A%3A%3D%5Ctext%7Bargmin%7D_%7B%5Cvec%7Bx%7D%7D%5CBig%5B%5Csum_%7Bi%3D1%7D%5EN%5Cvarrho%5Cbig%28%28A%5Cvec%7Bx%7D-%5Cvec%7Bb%7D%29_i%5E2%29%2B%5Calpha%5C%2C%5Ctext%7BReg%7D%28%5Cvec%7Bx%7D%29%5CBig%5D%2C">
</p>

in 1D, 2D or 3D for a variety of regularizers ![img](http://latex.codecogs.com/svg.latex?%5Ctext%7BReg%7D) and data loss functions ![img](http://latex.codecogs.com/svg.latex?%5Cvarrho).

---

The **available regularizers**, depending on the minimization problem, include
<!-- * Zeroth-order Tikhonov (TK0): $`\text{Reg}(\vec{x}) = \frac{1}{2}\Vert \vec{x} \Vert_{\ell^2}^2`$ -->
<!-- * First-order Tikhonov (TK1): $`\text{Reg}(\vec{x}) = \frac{1}{2}\Vert \nabla \vec{x} \Vert_{\ell^2}^2`$ -->
<!-- * Isotropic Total Variation (TV): $`\text{Reg}(\vec{x}) = \text{TV}_\text{iso}(\vec{x}) = \big\Vert |\nabla \vec{x}| \big\Vert_{\ell^1}`$ -->
<!-- * Huber Function: $`\text{Reg}(\vec{x}) = \frac{1}{2\gamma} \big| |\nabla \vec{x}| \big|_{\gamma}`$ -->

* Zeroth-order Tikhonov (TK0): ![img](http://latex.codecogs.com/svg.latex?%5Ctext%7BReg%7D%28%5Cvec%7Bx%7D%29%3D%5Cfrac%7B1%7D%7B2%7D%5CVert%5Cvec%7Bx%7D%5CVert_%7B%5Cell%5E2%7D%5E2)
* First-order Tikhonov (TK1): ![img](http://latex.codecogs.com/svg.latex?%5Ctext%7BReg%7D%28%5Cvec%7Bx%7D%29%3D%5Cfrac%7B1%7D%7B2%7D%5CVert%5Cnabla%5Cvec%7Bx%7D%5CVert_%7B%5Cell%5E2%7D%5E2)
* Isotropic Total Variation (TV): ![img](http://latex.codecogs.com/svg.latex?%5Ctext%7BReg%7D%28%5Cvec%7Bx%7D%29%3D%5Ctext%7BTV%7D_%5Ctext%7Biso%7D%28%5Cvec%7Bx%7D%29%3D%5Cbig%5CVert%7C%5Cnabla%5Cvec%7Bx%7D%7C%5Cbig%5CVert_%7B%5Cell%5E1%7D)
* Huber Function: ![img](http://latex.codecogs.com/svg.latex?%5Ctext%7BReg%7D%28%5Cvec%7Bx%7D%29%3D%5Cfrac%7B1%7D%7B2%5Cgamma%7D%5Cbig%7C%7C%5Cnabla%5Cvec%7Bx%7D%7C%5Cbig%7C_%7B%5Cgamma%7D)

---

**Data loss functions** ![img](http://latex.codecogs.com/svg.latex?%5Cvarrho) are motivated by [SciPy](https://docs.scipy.org/doc/scipy-0.19.0/reference/generated/scipy.optimize.least_squares.html) and allow for robust outlier rejection. Implemented data loss functions are:
<!--$`\varrho(e)=e`$-->
<!--$`\varrho(e)=2(\sqrt{1+e}-1)`$ -->
<!--$`\varrho(e)=|e|_\gamma=\begin{cases}e,&e<\gamma^2\\2\gamma\sqrt{e}-\gamma^2,&e\ge\gamma^2\end{cases}`$-->
<!--$`\varrho(e)=\arctan(e)`$-->
<!--$`\varrho(e)=\ln(1 + e)`$-->
* `linear`: ![img](http://latex.codecogs.com/svg.latex?%5Cvarrho%28e%29%3De)
* `soft_l1`: ![img](http://latex.codecogs.com/svg.latex?%5Cvarrho%28e%29%3D2%28%5Csqrt%7B1%2Be%7D-1%29)
* `huber`: ![img](http://latex.codecogs.com/svg.latex?%5Cvarrho%28e%29%3D%7Ce%7C_%5Cgamma%3D%5Cbegin%7Bcases%7De%2C%26e%3C%5Cgamma%5E2%5C%5C2%5Cgamma%5Csqrt%7Be%7D-%5Cgamma%5E2%2C%26e%5Cge%5Cgamma%5E2%5Cend%7Bcases%7D)
* `arctan`: ![img](http://latex.codecogs.com/svg.latex?%5Cvarrho%28e%29%3D%5Carctan%28e%29)
* `cauchy`: ![img](http://latex.codecogs.com/svg.latex?%5Cvarrho%28e%29%3D%5Cln%281%2Be%29)

---

Additionally, the choice of finding optimal reconstruction parameters is facilitated by providing several evaluation methods including
* **L-curve studies**, and 
* the **evaluation of similarity measures** (in case a reference image is available) 

in the course of **parameter studies**. Implemented similarity measures are

* Sum of Squared Differences (SSD)
* Mean Absolute Error (MAE)
* Mean Square Error (MSE)
* Root Mean Square Error (RMSE)
* Peak-Signal-to-Noise Ratio (PSNR)
* Mutual Information (MI)
* Normalized Mutual Information (NMI)
* Structural Similarity (SSIM)
* Normalized Cross Correlation (NCC)

## Installation

NSoL was developed in

* Mac OS X 10.10 and 10.12
* Ubuntu 14.04 and 16.04

and tested for Python 2.7.12 and 3.5.2.

In case NSoL is used in conjuction with any of the toolkits of [NiftyMIC][niftymic], [Volumetric Reconstruction From Printed Films][volumetricreconstructionfromprintedfilms] or [SimpleReg][simplereg], please 
* install [ITK_NiftyMIC][itkniftymic]

If NSoL is used standalone, please run instead
* `pip install itk`

Afterwards, clone this repository via 
* `git clone git@github.com:gift-surg/NSoL.git`

where all remaining dependencies can be installed using `pip`:
* `pip install -e .`


## Usage

### Denoising

TVL1/TVL2/HuberL2/HuberL1 Denoising can be run via

```
nsol_run_denoising \
--observation path-to-observation-png-nii-mat \
--result path-to-denoised-result-png-nii-mat \
--reconstruction-type TVL1 \
--alpha 0.05 \
--iterations 50
```
```
nsol_run_denoising \
--observation path-to-observation-png-nii-mat \
--reference path-to-reference-png-nii-mat \
--result path-to-denoised-result-png-nii-mat \
--reconstruction-type TVL1 \
--alpha 0.05 \
--iterations 50
```

### Deconvolution
Examples for TK0L2/TK1L2/TVL2/HuberL2 deconvolution calls are

```
nsol_run_deconvolution \
--observation path-to-observation-png-nii-mat \
--result path-to-denoised-result-png-nii-mat \
--reconstruction-type HuberL2 \
--blur 1.2 \
--alpha 0.05 \
--iterations 50
```
```
nsol_run_deconvolution \
--observation path-to-observation-png-nii-mat \
--reference path-to-reference-png-nii-mat \
--result path-to-denoised-result-png-nii-mat \
--reconstruction-type HuberL2 \
--alpha 0.05 \
--blur 1.2 \
--iterations 50 \
--data-loss soft_l1 \
--minimizer L-BFGS-B
```

### Parameter Studies
Parameter studies for the denoising problem (and, similarly, for deconvolution problem) can be performed by, e.g.,
```
nsol_run_denoising_study \
--observation path-to-observation-png-nii-mat \
--dir-output path-to-parameter-study \
--reference path-to-reference-png-nii-mat \
--reconstruction-type TVL2 \
--study-name TVL2-Denoising \
--measures RMSE PSNR NCC NMI SSIM \
--alpha-range 0.001 0.05 20
```

The results can be visualized by
```
nsol_show_parameter_study \
--dir-input path-to-parameter-study \
--study-name TVL2-Denoising \
--dir-output-figures dir-to-figures
```

## Licensing and Copyright
Copyright (c) 2018, [University College London][ucl].
This framework is made available as free open-source software under the [BSD-3-Clause License][bsd]. Other licenses may apply for dependencies.

## Funding
This work is partially funded by the UCL [Engineering and Physical Sciences Research Council (EPSRC)][epsrc] Centre for Doctoral Training in Medical Imaging (EP/L016478/1), the Innovative Engineering for Health award ([Wellcome Trust][wellcometrust] [WT101957] and [EPSRC][epsrc] [NS/A000027/1]), and supported by researchers at the [National Institute for Health Research][nihr] [University College London Hospitals (UCLH)][uclh] Biomedical Research Centre.

## References
Associated publications are 
* [[EbnerWang2018]](http://link.springer.com/10.1007/978-3-030-00928-1_36) Ebner, M., Wang, G., Li, W., Aertsen, M., Patel, P. A., Melbourne, A., Doel, T., David, A. L., Deprest, J., Ourselin, S., & Vercauteren, T. (2018). An Automated Localization, Segmentation and Reconstruction Framework for Fetal Brain MRI. In Medical Image Computing and Computer-Assisted Intervention -- MICCAI 2018 (pp. 313–320). Springer
* [[Ebner2018]](https://www.sciencedirect.com/science/article/pii/S1053811917308042) Ebner, M., Chung, K. K., Prados, F., Cardoso, M. J., Chard, D. T., Vercauteren, T., & Ourselin, S. (2018). Volumetric reconstruction from printed films: Enabling 30 year longitudinal analysis in MR neuroimaging. NeuroImage, 165, 238–250.
* [[Xie2017]](https://www.spiedigitallibrary.org/journals/Journal_of_Biomedical_Optics/volume-22/issue-11/116006/Wide-field-spectrally-resolved-quantitative-fluorescence-imaging-system--toward/10.1117/1.JBO.22.11.116006.full) Xie, Y., Thom, M., Ebner, M., Wykes, V., Desjardins, A., Miserocchi, A., Ourselin, S., McEvoy, A. W., and Vercauteren, T. (2017). Wide-field spectrally resolved quantitative fluorescence imaging system: toward neurosurgical guidance in glioma resection. Journal of Biomedical Optics, 22(11).
* [[Ranzini2017]](https://mski2017.files.wordpress.com/2017/09/miccai-mski2017.pdf) Ranzini, M. B., Ebner, M., Cardoso, M. J., Fotiadou, A., Vercauteren, T., Henckel, J., Hart, A., Ourselin, S., and Modat, M. (2017). Joint Multimodal Segmentation of Clinical CT and MR from Hip Arthroplasty Patients. MICCAI Workshop on Computational Methods and Clinical Applications in Musculoskeletal Imaging (MSKI) 2017.
* [[Ebner2017]](https://link.springer.com/chapter/10.1007%2F978-3-319-52280-7_1) Ebner, M., Chouhan, M., Patel, P. A., Atkinson, D., Amin, Z., Read, S., Punwani, S., Taylor, S., Vercauteren, T., and Ourselin, S. (2017). Point-Spread-Function-Aware Slice-to-Volume Registration: Application to Upper Abdominal MRI Super-Resolution. In Zuluaga, M. A., Bhatia, K., Kainz, B., Moghari, M. H., and Pace, D. F., editors, Reconstruction, Segmentation, and Analysis of Medical Images. RAMBO 2016, volume 10129 of Lecture Notes in Computer Science, pages 3–13. Springer International Publishing.

[mebner]: https://www.linkedin.com/in/ebnermichael
[tig]: http://cmictig.cs.ucl.ac.uk
[bsd]: https://opensource.org/licenses/BSD-3-Clause
[giftsurg]: http://www.gift-surg.ac.uk
[cmic]: http://cmic.cs.ucl.ac.uk
[guarantors]: https://guarantorsofbrain.org/
[ucl]: http://www.ucl.ac.uk
[uclh]: http://www.uclh.nhs.uk
[epsrc]: http://www.epsrc.ac.uk
[wellcometrust]: http://www.wellcome.ac.uk
[mssociety]: https://www.mssociety.org.uk/
[nihr]: http://www.nihr.ac.uk/research
[itkniftymic]: https://github.com/gift-surg/ITK_NiftyMIC/wikis/home
[niftymic]: https://github.com/gift-surg/NiftyMIC
[nsol]: https://github.com/gift-surg/NSoL
[simplereg]: https://github.com/gift-surg/SimpleReg
[simplereg-dependencies]: https://github.com/gift-surg/SimpleReg/wikis/simplereg-dependencies
[pysitk]: https://github.com/gift-surg/PySiTK
[volumetricreconstructionfromprintedfilms]: https://github.com/gift-surg/VolumetricReconstructionFromPrintedFilms

# Multi-dimensional  B-spline
The recent development of single molecule imaging techniques has enabled not only high accuracy spatial resolution imaging but also information rich functional imaging. Abundant information of the single molecules can be encoded in its diffraction pattern and be extracted precisely (e.g. 3D position, wavelength, dipole orientation). However, sophisticated high dimensional point spread function (PSF) modeling and analyzing methods have greatly impeded the broad accessibility of these techniques. Here, we present a graphics processing unit (GPU)-based B-spline PSF modeling method which could flexibly model high dimensional PSFs with arbitrary shape without greatly increasing the model parameters. Our B-spline fitter achieves 100 times speed improvement and minimal uncertainty for each dimension, enabling efficient high dimensional single molecule analysis. We demonstrated, both in simulations and experiments, the universality and flexibility of our B-spline fitter to accurately extract the abundant information from different types of high dimensional single molecule data including multicolor PSF (3D + color), multi-channel four-dimensional 4Pi-PSF (3D + interference phase) and five-dimensional vortex PSF (3D + dipole orientation).
![fig 2](https://github.com/Li-Lab-SUSTech/MLE-by-bspline/assets/67534747/f92d3bc3-f8b1-4c2d-a00a-c9767e777eab)
# Requirements
* Matlab R2019a or newer  
* The GPU fitter requires:
  - Microsoft Windows 10 or newer, 64-bit
  - CUDA capable graphics card, minimum Compute Capability 6.1

# Run the demo
For multi-dimensional PSF simulation and localization, three example codes are available in the file:  
* 1 demo_4pi/bspline_based_4piloc_v4.m
* 2 demo_color_classification/bspline_based_color_classification_v3.m
* 3 demo_orientation/bspline_based_5D_fitting_v3.m
# GUI of color classification and orientation localization
For easier and faster calling of the multidimensional fitter, we integrated it into the lightweight single-molecule fitter in file [MutiD_fitter](https://baidu.com) and example datas in [here].

# Contact
For any questions / comments about this software, please contact [Li Lab](https://faculty.sustech.edu.cn/liym2019/en/).

# Copyright
Copyright (c) 2023 Li Lab, Southern University of Science and Technology, Shenzhen.

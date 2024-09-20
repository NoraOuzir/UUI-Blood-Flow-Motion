# Joint Blood Flow, Tissue and Motion Estimation for Ultrafast Ultrasound Imaging 
Demo code for the paper: 

N. Ouzir, V. Pustovalov, D.-H. Pham, D. kouamé, and J.-C. Pesquet, *A Proximal Algorithm for Joint Blood Flow Computation and Tissue Motion Compensation in Doppler Ultrafast Ultrasound Imaging*, 2023.

- run 'ḿain.m' to load the data, run the multiresolution scheme with 6 levels, compute errors and plot the results. 
- 'ComputeFlow.m' runs the proximal algorithm for 1 level and solves the following optimization problem :

## OPTIMIZATION PROBLEM 


  $$min_{B,T,d} \Vert S+ GS D(d) - (T+B)\Vert^2_F + \mu \Vert T \Vert^* + \lambda  \Vert B \Vert _{12} + \rho \mathcal{R}(d)$$
 
    S: images
    B: blood 
    T: tissues 
    d: motions
    G: image gradient operator

    ||.||*   : nuclear norm 
    ||.||_12 : L12-norm for row-sparsity (or L1)
    ||.||_F  : Frobenius norm
    R(.)     : Tikhonov or TV-regularization 

  - Inputs: S , rho, mu, l, initializations
  - Outputs: B, T, d 

  - Hyperparameters : rho, mu, l 

  - Optimization parameters : max iter, tolerance, gamma (step) 



## SIMULATED DATA WITH MOTIONS 
The simulated data is in the folder named  'data'
Data contains: 4 sequences with 4 types of motion ####


#### Data without noise
MAP_B1, MAP_B5, MAP_T1, MAP_T5
containing motions and max amplitudes: Bspline 1, Bspline 5, Translation 1, Translation 5.

#### Grountruth flow
MaskFlow - Flwo mask
MAP_Flow  - Groundtruth flow without motion

#### Noise matrices 35 dB 
Noise_B1, Noise_B5, Noise_T1, Noise_T5



## RESULTS 
### Examples of regularization parameters for R_tik (tikhonov_box): 

resLevels      = 6; 						           % multiresolution levels


*****************************************************************
---> T1: TIME =26.4942|NRMSE = 0.1416|PSNR =18.1532|SSIM = 0.6314

---> T5: TIME =26.4453|NRMSE = 0.1411|PSNR =18.1851|SSIM = 0.5860

H              = 1; 

rho            = 1e4;                                                      % motion smoothness 

mu0            = 1.7e-2;                                                   % low-rank parameter

l0             = 6.5;                                                      % flow sparsity

*****************************************************************
---> B1: TIME =26.5007|NRMSE = 0.1643|PSNR =16.8652|SSIM = 0.3854

H              = 1 ; 

rho            = 5e-1;                                                     % motion smoothness 

mu0            = 1.7e-2;                                                   % low-rank parameter

l0             = 5.3;                                                      % flow sparsity
*****************************************************************
---> B5: TIME =25.7756|NRMSE = 0.2705|PSNR =12.5314|SSIM = 0.1930
 
H              = 5;

rho            = 1e-1;                                                      % motion smoothness 

mu0            = 1.7e-2;                                                    % low-rank parameter

l0             = 5.3;                                                       % flow sparsity
*****************************************************************  

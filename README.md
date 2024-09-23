# Joint Blood Flow, Tissue and Motion Estimation for Ultrafast Ultrasound Imaging 
Demo code for the paper: 

N. Ouzir, V. Pustovalov, D.-H. Pham, D. kouamé, and J.-C. Pesquet, *A Proximal Algorithm for Joint Blood Flow Computation and Tissue Motion Compensation in Doppler Ultrafast Ultrasound Imaging*, 2023.

- run 'main.m' to load the data, run the multiresolution scheme with 6 levels, compute errors and plot the results. 
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
Data contains 4 sequences with 4 types of motion ####
  
### Data without noise: 
    MAP_B1 : data with Bspline motions of max magnitude 1
    MAP_B5 : data with Bspline motions of max magnitude 5
    MAP_T1 : data with translations of max magnitude 1
    MAP_T5 : data with translations of max magnitude 5 
### Grountruth flow: 
    MaskFlow : flow mask
    MAP_Flow : groundtruth flow without motion 
### Noise matrices 35 dB : 
    Noise_B1 : noise for MAP_B1 sequences
    Noise_B5 : noise for MAP_B5 sequences
    Noise_T1 : noise for MAP_T1 sequences
    Noise_T5 : noise for MAP_T5 sequences



## RESULTS 
Examples of regularization parameters and results with $$\mathcal{R}_{tik}$$ (tikhonov_box): 
#### Translations 
    resLevels      = 6;                                                        % multiresolution levels
    H              = 1;                                                        % no filter
    rho            = 1e4;                                                      % motion smoothness 
    mu0            = 1.7e-2;                                                   % low-rank parameter
    l0             = 6.5;                                                      % flow sparsity

    ---> T1: TIME = 26.4942 | NRMSE = 0.1416 | PSNR =18.1532 | SSIM = 0.6314

    ---> T5: TIME = 26.4453 | NRMSE = 0.1411 | PSNR =18.1851 | SSIM = 0.5860

#### Bspline motions
    resLevels      = 6;                                                        % multiresolution levels
    H              = 1 ;                                                       % no filter
    rho            = 5e-1;                                                     % motion smoothness 
    mu0            = 1.7e-2;                                                   % low-rank parameter
    l0             = 5.3;                                                      % flow sparsity

    ---> B1: TIME = 26.5007 | NRMSE = 0.1643 | PSNR =16.8652 | SSIM = 0.3854
*****************************************************************
    resLevels      = 6;                                                        % multiresolution levels
    H              = 5;                                                        % apply Gaussian filter of size 5
    rho            = 1e-1;                                                     % motion smoothness 
    mu0            = 1.7e-2;                                                   % low-rank parameter
    l0             = 5.3;                                                      % flow sparsity
    
    ---> B5: TIME = 25.7756 | NRMSE = 0.2705 | PSNR =12.5314 | SSIM = 0.1930
  

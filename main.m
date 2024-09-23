%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%        -------------------------------------------------------
%       |        JOINT BLOOD, MOTION & TISSUE ESTIMATION        |
%        -------------------------------------------------------
%
%  min_{B,T,d} ||S+GS*D(d)-(T+B)||^2_F + mu*||T||* + l*||B||_12 + rho*R(d)
% 
%  - S: images
%    B: blood 
%    T: tissues 
%    d: motions
%    G: image gradient operator
%
%  - ||.||*   : nuclear norm 
%    ||.||_12 : L12-norm for row-sparsity (or L1)
%    ||.||_F  : Frobenius norm
%    R(.)     : Tikhonov or TV-regularization 
%
%  - Inputs: S , rho, mu, l, initializations
%  - Outputs: B, T, d 
%
%  - Hyperparameters : rho, mu, l 
%
%  - Optimization parameters : max iter, tolerance, gamma (step) 
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

p = genpath('../UUI-Blood-Flow-Motion');
addpath(p);

%% -- LOAD DATA -----------------------------------------------------------
clear;
data_folder = 'data';
load([data_folder '/MaskFlow.mat'])
load([data_folder '/MAP_Flow.mat'])
load([data_folder '/MAP_T5.mat'])
load([data_folder '/Noise_T5.mat'])
% 

Bgt       = MAP_Flow.*MaskFlow;                                            % Groundtruth blood flow
S         = MAP+Noise35dB;                                                 % Images + noise

clear MAP MAP_Flow MaskFlow;

S    = S./max(abs(S(:)));                                                  % Normalize image sequence


%% -- INIT + PARAMETER CHOICES --------------------------------------------
clear B T d;
groundtruth = 1;                                                           % compute errors if 1 
[Nx,Nz,Nt]  = size(S);

Nf              = Nt;                                                      % number of frames to process
resLevels       = 6;                                                       % multiresolution scheme levels: based on max expected displacement
resLevels       = 2.^(flip(1:resLevels)-1);

% Initial displacements and Blood at level j
dx_j            = zeros([Nx Nz Nf]);
dz_j            = dx_j;
B_j             = zeros(ceil(Nx/resLevels(1)),ceil(Nz/resLevels(1)),Nf);  
S_j             = zeros(Nx,Nz,Nf);

% Grid of original locations for warping
[x,z]           = meshgrid(1:Nz,1:Nx);

% Optimization parameters
optimParameters = struct('gamma',1.95,'max_iter',300,'tolerance',1e-3);    % step, max iterations and tolerance

% Motion and blood regularizations
optimParameters.motionreg_type = 'Tikhonov_box';                           % Other choices: 'TVl1_box_acc';'TVl2_box_acc';'Tikhonov';'TVl1_box';'TVl2_box';
optimParameters.bloodreg_type  = 'L12';                                    % Or 'L1'

%****************************************
% Regularization parameters | CHANGE THIS
%****************************************
rho            = 1e4;                                                      % spatial smoothness parameter for motions 
mu0            = 1.7e-2;                                                   % low-rank parameter
l0             = 6.5;                                                      % sparsity parameter
%****************************************
hyperParameters = struct('rho',rho,'mu0',mu0,'lamda',[],'beta',0);         % beta : regularization parameter when using a quadratic regularization
                                                                           % with a reference d (dinit): beta*||d-dinit||^2_F     

% Interp and warping filter
Hsize = 1;                                                                 % 1 means no filtering, but if the distortions are too large, 
H     = fspecial('gaussian',Hsize,0.6*sqrt(.5^-2 -1));                     % it is necessary to filter to reduce interpolation errors (H = 5 or 3)


fprintf('*-----------------------------------------------------------*\n')
fprintf('-- mu0 = %.2e -- l0 = %.2f -- rho = %.2e -- H = %i: \n',mu0,l0,rho,Hsize)
fprintf('*-----------------------------------------------------------*\n')

%% -- START MULTIRESOLTUION SCHEME ----------------------------------------

tic;
for j = 1:numel(resLevels)
   
   % Image size at level j
   Nxlevel = ceil(Nx/resLevels(j));
   Nzlevel = ceil(Nz/resLevels(j));    

   % Apply displacements from previous level and resize images 
   for jj = 1:Nf
   S_j(:,:,jj) = interp2(S(:,:,jj),x+dx_j(:,:,jj),z+dz_j(:,:,jj),'bicubic',0);
   end

   % Process borders and normalize (filter if extreme deformations)
   S_j(1,:,:)  = S(1,:,1:Nf);   S_j(end,:,:) = S(end,:,1:Nf); 
   S_j(:,1,:)  = S(:,1,1:Nf);   S_j(:,end,:) = S(:,end,1:Nf) ;  
   S_j         = imfilter(S_j,H,0);
   S_j_resized = imresize(S_j,[Nxlevel Nzlevel],'bicubic');  
   S_j_resized = S_j_resized ./max(abs(S_j_resized(:)));
   
   % Choose initializations for B and T
   optimParameters.Tinit  = S_j_resized ;
   optimParameters.Binit  = zeros(Nxlevel,Nzlevel,Nf);
 
   % Adjust sparsity parameter for the current level
   hyperParameters.lamda = l0/sqrt(max(Nxlevel*Nzlevel,Nf));
 
   fprintf('Level | %i | mu0 = %.2e -- l = %.2e -- rho = %.2e: \n',numel(resLevels)-j+1,hyperParameters.mu0,hyperParameters.lamda,hyperParameters.rho)
   fprintf('-----------\n')
    
   % Optimization
   [B,T,dx,dz,cost] = ComputeFlow(S_j_resized,hyperParameters,optimParameters);

   % Increment motions
   dx_j = dx_j+imresize(dx,[Nx Nz],'bicubic')*resLevels(j);
   dz_j = dz_j+imresize(dz,[Nx Nz],'bicubic')*resLevels(j); 

end

    TIME = toc 

    % Final displacements 
    dx  = dx_j+dx; 
    dz  = dz_j+dz; 

%%  -- RESULTS ------------------------------------------------------------

    % Compute errors   
    if groundtruth 
    B_crop   = B(j+1:end-j,j+1:end-j,1:Nf);   %Process borders after multiresolution scheme (j number of levels)
    Bgt_crop = Bgt(j+1:end-j,j+1:end-j,1:Nf);
     
    [NRMSE, PSNR] = ComputeErrors(PowerDopplerImage(B_crop,0),PowerDopplerImage(Bgt_crop,0))
    SSIM          = ssim(PowerDopplerImage(B,0,35),PowerDopplerImage(Bgt,0,35))
    end


   % Plot Power Doppler images and motions
    Amp         = 35;    
    gd          = ceil(Nz/20);
    [x_dd,y_dd] = meshgrid(1:gd:Nz,1:gd:Nx);

    figure(1),clf; colormap(hot),
    subplot(141), imagesc(PowerDopplerImage(S(:,:,1:Nf),0,Amp)), xlabel('S'),caxis([-Amp 0]); 
    subplot(142), imagesc(PowerDopplerImage(B_crop(:,:,1:Nf),0,Amp)), xlabel('B'),caxis([-Amp 0]); 
    subplot(143), imagesc(PowerDopplerImage(T,0,Amp)), xlabel('T'),caxis([-Amp 0]); 
    subplot(144), imagesc(abs(complex(dx(:,:,ceil(Nf/2)),dz(:,:,ceil(Nf/2))))), hold on;   
    quiver(x_dd,y_dd,dx(1:gd:Nx,1:gd:Nz,ceil(Nf/2)),dz(1:gd:Nx,1:gd:Nz,ceil(Nf/2)),...
               2,'LineWidth',1.5), hold off;colorbar; caxis([0 (max(abs(complex(dx(:),dz(:))))+1e-7)]); xlabel('d_{Nt/2}'), drawnow;

    % Play video (original sequence | sequence before the final level | blood | tissues)
    fps = 100; 
    tre = 1e-2;

    implay([iq2bmode(S(j+1:end-j,j+1:end-j,1:Nf),tre) iq2bmode(S_j(j+1:end-j,j+1:end-j,1:Nf),tre)  iq2bmode(B_crop,tre) iq2bmode(T(j+1:end-j,j+1:end-j,1:Nf),tre)],fps);


    %% -- functions -------------------------------------------------------
    
    % Function for creating the power doppler image as in (15)
    function [logPWTD,PWTD1,varargout]=PowerDopplerImage(X,plot_image,varargin)
    if nargin>2
        Amp = varargin{1};
    else
        Amp = 35;
    end

    [~,~,Nt] =size(X);
    PW = 1/Nt*sum(abs(X).^2,3);
    PWTD = PW/max(max(PW));
    PWTD1=max(PWTD,10^(-Amp/10));
    logPWTD = 10*log10(max(PWTD,10^(-Amp/10)));

    if plot_image
        h1=figure();
        set(gcf,'Color',[1,1,1]);
        colormap hot;
        h = imagesc(logPWTD(:,:),[-Amp,0]);
        xlabel('N_X [nb]');
        ylabel('N_Z [nb]')
        if nargout>3
            varargout{1} = h;
        end
    end
    end

    % Error computation
    function [NRMSE, PSNR] = ComputeErrors(DopplerB,DopplerBgt)

    DopplerBgt = DopplerBgt(:,:,1:size(DopplerB,3));

    NRMSE = (norm(DopplerBgt(:)-DopplerB(:))^2)/norm(DopplerBgt(:))^2;
    NRMSE = sqrt(NRMSE);

    dmax = 35;
    MSE = (norm(DopplerBgt(:)-DopplerB(:))^2)/numel(DopplerB);
    PSNR = 10*log10((dmax^2)/MSE);
    end
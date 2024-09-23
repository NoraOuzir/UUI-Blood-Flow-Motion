function [B,T,dx,dz,varargout] = ComputeFlow(S,hyperParameters,optimParameters)

    %% Input parmameters
    
    % Regularization
    rho       = hyperParameters.rho;                                       % motion spatial smoothness
    lamda     = hyperParameters.lamda;                                     % sparsity
    mu0       = hyperParameters.mu0;                                      % low-rankness
    dmin      = -1;                                                        % lower bound on the displacement values 
    dmax      =  1;                                                        % upper bound on the displacement values 
    
    % Optimization
    gamma     = optimParameters.gamma;                                     % constant for step size computation      
    tolerance = optimParameters.tolerance;                                 % tolerance for stopping criterion
    max_iter  = optimParameters.max_iter;                                  % max iterations 
   
    d_step_mode = 1;                                                       % 0: for separating dx and dz 1 for d = [dx(:);dz(:)]    
    dinit_reg   = 0;                                                       % 1: Add a quadratic regularization with a reference d (dinit): beta*||d-dinit||^2_F 
    beta        = 0;                                                       % Reference regularization parameter                                                            
    

    % Blood 
    if isempty(optimParameters.motionreg_type)
        motionreg_type = 'Tikhonov_box';
    else
        motionreg_type = optimParameters.motionreg_type;
    end

    if isempty(optimParameters.bloodreg_type)
        bloodreg_type = 'L12';
    else
        bloodreg_type = optimParameters.bloodreg_type;
    end
    
    % Motion
    if strcmp(motionreg_type,'Tikhonov_box')
         OFinit = 1;                                                       % 1: if initialize d with a few Optical flow steps (requires reference frame, chosen as the central frame)
    else
         OFinit = 0;
    end

    global prints 
    prints = 0;                                                            % Print progression
     
    caso = 0;                                                              % process as Casorati matrix and uses FFT 0 process as Nd array and ues connv
    if caso
         diml = ndims(S);
         rp   = [];
    else
         diml = ndims(S)+1;
         rp   = 1;
    end
    otherdims  = repmat({':'},1,diml-1); 

     
%% Sizes and kernels (global variables)

    global Nx Nz Nt;
    [Nx,Nz,Nt]  = size(S);

    im2caso = @(X) reshape(X,[Nx*Nz,Nt]);                                  % 3D sequence to 2D Casorati matrix
    caso2im = @(X) reshape(X,[Nx,Nz,Nt]);                                  % 2D Casorati matrix to 3D sequence
    
    % Choose convolution kernel for spatial gradient computation (motion)
    [kx,kz] = switch_kernel('simple');
    [kxz,~] = switch_kernel('laplacian');

    % Create circulant gradient operator matrices
    global Dx Dz Dxz;
    Dx  = psf2otf(-kx, [Nx Nz]);
    Dz  = psf2otf(-kz, [Nx Nz]); 
    Dxz = psf2otf(-kxz,[Nx Nz]); 
    
    if caso 
       Dx=Dx';  Dx=Dx(:); Dz=Dz';Dz=Dz(:); Dxz=Dxz';Dxz=Dxz(:);            % Resize operators for casorati matrix processing 
    end 
    
%% Variable Initialization
    
    % Motion initialization 
    if caso 
     d = zeros(Nx*Nz,Nt,2);
    else
    d  = zeros(Nx,Nz,Nt,2);                                                % Always zero because of multiresolution approach,
    end                                                                    % but can set OFinit =1 to perform a few Optical flow steps to init    
    
    % Dual variable for prox TV computation and others 
    if strcmp(motionreg_type,'Tikhonov') || strcmp(motionreg_type,'Tikhonov_box') 
    y = [];
    else
    y  = zeros(Nx,Nz,Nt,2,2);     
    end                                         

    S_lin       = zeros(Nx,Nz,Nt);                                         % init variables used during optim
    data_term_d = S_lin;
    
    
    % Blood initialization
    if isempty(optimParameters.Binit)
    B = zeros(Nx,Nz,Nt);
    else    
    B  = optimParameters.Binit;
    end

    % Tissue initialization
    if isempty(optimParameters.Tinit)
    T = S-B;                                                               % Another option is : T = repmat(S(:,:,ceil(Nt/2)), [1,1,Nt]);  
    else    
    T = optimParameters.Tinit;
    end
     
    T(isnan(T)) = 0; 
    B(isnan(B)) = 0; 
    S(isnan(S)) = 0; 
 
    if caso        
    B  = im2caso(B);                                                       % Resize variables for casorati matrix processing 
    T  = im2caso(T);
    end
   
%% Compute image gradients

    % choose kernel for approximating image derivatives 
    [kx_im,kz_im] = switch_kernel('central');                              % Other options: Bruhn, sobel, laplacian,, HS, laplcaian_HS,
                                                                           % DO NOT choose 'simple' !!!
    % Boundary processing 
    bound_options = 'replicate';                                           % Others: 'circular' 

    % Spatial gradient of the images with temporal averaging
    %   DxS = cat(3,.5*(convn(S(:,:,1:end-1),kx,'same')+convn(S(:,:,2:end),kx,'same')),.5*(convn(S(:,:,end),kx,'same')+convn(S(:,:,end-1),kx,'same')));
    %   DzS = cat(3,.5*(convn(S(:,:,1:end-1),kz,'same')+convn(S(:,:,2:end),kz,'same')),.5*(convn(S(:,:,end),kz,'same')+convn(S(:,:,end-1),kx,'same')));       
    % or
    %   DxS = cat(3,convn(S(:,:,1:end-1),0.25* [-1 1; -1 1],'same')+convn(S(:,:,2:end),0.25*[-1 1; -1 1],'same'),(convn(S(:,:,end),kz_im,'same')));
    %   DzS = cat(3,convn(S(:,:,1:end-1),0.25* [-1 -1; 1 1],'same')+convn(S(:,:,2:end),0.25*[-1 -1; 1 1],'same'),(convn(S(:,:,end),kz_im,'same')));
    
    
    % Simple spatial gradient of the images without temporal averaging     (uncomment one of the two options below to use this approach)
    DxS      = imfilter(S, -kx_im, 'corr', bound_options, 'same');  
    DzS      = imfilter(S, -kz_im, 'corr', bound_options, 'same');
  
    % Use the code below if images need to be filtered 
    % H = fspecial('gaussian',3,1);
    % DxS   = imfilter(imfilter(S,H), -kx_im, 'corr', bound_options, 'same');  
    % DzS   = imfilter(imfilter(S,H), -kz_im, 'corr', bound_options, 'same');

    % Regularization parameters adjusted to current level (see paper Setion IV.A.1)
    mu    = mu0 * sqrt( max( eig(im2caso(S)'*im2caso(S)) ) );
    lamda = lamda*mu;

    % Resize if needed and store -
    if caso 
        S   = im2caso(S);
        DxS = im2caso(DxS);
        DzS = im2caso(DzS);
    end

    DS  = cat(diml,DxS,DzS);
    DSC = conj(DS);
   
    normS  = norm(im2caso(S),'fro'); 

%% Compute step parameters    

    % Step size for B and T
    B_step  = gamma;                                                       % step size for (B) (step =gamma since Lipschitz cst =1)
    T_step  = gamma;                                                       % step size for (T) (step =gamma since Lipschitz cst =1)

    % Step size for d = [dx,dz]
    gamma_d = 1.95;                                                        % gamma for d step
    if d_step_mode
        if strcmp(motionreg_type,'Tikhonov_box')
          L_bt = max(abs(DxS(:)).^2 + abs(DzS(:).^2))+ 2*beta + 2*rho*8;   % Lipschitz constant
        else
          L_bt = max(abs(DxS(:)).^2 + abs(DzS(:)).^2) + 2*beta;            % Lipschitz constant
        end

        d_step   = gamma_d/L_bt;                                           % step size for d

    % Step size for dx and dz separately
    else
        if strcmp(motionreg_type,'Tikhonov_box')
            L_bt     = [max((abs(DxS(:)).^2)) , max((abs(DzS(:)).^2))] + 2*beta + 2*rho*4;      
            else
            L_bt     = [max((abs(DxS(:)).^2)) , max((abs(DzS(:)).^2))] + 2*beta;      
        end
        
        d_step     = gamma_d./L_bt;
    end
    

%% Compute proximal step matrix for the 'Tikhonov' alone case  (can be implemented as gradient step but this is faster)
    if  strcmp(motionreg_type,'Tikhonov') %|| OFinit 
        if d_step_mode
            IW  = 1./(2*d_step*rho*(abs(Dx).^2+ abs(Dz).^2 )+ 1);
            IW  = repmat(IW, [1 rp Nt 2]);
        else
        IW = repmat(1./(2*d_step*rho*(abs(Dx).^2+ abs(Dz).^2 )+ 1), [1 rp Nt 1]); % !! check this syntax
        clear rp;
        end
    else 
        IW =[];
    end

%% Init d with a few prox iterations if OFinit = 1 (Tikhonov reg only)
    if OFinit
        % Reference frame for initial OF computation 
        T = repmat(S(otherdims{1:end-1},1), [1 rp Nt]) -B;  
        d_step0 = 1*gamma_d/(max(abs(DxS(:)).^2 + abs(DzS(:)).^2) + 2*beta); 
        IW  = 1./(2*d_step0*1*rho*(abs(Dx).^2+ abs(Dz).^2 )+ 1);
        IW  = repmat(IW, [1 rp Nt 2]);
   
        for i=1:50
        dold = d; 
        S_lin = sum(d.*DS,diml) + S; 

        %------------------------------------------------------------------        
        %- Update T
        %------------------------------------------------------------------       
        %- Gradient step    
        T = T-T_step*(T+B-S_lin);        
        
        %-- Proximal steps
        T = prox_T(mu*T_step,T,caso);

        %------------------------------------------------------------------        
        %- Update B
        %------------------------------------------------------------------
        %- Gradient step
        B = B - B_step.*(T+B-S_lin);
 
        %- Proximal step
        B = prox_B(lamda*B_step, B, bloodreg_type,caso);   

        %------------------------------------------------------------------        
        %- Update d 
        %------------------------------------------------------------------
        %-- Gradient step
        d = d - d_step0 .* real(DSC.*(S_lin - T - B)); 
            
        %-- Proximal step
        d = real(ifft2(fft2(d).*IW)); 

        if (norm(d(:)-dold(:))/norm(dold(:)) < 1e-3)
            break; 
        end
        end

        if ~strcmp(motionreg_type, 'Tikhonov_box')
        y   = cat(5,convn(d,kx,'same'),convn(d,kz,'same'));
        end
    end

    d    = proj_box(d,dmin,dmax);                                          % Projection onto the interval [dmin, dmax]
    clear DxS DzS dold;
%% Optimization Algorithm

        for i=1:max_iter
        
        Bold = B; 
        Told = T; 
        dold = d; 
               
        S_lin = sum(d.*DS,diml) + S;                                       % Linearization of S(x+d)

        %  if ~mod(i,10)                                                   % Uncomment to update every 10 iteration
        %-----------------------------------------------------------------
        %- Update T
        %-----------------------------------------------------------------
 
        %- Gradient step   
        T = T-T_step*(T+B-S_lin);        
        
        %-- Proximal steps
        T = prox_T(mu*T_step,T,caso); T(isnan(T))=0;
        %   end

        %------------------------------------------------------------------        
        %- Update B
        %------------------------------------------------------------------
        
        %- Gradient step3
        B = B-B_step*(T+B-S_lin); 
        
        %- Proximal step
        B = prox_B(lamda*B_step,B,bloodreg_type,caso);     
       
        %------------------------------------------------------------------
        %- Update d
        %-----------------------------------------------------------------
        data_term_d = (S_lin - (T + B)./1);

        if i>0                                                              % change to value > 0 to start motion estimation after a few iters          
        %-- Gradient step
        d =  grad_d(d,DSC,data_term_d,d_step,rho,beta,motionreg_type,dinit_reg,caso);
            
        %-- Proximal step
        [d,y,cost.R(i)] = prox_d(d,y,rho,dmin,dmax,d_step,IW,motionreg_type,d_step_mode,kx,kz,caso);

        else 
        cost.R(i) = 0;
        end

        err1 = norm(B(:)-Bold(:))/normS;

      % Compute current cost ----------------------------------------------
        cost.h1(i)   = .5*sum(abs(data_term_d).^2,'all');                  % data fidelity term 
        cost.l1(i)   = cost_B(lamda,B,bloodreg_type,caso);                 % blood regularization  - L1 or L_{1,2}
        cost.nuc(i)  = cost_T(mu,T,caso);                                  % tissue regularization - nuclear norm       
        cost.reg(i)  = cost.l1(i)  +cost.nuc(i) + cost.R(i);
        cost.all(i)  = cost.h1(i) + cost.reg(i);
        cost.all_iter(i) =  cost.all(i);
        
      % print cost or change 
        if prints
        fprintf('( %i ) cost = %.4e | h1 = %.2e  | l1 = %.2e | nuc = %.2e | R = %.2e\n',i,cost.all(i),cost.h1(i),cost.l1(i),cost.nuc(i),cost.R(i)); 
        fprintf('*\n');
        fprintf('( %i ) -- Change B = %.2e \n',i,err1);
        end 

       %------------------------------------------------------------------      
        if i>1 % && ~mod(i,1) 
           % stop when relative cost function cahnge is below tolerance
           % if abs(cost.all_iter(i)-cost.all_iter(i-1))/cost.all_iter(i-1)<tolerance
           % break;
           % end

           % stop when relative variable change is below tolerance 
            if (err1 < tolerance)
                   break;            
            end
        end 
        
end 

%% Outputs  

    if caso 
    B  = caso2im(B); 
    T  = caso2im(T);
    dx = caso2im(d(:,:,1));
    dz = caso2im(d(:,:,2));
    else
        dx = d(:,:,:,1);   
        dz = d(:,:,:,2); 
    end
    
    if nargout>4 
        varargout{1}= cost; 
    end 

end

%% -- Functions -- Prox, projections, grads and costs

%-- Prox for blood regularization -- Sparsity -----------------------------
    function r = prox_B(tau,B,norm_type,caso)
    global Nx Nz Nt;
    if ~caso
        B = reshape(B,[Nx*Nz,Nt]);
    end
    
    if strcmp(norm_type,'L1')
        r = sign(B) .* max(abs(B) - tau, 0);                                   % shrinkage operator
    elseif strcmp(norm_type,'L12')
        r = rowShrinkage(B,tau);                                               % row-shrinkage operator
    else
        printf('Unknown B regularization')
    end
    
    if ~caso
        r = reshape(r,[Nx,Nz,Nt]);
    end
    end


%-- Prox for tissue regularization -- Low-rank-----------------------------
    function r = prox_T(tau,T,caso)
    global Nx Nz Nt;
    if ~caso
        T = reshape(T,[Nx*Nz,Nt]);
    end
    
    [U, D, V] = svdecon(T);                                                    % shrinkage operator for singular values
    r = U*prox_B(tau,D,'L1',1)*V';
    
    if ~caso
        r = reshape(r,[Nx,Nz,Nt]);
    end
    end

%-- Projection onto the box constraint ------------------------------------
    function p = proj_box(x,low,high)
    p = min(x, high);
    p = max(p, low);
    end

%-- Cost for blood regularization -- Sparsity -----------------------------
    function c = cost_B(lamda,B,norm_type,caso)
    global Nx Nz Nt;
    
    switch norm_type
        case 'L1'
            c  =  lamda * sum(abs(B(:)));                                      % L1 norm
        case 'L12'
            if ~caso
                B = reshape(B,[Nx*Nz,Nt]);
            end
            c  = lamda * sum(vecnorm(B,2,2));                                  % L_{1,2} norm
        otherwise
            error('Unknown B regularization')
    end
    end

%-- Cost for tissue regularization -- Nuclear norm ------------------------                   
    function c = cost_T(mu,T,caso)
    global Nx Nz Nt;
    if ~caso
        T = reshape(T,[Nx*Nz,Nt]);
    end
    [~,tmp,~] = svdecon(T);
    %c  = mu * norm(diag(tmp),1);
    c  = mu * sum(diag(tmp));
    end

%-- Indicator function ---------------------------------------------------- 
    function p = indicator_box(x, low, high)
    % The procedure evaluate the indicator function of the constraint set:
    %
    %                     low <= x <= high
    %
    % When the input 'x' is an array, the output 'p' is the element-wise sum.
    %
    %  INPUTS
    % ========
    %  x    - ND array
    %  low  - scalar or ND array with the same size as 'x'
    %  high - scalar or ND array with the same size as 'x'
    
    % check inputs
    if ~isscalar(low) && any(size(low) ~= size(x))
        error('''low'' must be either scalar or the same size as ''x''')
    end
    if ~isscalar(high) && any(size(high) ~= size(x))
        error('''high'' must be either scalar or the same size as ''x''')
    end
    if any( low(:) >= high(:) )
        error('''low'' must be lower than ''high''')
    end
    %-----%
    % x(isnan(x)) = 0;
    % check the constraint
    mask = low <= x & x <= high;
    
    % evaluate the indicator function
    if all(mask(:))
        p = 0;
    else
        p = Inf;
    end
    end

%-- Prox and cost of motion regularization --------------------------------
    function [d,y,cost] = prox_d(d,y,rho,dmin,dmax,d_step,IW,motionreg_type,d_step_mode,kx,kz,caso)
    global Nx Nz Nt Dx Dz;
    
    switch motionreg_type
        case 'Tikhonov'                                                        % Tikhonov regularization alone
            ConvC = @(X,D)  real(ifft2(fft2(X).*D));                           % Convolution in the FFT domain
            if caso
                d(:,:,1)    = ConvC(d(:,:,1),IW(:,:,1));
                d(:,:,2)    = ConvC(d(:,:,2),IW(:,:,2));
                cost = rho * (sum(cat(4,ConvC(d(:,:,1),Dx),ConvC(d(:,:,1),Dz),...
                    ConvC(d(:,:,2),Dx),ConvC(d(:,:,2),Dz)).^2,'all'));
            else
                d = ConvC(d,IW);
                cost = rho * (sum(cat(4,bound0(convn(d(:,:,:,1),kx,'same'),1),...
                    bound0(convn(d(:,:,:,1),kz,'same'),2),bound0(convn(d(:,:,:,2),kx,'same'),1),...
                    bound0(convn(d(:,:,:,2),kz,'same'),2)).^2,'all'));
            end
    
        case 'Tikhonov_box'
            d    = proj_box(d,dmin,dmax);                                      % Projection onto the interval [dmin, dmax]
            if caso
                Convx = @(X)  bound0(convn(reshape(X,[Nx Nz Nt]),kx,'same'),1);
                Convz = @(X)  bound0(convn(reshape(X,[Nx Nz Nt]),kz,'same'),2);
                cost =  rho * (sum([Convx(d(:,:,1));Convz(d(:,:,1))].^2,'all') ...
                    + sum([Convz(d(:,:,2)); Convx(d(:,:,2))].^2,'all'))+indicator_box(d,dmin,dmax);
            else
    
                bound_options = 'replicate';%'circular'; %
                Convx = @(X)  imfilter(X,kx,'same',bound_options);
                Convz = @(X)  imfilter(X,kz,'same',bound_options);
                cost =  rho * sum(cat(5,Convx(d),Convz(d)).^2,'all') + indicator_box(d,dmin,dmax);
            end
    
        case 'TVl1_box'                                                        % Anisotropic total-variation with p=1 and box constraints (solved with Dual-foward backward)
            [d,y,cost] = DFB_TV(d,y,dmin,dmax,rho,d_step,1,d_step_mode,0);
    
        case 'TVl2_box'                                                        % Isotropic total-variation with p=2 and box constraints (solved with Dual-foward backward)
            [d,y,cost] = DFB_TV(d,y,dmin,dmax,rho,d_step,2,d_step_mode,0);
    
        case 'TVl1_box_acc'                                                    % Anisotropic total-variation with p=1 and box constraints (solved with ACCELERATED Dual-foward backward)
            [d,y,cost] = DFB_TV(d,y,dmin,dmax,rho,d_step,1,d_step_mode,1);
    
        case 'TVl2_box_acc'                                                    % Anisotropic total-variation with p=1 and box constraints (solved with ACCELERATED Dual-foward backward)
            [d,y,cost] = DFB_TV(d,y,dmin,dmax,rho,d_step,2,d_step_mode,1);
    
        otherwise
            error('Unknown motion regularization');
    end
    
    dinit_reg = 0;
    
    if dinit_reg
        dinit = 0;
        cost = cost + beta*sum(abs(d-dinit).^2,'all');                         % Add regularization based on a reference d
    end
    end

%-- Gradient step for the motion term -------------------------------------
    function d = grad_d(d,DSC,data_term_d,d_step,rho,beta,motionreg_type,dinit_reg,caso)    
    global Nx Nz Nt Dxz;
    if strcmp(motionreg_type,'Tikhonov_box')
        if caso
            im2caso2 = @(X) reshape(X,[Nx*Nz,Nt,2]);
            caso2im2 = @(X) reshape(X,[Nx,Nz,Nt,2]);
            kxz = [0,-1,0;-1,4,-1;0,-1,0];
            d = d - d_step .*(real(DSC.*data_term_d) + im2caso2(2*rho*(...
                imfilter(caso2im2(d),kxz,'same','circular'))));
    
        else
    
            bound_options = 'replicate';%'circular';
            d = d - d_step .* (real(DSC.*data_term_d) + 2*rho*(...
                imfilter(d,[0,-1,0;-1,4,-1;0,-1,0],'same',bound_options)));
        end
    
    else
        d = d - d_step*real(DSC.*data_term_d);
    end
    
    dinit = 0;
    if dinit_reg
        d = d - d_step*2*beta*(d - dinit);                                     % Add regularization based on a reference d
    end
    end

%-- Choose kernel for gradients -------------------------------------------
    function [kx,kz] = switch_kernel(kernel_type)        
    
            switch kernel_type
                case 'simple'
            kx = [1,-1];                                                      
            kz = [1;-1];
                case 'central' 
            kx = .5*[1,0,-1];                                                     
            kz = .5*[1;0;-1];
                case 'sobel'
            kz =(1/8)*[1 2 1;0 0 0;-1 -2 -1];                                   
            kx = kz' ; 
                case 'HS'                                                      % Horn&Schunck
            kx = -0.5*[-1 1; -1 1];                                                 
            kz = -0.5*[-1 -1; 1 1]; 
                 case 'Bruhn'                                                  %Bruhn et al "combing "IJCV05' page218
            kx = [-1 9 -45 0 45 -9 1]/60; 
            kz = kx';
                case 'laplacian'
            kx = [0,1,0;1,-4,1;0,1,0]; 
            kz = kx;
                case 'laplacian_HS'
            kx = (-1/6)*[0.5,1,0.5;1,-6,1;.5,1,.5]; 
            kz = kx;
                otherwise 
            printf('Unknown kernel type')
            end
    end

%-- Add zeros to boundaries -----------------------------------------------
    function im0 = bound0(im,dir)
    [Nx,Nz,~] = size(im);
    otherdimsx  = repmat({':'},1, ndims(im)-1);
    otherdimsz  = repmat({':'},1, ndims(im)-2);
    
    if dir >1
        im(Nx,otherdimsx{:}) = 0;
    else
        im(:,Nz,otherdimsz{:}) = 0;
    end
    im0 = im;
    end

%-- Row Shrinkage ---------------------------------------------------------
    function P = rowShrinkage(P,lambda)
    n = size(P,2);
    r = vecnorm(P,2,2);
    r= repmat(r, [1 n]);
    z=zeros(size(r));
    P = (P./r).*max(z,(r-lambda));
    
    % % Z = zeros(size(P));
    % % idx = r>lambda;
    % % Z(idx,:) = Z(idx,:).*(vecnorm(Z(idx,:),2,2)-lambda)./vecnorm(Z(idx,:),2,2);
    % % P = Z;
    
    P(isinf(P)) = 0;
    P(isnan(P)) = 0;
    end

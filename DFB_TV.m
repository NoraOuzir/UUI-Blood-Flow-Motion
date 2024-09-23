function  [d,y,cost] = DFB_TV(d_tilde,y,dmin,dmax,rho,d_step,p,d_step_mode,acc)
global Nx Nz Nt prints; 

% Dimensions 
dim        = ndims(d_tilde);
otherdims  = repmat({':'},1,ndims(y)-1);
otherdims2 = repmat({':'},1,dim-2);

if d_step_mode
dimx = 4;
 if dim<dimx
    im2caso = @(X) reshape(X,[Nx*Nz,Nt,2]);
    d_tilde = reshape(d_tilde,[Nx,Nz,Nt,2]);
    otherdims2 = repmat({':'},1,dim-1);
 end
else 
dimx = 3; 
 if dim<dimx
    im2caso = @(X) reshape(X,[Nx*Nz,Nt]);
    d_tilde = reshape(d_tilde,[Nx,Nz,Nt]);
    otherdims2 = repmat({':'},1,dim-1);
 end
end 

% Spatial Gradient Operator
[kx,kz] = switch_kernel('simple');

% Forward and backward differences (with Neumann boundary conditions)
 hor_forw = @(X) bound0(convn(X,kx,'same'),1); 
 ver_forw = @(X) bound0(convn(X,kz,'same'),2);


if numel(kx)>2
hor_back = @(X) convn(X,-kx,'same'); 
ver_back = @(X) convn(X,-kz,'same');
else
hor_back = @(X) [-X(:,1,otherdims2{:}),convn(X(:,1:end-1,otherdims2{:}),-kx,'valid'), X(:,end-1,otherdims2{:})]; 
ver_back = @(X) [-X(1,:,otherdims2{:});convn(X(1:end-1,:,otherdims2{:}),-kz,'valid'); X(end-1,:,otherdims2{:})]; 
end

% Direct and adjoint operators
h.dir_op = @(x) cat(dimx+1, hor_forw(x), ver_forw(x) );
h.adj_op = @(y) hor_back( y(otherdims{:}, 1) ) + ver_back( y(otherdims{:}, 2) );


% Optimization parameters
h.beta = 8;                                                                % operator norm
if acc
    sigma = 1;                                                             % optimization constant different when Accelerated dualFB 
    ksi    = 2.1;  % 10;%                                                  % optimization constant (smaller is faster)
    tol    = 1e-2;                                                         % tolerance for stopping criterion 
    else
    sigma  = 1.95;
    tol    = 1e-3;                                                         % tolerance for stopping criterion 
end
sigma2 = sigma/h.beta;                                                     % optimization constant
iter   = 50;                                                               % max iterations

% Function and proximity operator 
if p<2
    % TV-l1 (p=1)  
    h.fun  = @(y) fun_L1(y, rho, 4);
    h.prox = @(y,gamma) prox_L1(y, gamma*(rho*d_step), 4);
    else
    % TV-l2 (p=2)    
    h.prox = @(y,gamma) prox_L2(y, gamma*(rho*d_step), 4);
    h.fun  = @(y) fun_L2(y, rho, 4);
end

% Initialization
d = d_tilde;
if acc 
    z = y;                                                                 % dual variable for accelerated dualFB
end

% Start optimization: accelerated Dual Forward Backward
if acc  
    for i=1:iter   
    d0      = d;  
    y0      = y; 
    d       = proj_box(d_tilde-h.adj_op(z),dmin,dmax);
    y_tilde = z + sigma2*h.dir_op(d);
    y       = y_tilde - sigma2*h.prox(y_tilde/sigma2,1/sigma2);  
    ll = i/(i+1+ksi);  
    z  = y + ll.*(y - y0); 
    
    % stopping criterion
     % cost2(i) = h.fun(h.dir_op(d))+ indicator_box(d, dmin,dmax);
    if norm( d(:) - d0(:) ) < tol * norm( d0(:) )% && i > 5
       if prints
       fprintf('-> End of prox_{TV} at  %i \n', i);
       end
       break;
    end
    end
else
% Start optimization: normal Dual Forward Backward
    for i=1:iter   
        d0      = d;     
        d       = proj_box(d_tilde-h.adj_op(y),dmin,dmax);

        y_tilde = y + sigma2*h.dir_op(d);
        y       = y_tilde - sigma2*h.prox(y_tilde/sigma2,1/sigma2);  

        % stopping criterion
       %cost3(i) = h.fun(h.dir_op(d))+ indicator_box(d, dmin,dmax);
        if norm( d(:) - d0(:) ) < tol * norm( d0(:) ) % && i > 3
           if prints 
           fprintf('-> End of prox_{TV} at  %i \n', i);
           end
           break;
        end
    end
end

cost =0;%h.fun(h.dir_op(d))+ indicator_box(d, dmin,dmax);%0;%0;%

if dim<dimx
    d =im2caso(d); 
end
end

%% Functions and proximity operators
function p = proj_box(x,low,high)
p = min(x, high);
p = max(p, low);
end

%--------------------------------------------------------------------------
function p = prox_L2(x, gamma, dir)
% default inputs
if nargin < 3 || (~isempty(dir) && dir == 0)
    dir = [];
end

% check input
sz = size(x); sz(dir) = 1;
if any( gamma(:) <= 0 ) || ~isscalar(gamma) && any(size(gamma) ~= sz)
    error('''gamma'' must be positive and either scalar or compatible with the blocks of ''x''')
end
%-----%

% linearize
sz = size(x);
if isempty(dir)
    x = x(:);
end
    
% compute the prox
% xx = sqrt( sum(x.^2, dir) );
xx = vecnorm(x,2, dir);
pp = max(0, 1 - gamma ./ xx);
p = bsxfun(@times, x, pp);

% revert back
p = reshape(p, sz);

 end
 
%--------------------------------------------------------------------------
function p = prox_L1(x, gamma, ~)
    p = sign(x) .* max(abs(x) - gamma, 0);                                   % shrinkage operator
end

%--------------------------------------------------------------------------
function p = fun_L2(x, gamma, dir)
%function p = fun_L2(x, gamma, dir)
%
% This procedure evaluates the function:
%
%                    f(x) = gamma * ||x||_2
%
% When the input 'x' is an array, the computation can vary as follows:
%  - dir = 0 --> 'x' is processed as a single vector [DEFAULT]
%  - dir > 0 --> 'x' is processed block-wise along the specified direction
%
%  INPUTS
% ========
%  x     - ND array
%  gamma - positive, scalar or ND array compatible with the blocks of 'x'
%  dir   - integer, direction of block-wise processing

% default inputs
if nargin < 3 || (~isempty(dir) && dir == 0)
    dir = [];
end
% check input
sz = size(x); sz(dir) = 1;
if any( gamma(:) <= 0 ) || ~isscalar(gamma) && any(size(gamma) ~= sz)
    error('''gamma'' must be positive and either scalar or compatible with the blocks of ''x''')
end
% linearize
if isempty(dir)
    x = x(:); 
end
    
% evaluate the function
% xx = gamma .* sqrt( sum(x.^2, dir) );
xx = gamma .* vecnorm(x,2,dir);

p = sum( xx(:) );
  end

%--------------------------------------------------------------------------
function p = fun_L1(x, gamma, dir)
% default inputs
if nargin < 3 || (~isempty(dir) && dir == 0)
    dir = [];
end

% check input
sz = size(x); sz(dir) = 1;
if any( gamma(:) <= 0 ) || ~isscalar(gamma) && any(size(gamma) ~= sz)
    error('''gamma'' must be positive and either scalar or compatible with the blocks of ''x''')
end
%-----%
% linearize
if isempty(dir)
    x = x(:); 
end
   
% evaluate the function
% xx = gamma .* sqrt( sum(x.^2, dir) );
% xx = gamma .* vecnorm(x,1,dir);
% xx =  gamma .* sum(abs(x), dir);
% 
% p = sum( xx(:) );
p =  gamma .* sum(abs(x(:)));
  end 
 
%--------------------------------------------------------------------------  
function im0 = bound0(im,dir)
[Nx,Nz,~] = size(im); 
if dir==1
im(:,Nz,:) = 0; 
elseif dir == 2
im(Nx,:,:) = 0; 
elseif dir == -1 
im(1,:,:) = 0; 
elseif dir == -2 
im(:,1,:) = 0;     
end
im0 = im; 
end

%--------------------------------------------------------------------------
function p = indicator_box(x, low, high)
%function p = indicator_box(x, low, high)
%
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


% check the constraint
mask = low <= x & x <= high;

% evaluate the indicator function
if all(mask(:))
	p = 0;
else
	p = Inf;
end
end

%--------------------------------------------------------------------------
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
            case 'HS'
        kx = -0.5*[-1 1; -1 1];                                                 
        kz = -0.5*[-1 -1; 1 1]; 
            otherwise 
        printf('Unknown kernel type')
        end
end


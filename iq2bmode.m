function modeB = iq2bmode(IQ, increase)
%% rf2modeb Converts an IQ image into a mode B one.
%  modeB = iq2modeB(IQ, increase)
%
%         IQ: IQ image
%   increase: increase factor to adjust contrast


clear modeB


%% Initialization
if nargin==1
    increase=0;
elseif nargin<1 || nargin>2
    error('There must be 1 or 2 arguments.')
end


%% Computation
for i=1:size(IQ,3)
    modeB_temp=20*log(abs(IQ(:,:,i))+increase);
    modeB_temp=modeB_temp-min(modeB_temp(:));
    max_modeB=max(max(modeB_temp));
    modeB_temp = 255.*modeB_temp/max_modeB;
%     modeB_temp = uint8(255.*modeB_temp/max_modeB);
    modeB(:,:,i)=modeB_temp/255;
end


modeB(isnan(modeB)) =0 ;
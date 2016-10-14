%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% the function for superpixel-level propagation in FAST DEFOCUS MAP ESTIMATION
% ---------------------------------------------------------------------------
%   The Code is created based on the method described in the following paper 
%   [1] "Fast Defocus Map Extimation", 
%       Ding-Jie Chen, Hwann-Tzong Chen, and Long-Wen Chang
%       IEEE International Conference on Image Processing, 2016. 
%   The code and the algorithm are for non-comercial use only.
%  
%   Author: Ding-Jie Chen (dj_chen_tw@yahoo.com.tw)
%   Copyright 2016, National Tsing Hua University.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [SPDMap]=SPP(imgRGB,spMap,seedMap,sparseDMap)
 
sz    = size(imgRGB);
numSP = max(spMap(:));
beta  = 60;           % gaussian weight
alpha = 0.0001;       % parameter for propagation

% initial superpixel depth values
valSP=zeros(numSP,1);
for i=1:numSP
    idx=find(seedMap&(spMap==i));
    if numel(idx)<10, valSP(i)=-1; else val=sparseDMap(idx); valSP(i)=median(val); end
end

% pixels' and superpixels' features
imgYCbCr=colorspace('YCbCr<-', imgRGB); % YCbCr
labP=reshape(imgYCbCr,[],3); labSP=zeros(numSP,sz(3));
tmp=spMap(:);
for i=1:numSP, idx=tmp==i; labSP(i,:)=mean(labP(idx,:)); end
nodeVals=labSP;
    
% graph building     
[~,allPairs]=lattice(sz(1),sz(2),0);
tmp=allPairs(spMap(allPairs(:,1))~=spMap(allPairs(:,2)),:); % pairs inter SPs
pairSP=[spMap(tmp(:,1)),spMap(tmp(:,2))]; 
pairSP=sort(pairSP,2);
tmp=false(numSP,numSP);
tmp(numSP*(pairSP(:,2)-1)+pairSP(:,1))=true;
[SP1,SP2]=find(tmp); 
pairSP=[SP1, SP2];            
 
% make affinity matrix
weights=getGaussianWeights(double(pairSP),nodeVals,beta);
W=sparse([pairSP(:,1);pairSP(:,2)],[pairSP(:,2);pairSP(:,1)],[weights;weights],numSP,numSP);
D=sparse(1:numSP,1:numSP,sum(W)); 
A=alpha*(D-(1-alpha)*W)\eye(numSP);

% propagation
A(:,valSP==-1)=0;
A_hat=single(diag(sparse(1./sum(A')))*A);  % make row sum is 1
newValSP=A_hat*valSP;
depthMap=zeros(sz(1:2));
for i=1:numSP, depthMap(spMap==i)=newValSP(i); end
SPDMap = mat2gray(depthMap); % normalized to 0~1

end




%% auxiliary functions ----------------------------------------------------------------------------
function [weights]=getGaussianWeights(edges,nodeVals,featScale)
	%Compute intensity differences
	if featScale > 0
		featDist=sqrt(  sum( (nodeVals(edges(:,1),:)-nodeVals(edges(:,2),:)).^2 ,2 )  );
		featDist=normalize(featDist); %Normalize to [0,1]
	else
		featDist=zeros(size(edges,1),1);
		featScale=0;
	end
	%Compute Gaussian weights
	weights=exp(-(featScale*featDist))+eps;
end
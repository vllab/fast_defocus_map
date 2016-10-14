%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Demo program for FAST DEFOCUS MAP ESTIMATION
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
function demo()
clc; clear all; close all; warning off;
cd(fileparts(mfilename('fullpath'))); % pwd to current folder
addpath(genpath(fileparts(mfilename('fullpath'))));

% parameter setting
inPath   = '.\Input\';
outPath  = '.\Output\';
filelist = dir(fullfile(inPath,'*.png'));
numImg   = length(filelist);
maxHWset = [ 200 400 600 ];
std      = 1;  % parameter in defocusEst function (see Zhuo and Sim, PR 2011)

timeRecord = zeros(1,numImg);
timePlot   = zeros(1,numel(maxHWset));
for j=1:numel(maxHWset)
    maxHW = maxHWset(j);
    for i=1:numImg    
        disp([filelist(i).name '----------']);
        % read image ----------------------------------------------------
        img=imread(strcat(inPath,filelist(i).name));    
        sz=size(img);
        if max(sz)>maxHW, img=imresize(img,1/max(sz/maxHW)); else img=imresize(img,maxHW/max(sz)); end    
        sz=size(img); img=im2double(img);

        % defocus estimation --------------------------------------------
        tt=tic;
        [SPDMap]=defocusEst(img,std); 
        timeRecord(1,i)=toc(tt);
        disp(['Prop-Canny :         [' num2str(toc(tt)) ' s]']); fprintf('\n'); 
        imwrite(SPDMap,strcat(outPath,filelist(i).name(1:strfind(filelist(i).name,'.')-1),'_PC.png'));
    end
    save(['time' num2str(maxHW) '.mat'],'timeRecord');
end
for i=1:numel(maxHWset)
    load(['time' num2str(maxHWset(i)) '.mat']);
    timePlot(:,i)=median(timeRecord,2);
end
figure;
h=plot(timePlot','o-','DisplayName','timePlot');
legend('Our-C','Location','NorthWest');
set(h,'LineWidth',3)
xlabel('image size (maximum side length)')
ylabel('execution time (second)')
set(gca,'xtick',1:numel(maxHWset),'xticklabel',maxHWset)
end   
 
    
    
    
%% auxiliary functions ----------------------------------------------------------------------------
function [SPDMap] = defocusEst(imgC,std)
    imgG = rgb2gray(imgC);
    sz = size(imgG);

    % edge detection ------------------------------------------------
    tt=tic;
        edgeMap = edge(rgb2gray(imgC),'canny',0.1,1);
    disp(['   edgeDetection :    ' num2str(toc(tt)) ' s']);
 
    % over-segmentation ---------------------------------------------
    tt=tic;    
        imwrite(uint8(imgC*255), 'tmp.bmp'); % the slic software support only the '.bmp' image
        spNum = 200; % [200]
        spatialWeight = 20;  % [10] 1~30
        comm = ['External\SLIC\SLICSuperpixelSegmentation' ' ' 'tmp.bmp' ' ' int2str(spatialWeight) ' ' int2str(spNum) ' ' '.\'];
        system(comm);  fprintf(['\b']);
        spMap = double(ReadDAT(sz,'tmp.dat')); % superpixel label matrix
        spNum = max(spMap(:)); % the actual superpixel number       
    disp(['   oversegmentation : ' num2str(toc(tt)) ' s']);    
 
    % generate the sparse defocus map from the gaussian gradient magnitude ratio 
    % see Zhuo and Sim, PR 2011
    tt=tic;     
        std1 = std; std2 = 1.5*std;
        fx=makeGGMxd(std1); fy=fx'; gradx=filter2(fx,imgG,'same'); grady=filter2(fy,imgG,'same'); mg1=(gradx.^2+grady.^2).^.5;
        fx=makeGGMxd(std2); fy=fx'; gradx=filter2(fx,imgG,'same'); grady=filter2(fy,imgG,'same'); mg2=(gradx.^2+grady.^2).^.5;
        gRatio = mg1./mg2;
        sparseDMap = std2 ./ sqrt( (gRatio.^2-1) );
        sparseDMap(gRatio<=1) = 0; % prevent complex number
        sparseDMap(gRatio>(std2/std1)) = 0;  
        sparseDMap(~edgeMap) = 0; % select edge
    disp(['   sparseDefocusMap : ' num2str(toc(tt)) ' s']);

    % build SP-level defocus map ------------------------------------
    tt=tic;
        seedMap=sparseDMap>0.0001;
        [SPDMap] = SPP(imgC,spMap,seedMap,sparseDMap);
    disp(['   propagaration :    ' num2str(toc(tt)) ' s']); 
end

function [filter] = makeGGMxd(std) % Make Gaussian gradient magnitude (x direction)
    w      = 4*std; %(2*ceil(2* std1))+1;   
    [X,Y]  = meshgrid([-w:w]);
    var    = std.^2;
    filter = -(X./(2*pi*var.^2)) .* exp(-(X.^2 + Y.^2)./(2*var));
end
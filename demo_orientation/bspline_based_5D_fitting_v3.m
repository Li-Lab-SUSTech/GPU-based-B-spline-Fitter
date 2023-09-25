% Copyright (c) 2021 Li Lab, Southern University of Science and Technology, Shenzhen
% author: Yiming Li
% email: liym2019@sustech.edu.cn
% date: 2022.10.25
% Tested with CUDA 11.1 (Express installation) and Matlab 2021b
%%
clear
clc
close all
addpath(genpath('..\MultiD_bspline'))
addpath('.\store')
%% hyper parameters for PSF model used for fit
paraSim.NA = 1.43;                                                % numerical aperture of obj             
% paraSim.refmed = 1.35;                                            % refractive index of sample medium
paraSim.refmed = 1.33;
paraSim.refcov = 1.523;                                           % refractive index of converslip
paraSim.refimm = 1.518;                                           % refractive index of immersion oil
paraSim.lambda = 600;                                             % wavelength of emission
paraSim.zemit0 = 0;  % reference emitter z position, nm, distance of molecule to coverslip

% paraSim.zemit0 = -1*deltobjStage+2000;
paraSim.objStage0 = -1*paraSim.refimm/paraSim.refmed*(paraSim.zemit0);
% paraSim. pixelSizeX = 100;                                        % nm, pixel size of the image
% paraSim. pixelSizeY = 100;                                        % nm, pixel size of the image
paraSim. pixelSizeX =65;                                        % nm, pixel size of the image
paraSim. pixelSizeY = 65;                                        % nm, pixel size of the image

paraSim.Npupil = 64;                                             % sampling at the pupil plane

paraSim.aberrations = [2,0,0.0;2,-2,0.0; 2,2,0.0; 3,-1,0.0; 3,1,0.0; 4,0,0.0; 3,-3,0.0; 3,3,0.0; 4,-2,0.0; 4,2,0.00; 5,-1,0.0; 5,1,0.0; 6,0,0.0; 4,-4,0.0; 4,4,0.0;  5,-3,0.0; 5,3,0.0;  6,-2,0.0; 6,2,0.0; 7,1,0.0; 7,-1,0.00; 8,0,0.0];
% paraSim.aberrations(:,3) =  paraSim.aberrations(:,3)*paraSim.lambda;
% paraSim.aberrations(1:3,3)=(rand(1,3)-0.5)*50;
%random abberation
%% building model
paraSim.Nmol = 51;
paraSim.sizeX = 31;
paraSim.sizeY = 31;
paraSim.Npolar=36;          %Number of sampling in polar angle  delta polar=90/Npolar
paraSim.Nazim=36;           %Number of sampling in azimuthal angle  delta azim=360/Npolar
paraSim.WF='Vortex';        %Phase Mask type

g2=0.75;

Nmol=paraSim.Nmol;
paraSim.xemit = zeros(1,Nmol);                             %nm
%paraSim.xemit = 58.5*ones(1,Nmol);
paraSim.yemit = zeros(1,Nmol);                             %nm
paraSim.zemit = linspace(-500,500,Nmol)*1;                                      %nm
paraSim.objStage = linspace(-1000,1000,Nmol)*0;                                  %nm
%paraSim.objStage = deltobjStage*ones(1,Nmol);
    
[FreePSFs,FixPSFs,~] = vectorPSF_Final_dipor_v2(paraSim);

% FixPSFs=FixPSFs*Nphotons+bg;
% FreePSFs = FreePSFs*Nphotons+bg;
ss=sum(FixPSFs,[1 2]);
FixPSFs=FixPSFs/max(ss(:));
FixPSF_b=FixPSFs;

ByteSize(FixPSF_b);
disp('Converting 5D data ...')
tic
b5=bsarray(FixPSF_b);% it take few minutes
tb5=toc
ByteSize(b5);
b5.dx=paraSim. pixelSizeX;
b5.dy=paraSim. pixelSizeY;
b5.dz=paraSim.zemit(2)-paraSim.zemit(1);
b5.du=360/(paraSim.Nazim);
b5.dv=90/(paraSim.Npolar);

disp('Converting 3D data ...')
tic
b3=bsarray(FreePSFs);
tb3=toc
% disp('Done')
t1=g2;
t2=1-t1;
PSFall=[];
for ii=1:size(FixPSFs,4)
    for jj=1:size(FixPSFs,5)
        PSFall(:,:,:,ii,jj)=t2*FreePSFs+t1*squeeze(FixPSFs(:,:,:,ii,jj));
    end
end
% data = poissrnd(data,size(data));
%% CRLB calculating
addpath(genpath('E:\usr\LMF\5D\MultiD_bspline'))
dim=4; %3:z ,4:az ,5:polar, 6:g2
photon=5000;
bg=10;
g2=0.75;
sze=15; % roi size
[variable,CRLB53,PSFstack]=CRLB5n3(photon,bg,g2,b5,b3,dim,sze);
%% fitting and ploting
rep=200;
[STD,P,Ptmp,mP] =fitb5n3(PSFstack,b5,b3,sze,rep);
% close all
Plotb5n3(variable,STD,CRLB53,dim);





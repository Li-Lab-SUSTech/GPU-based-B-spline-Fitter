% Copyright (c) 2021 Li Lab, Southern University of Science and Technology, Shenzhen
% author: Yiming Li
% email: liym2019@sustech.edu.cn
% date: 2022.7.25
% Tested with CUDA 11.1 (Express installation) and Matlab 2019a
%%
addpath(genpath('./'))
addpath(genpath('..\MultiD_bspline'))
clear
close all
clc
%% hyper parameters for PSF model used for fit
paraSim.NA = 1.35;                                                % numerical aperture of obj
paraSim.refmed = 1.40;                                            % refractive index of sample medium
paraSim.refcov = 1.518;                                           % refractive index of converslip
paraSim.refimm = 1.40;                                           % refractive index of immersion oil
paraSim.lambda = 668;                                             % wavelength of emission
paraSim.objStage0_upper = -0;                                        % nm, initial objStage0 position,relative to focus at coverslip
paraSim.objStage0_lower = -0;                                        % nm, initial objStage0 position,relative to focus at coverslip
paraSim.zemit0_upper = -1*paraSim.refmed/paraSim.refimm*(paraSim.objStage0_upper);  % reference emitter z position, nm, distance of molecule to coverslip
paraSim.zemit0_lower = -1*paraSim.refmed/paraSim.refimm*(paraSim.objStage0_lower);  % reference emitter z position, nm, distance of molecule to coverslip


paraSim. pixelSizeX = 120;                                        % nm, pixel size of the image
paraSim. pixelSizeY = 120;                                        % nm, pixel size of the image
paraSim.Npupil = 64;                                             % sampling at the pupil plane

paraSim.aberrations(:,:,1) = [2,-2,0.0; 2,2,-0.1; 3,-1,0.0; 3,1,0.0; 4,0,0; 3,-3,0.0; 3,3,0.0; 4,-2,0.0; 4,2,0.00; 5,-1,0.0; 5,1,0.0; 6,0,0.0; 4,-4,0.0; 4,4,0.0;  5,-3,0.0; 5,3,0.0;  6,-2,0.0; 6,2,0.0; 7,1,0.0; 7,-1,0.00; 8,0,0.0];
paraSim.aberrations(:,:,2) = [2,-2,0.0; 2,2,0.1; 3,-1,0.0; 3,1,0.0; 4,0,0.0; 3,-3,0.0; 3,3,0.0; 4,-2,0.0; 4,2,0.00; 5,-1,0.0; 5,1,0.0; 6,0,0.0; 4,-4,0.0; 4,4,0.0;  5,-3,0.0; 5,3,0.0;  6,-2,0.0; 6,2,0.0; 7,1,0.0; 7,-1,0.00; 8,0,0.0];

paraSim.aberrations(:,3,:) =  paraSim.aberrations(:,3,:)*paraSim.lambda;

paraSim.offset = [0 0];
paraSim.phaseshift = [0 ,pi/2, pi, 3*pi/2];
%% parameters for molecules for simulation
Nmol = 101;
Npixels = 31;
% Nphotons = 5000 +10000*rand(1,Nmol);
% bg = 10 + 10*rand(1,Nmol);
paraSim.Nmol = Nmol;
paraSim.sizeX = Npixels;
paraSim.sizeY = Npixels;

paraSim.xemit = (-200+400*rand(1,Nmol))*0;                             %nm
paraSim.yemit = (-200+400*rand(1,Nmol))*0;                             %nm
paraSim.zemit = linspace(-1000,1000,Nmol)*1;                                      %nm
paraSim.objStage = linspace(-1000,1000,Nmol)*0;                                  %nm

[PSFs,PSFsUpper,PSFsLower,WaberrationUpper, WaberrationLower] = vectorPSF_4Pi(paraSim);

%% generate IAB & Bspline model
ipalm_im  = PSFs;

phaseshift = paraSim.phaseshift;
k = 2 * pi / paraSim.lambda; %lambda in nm
zcand = paraSim.zemit;% if move obj stage use paraSim.objStage
zstep = zcand(2) - zcand(1);

imsz = paraSim.sizeX;


I = squeeze((ipalm_im(:, :, :, 1) + ipalm_im(:, :, :, 3)) / 2);


kz2 = 2 * k * zcand';
kz2 = permute(repmat(kz2, 1, imsz, imsz), [2, 3, 1]);

F_phi1 = squeeze(ipalm_im(:, :, :, 1)) - I;
F_phi2 = squeeze(ipalm_im(:, :, :, 2)) - I;
phi1 = phaseshift(1);
phi2 = phaseshift(2);

A = (F_phi1 .* sin(kz2 + phi2) - F_phi2 .* sin(kz2 + phi1)) / sin(phi2 - phi1);
B = (-F_phi1 .* cos(kz2 + phi2) + F_phi2 .* cos(kz2 + phi1)) / sin(phi2 - phi1);


%check: restore PSF in the 4th quadrant
phi0 = 0;
deltaZ=10;  %nm  sampling rate in z
deltaPhi=k*deltaZ*2;  %sampling rate in phase,convered by delta z
Nphi=round(2*pi/deltaPhi);
phi=linspace(0,2*pi,Nphi);
deltaPhi=phi(2);
phi(end+1)=phi(end)+deltaPhi; % additional sampling
PSF_model=zeros([size(A) Nphi+1]);
for i=1:Nphi+1
    PSF_model(:,:,:,i) = I + A .* cos(phi(i) + phi0) + B .* sin(phi(i) + phi0);
end
tic
b4pi=bsarray(PSF_model);       %bspline method
tb=toc;
b4pi.dphidN=deltaPhi;
ByteSize(b4pi)
tic
Ispline = Spline3D_interp(I);  %cspline method
Aspline = Spline3D_interp(A);
Bspline = Spline3D_interp(B);
tc=toc;
PSF.Ispline = Ispline;
PSF.Aspline = Aspline;
PSF.Bspline = Bspline;

ByteSize(PSF)
%% generate test data

lambdanm = paraSim.lambda;
dz = zcand(2) - zcand(1);
z0 = round(length(zcand) / 2);
% k = 2 * pi * paraSim.refimm / lambdanm; %lambda in nm
NV = 6;


Nfits = 50;
Npixels = 13;
bg = [20 20 20 20];
phi0 = [0, pi/2, pi, 1.5 * pi];
N=1000;
Nz=20;
rep=1000;%repet times of each image
key=0;

ground_truth.x(:,1) = Npixels/2 -1 +2*rand([Nfits 1])*key;
ground_truth.y(:,1) = Npixels/2 -1 +2*rand([Nfits 1])*key;

ground_truth.N  = repmat(N,Nfits,4);
ground_truth.bg =repmat(bg, [Nfits 1]);
%      ground_truth.znm = -500+1000*rand([Nfits 1]);
ground_truth.znm = linspace(-500,500,Nfits)';%z step is 1000/Nfits,
ground_truth.zspline = ground_truth.znm / dz + z0;
ground_truth.phi =  wrapTo2Pi(2 * k * ground_truth.znm);

scale =0;
ground_truth.x(:,2)=ground_truth.x(:,1)+(rand(Nfits, 1))*scale;
ground_truth.y(:,2)=ground_truth.y(:,1)+(rand(Nfits, 1))*scale;

ground_truth.x(:,3)=ground_truth.x(:,1)+(rand(Nfits, 1))*scale;
ground_truth.y(:,3)=ground_truth.y(:,1)+(rand(Nfits, 1))*scale;

ground_truth.x(:,4)=ground_truth.x(:,1)+(rand(Nfits, 1))*scale;
ground_truth.y(:,4)=ground_truth.y(:,1)+(rand(Nfits, 1))*scale;

% coordinates for simulation
coordinates = zeros(Nfits,4,length(phi0));
for kk=1:1:length(phi0)
    coordinates(:,:,kk) = [ground_truth.x(:,kk) ground_truth.y(:,kk) ground_truth.zspline ground_truth.phi];
end

true_theta = zeros(Nfits,length(phi0),NV);
true_theta(:,:,1) = ground_truth.x;
true_theta(:,:,2) = ground_truth.y;
true_theta(:,:,3) = ground_truth.N;
true_theta(:,:,4) = ground_truth.bg;
true_theta(:,:,5) = repmat(ground_truth.zspline,1,4);
true_theta(:,:,6) = repmat(ground_truth.phi,1,4);
true_theta1=squeeze(true_theta(:,1,:));
true_theta1(:,7)=squeeze(true_theta(:,3,3));
true_theta =permute(true_theta,[3 2 1]);
shared=[1 1 1 1 1 1];
By_bsline=0;
% PSF generation 
if By_bsline
    [~, data] = simBSplinePSF(Npixels, b4pi, ground_truth.N, ground_truth.bg, coordinates, phi0);%simulate images by B-spline
else
    [~, data] = simSplinePSF(Npixels, PSF, ground_truth.N, ground_truth.bg, coordinates, phi0);%simulate images by conventional cubic spline
end

imstack=[];
for ii=1:Nfits
    imstack=cat(3,imstack,repmat(data(:,:,ii,:),1,1,rep,1));
end
imstack=poissrnd(imstack);

CRLB_YL=[];
for i=1:Nfits
    CRLB_YL =cat(1,CRLB_YL, calculate_CRLB_YL_shared(1, PSF, Npixels, phi0, true_theta(:,:,i),shared));
end
CRLBx0=CRLB_YL(:,1);
CRLBy0=CRLB_YL(:,2);
CRLBz0=CRLB_YL(:,5);
CRLBphi0=CRLB_YL(:,6);
%% bspline fitting 4pi

Nall=Nfits*rep;
numberchannel = 4;
phase_shiftb=phi0./b4pi.dphidN;
dTAll=zeros(Nall,NV,numberchannel);
zstart = [z0-0/dz];
dTAll=permute(dTAll,[2 3 1]);
% initPhase = [ pi/2 pi/2*3];
phi0A=repmat(phi0',[1 Nall]);
initZA=repmat(zstart(1),[1 Nall]);

initPhase=[1 1/2]*2*pi;
initPhaseA=repmat(initPhase(1),[1 Nall]);
initphi=initPhaseA./b4pi.dphidN;
[Pb,CRLB3, LogL]=GPUmleFit_LM_4Pi_bspline(single(imstack(:, :,:, :)),uint32(shared),100,single(b4pi.coeffs),...
    single(dTAll),single(initZA),single(initphi),single(phase_shiftb));
clear GPUmleFit_LM_4Pi_bspline;

if length(initPhase)>1
    for i=2:length(initPhase)
        initZA=repmat(zstart,[1 Nall]);
        initPhaseA=repmat(initPhase(i),[1 Nall]);
        initphi=initPhaseA./b4pi.dphidN;
        [Ph,CRLBh, LogLh3]=GPUmleFit_LM_4Pi_bspline(single(imstack(:, :,:, :)),uint32(shared),100,single(b4pi.coeffs),...
            single(dTAll),single(initZA),single(initphi),single(phase_shiftb));
        %     [Ph,CRLBh, LogLh3]=CPUmleFit_LM_4Pi_bspline_v2(single(imstack(:, :,:, :)),uint32(shared),100,single(b4pi.coeffs),...
        %         single(dTAll),single(phi0),single(initZA),single(initphi),single(phase_shift));
        indbetter=LogLh3-LogL>=1e-4; %copy only everything if LogLh increases by more than rounding error.
        Pb(indbetter,:)=Ph(indbetter,:);
        CRLB3(indbetter,:)=CRLBh(indbetter,:);
        LogL(indbetter)=LogLh3(indbetter);
        clear GPUmleFit_LM_4Pi_bspline;
    end
end
Pb(:,end-1)=(Pb(:,end-1)-1).*b4pi.dphidN;

X=Pb(:,2);
Y=Pb(:,1);
Z=Pb(:,5);
BG=Pb(:,4);
phib=Pb(:,6);
Znm = (Z-z0)*dz;
phib = wrapTo2Pi(phib);
zb_phi = z_from_phi_YL(Z, phib, k, z0, dz);
%% ploting result
ground_truthZ=ground_truth.znm;
% CRLB z
figure;subplot(1,2,1);scatter(1:size(Pb,1),Pb(:,end-1),5,'o','filled');box on
ylabel('\phi (rad)','FontSize',18,'LineWidth',1,'FontWeight','bold')
xlabel('frame');title('Interference phase')
set(gca,'fontsize',15,'fontweight','bold')
subplot(1,2,2);scatter(1:size(Pb,1),(Pb(:,end-2)-z0)*dz,5,'o','filled');box on
ylabel('Z (nm)','FontSize',15,'LineWidth',1,'FontWeight','bold')
xlabel('frame');title('Axial localization')
set(gca,'fontsize',15,'fontweight','bold')
set(gcf,'position',[500,200,1000,500])
Pb1=reshape(Pb,rep,Nall/rep,7);
stdphi=std(Pb1(:,:,end-1));

Zb_phi=reshape(zb_phi,rep,Nall/rep);
Zb_std=std(Zb_phi);
figure;scatter(1:size(zb_phi),zb_phi,10,'o','filled');title('Unwarp Z(nm)')
xlabel('frame');ylabel('z (nm)')
set(gca,'FontSize',15,'FontWeight','bold')
box on

figure;

hold on;
% legend('STD Z','CRLBz^{1/2}','FontWeight','bold','FontSize',15)
% xlabel('z(nm)','FontWeight','bold','FontSize',20);
% ylabel('z_\phi .loc.prec(nm) ','FontWeight','bold','FontSize',20);
% set(gca,'FontSize',15,'FontWeight','bold') 
box on
% CRLB x,y
CRLBX=sqrt(CRLBx0)*paraSim. pixelSizeX;
CRLBY=sqrt(CRLBy0)*paraSim. pixelSizeY;
X=reshape(X,rep,Nall/rep);
Y=reshape(Y,rep,Nall/rep);
Xb_std=std(X)*paraSim. pixelSizeX;
Yb_std=std(Y)*paraSim. pixelSizeY;

% figure;
plot(ground_truthZ,[CRLBX CRLBY],'LineWidth',3)
p1=plot(ground_truthZ,sqrt(CRLBphi0)./(k*2),'Color','#77AC30','LineWidth',3);

xlabel('z(nm)');
ylabel('X/Y/Z.loc.prec(nm) ');
scatter(ground_truthZ,Xb_std,'filled','blue','MarkerEdgeColor','black','MarkerEdgeAlpha',0.4)
scatter(ground_truthZ,Yb_std,'filled','red','MarkerEdgeColor','black','MarkerEdgeAlpha',0.8)
sc=scatter(ground_truthZ,Zb_std,'filled','green','MarkerEdgeColor','black','MarkerEdgeAlpha',0.8);

set(gca,'FontSize',15,'FontWeight','bold') 
legend({'CRLBx^{1/2}','CRLBy^{1/2}','CRLBz^{1/2}','STDx','STDy','STD Z'},'Location','North')
ylim([0 10]);



function [PSFs,FixedPSF,Waberration] = vectorPSF_Final_dipor_v2(parameters)
% parameters: NA, refractive indices of medium, cover slip, immersion fluid,
% wavelength (in nm), sampling in pupil
%optics para
% add norm function comparecd to v3 #need test
% add pixelX Y 

NA = parameters.NA;
refmed = parameters.refmed;
refcov = parameters.refcov;
refimm = parameters.refimm;
lambda = parameters.lambda;

%image para
xemit = parameters.xemit;
yemit = parameters.yemit;
zemit = parameters.zemit;
objStage=parameters.objStage;
zemit0 = parameters.zemit0;

objStage0 = parameters.objStage0;
Npupil = parameters.Npupil;
sizeX = parameters.sizeX;
sizeY = parameters.sizeY;
sizeZ = parameters.Nmol;
pixelSizeX = parameters.pixelSizeX;
pixelSizeY = parameters.pixelSizeY;
xrange = pixelSizeX*sizeX/2;
yrange = pixelSizeY*sizeY/2;
Npolar=parameters.Npolar;
Nazim=parameters.Nazim;
% pupil radius (in diffraction units) and pupil coordinate sampling
PupilSize = 1.0;
DxyPupil = 2*PupilSize/Npupil;
XYPupil = -PupilSize+DxyPupil/2:DxyPupil:PupilSize;
[YPupil,XPupil] = meshgrid(XYPupil,XYPupil);

% if exist('parameters.WF')
%     WaveFront=parameters.WF;
% else
%     WaveFront=zeros(size(YPupil));
% end
switch parameters.WF
    case 'Vortex'
        disp('Adding Vortex phase mask')
        phi = atan2(YPupil,XPupil);
        WaveFront=phi/(2*pi)*lambda;
        WaveFront=WaveFront-min(WaveFront(:));
    otherwise
        WaveFront=zeros(size(YPupil));
end

% calculation of relevant Fresnel-coefficients for the interfaces
% between the medium and the cover slip and between the cover slip
% and the immersion fluid
CosThetaMed = sqrt(1-(XPupil.^2+YPupil.^2)*NA^2/refmed^2);
CosThetaCov = sqrt(1-(XPupil.^2+YPupil.^2)*NA^2/refcov^2);
CosThetaImm = sqrt(1-(XPupil.^2+YPupil.^2)*NA^2/refimm^2);
%need check again, compared to original equation, FresnelPmedcov is
%multipiled by refmed
FresnelPmedcov = 2*refmed*CosThetaMed./(refmed*CosThetaCov+refcov*CosThetaMed);
FresnelSmedcov = 2*refmed*CosThetaMed./(refmed*CosThetaMed+refcov*CosThetaCov);
FresnelPcovimm = 2*refcov*CosThetaCov./(refcov*CosThetaImm+refimm*CosThetaCov);
FresnelScovimm = 2*refcov*CosThetaCov./(refcov*CosThetaCov+refimm*CosThetaImm);
FresnelP = FresnelPmedcov.*FresnelPcovimm;
FresnelS = FresnelSmedcov.*FresnelScovimm;

% Apoidization for sine condition
apoid = sqrt(CosThetaImm)./CosThetaMed;
% definition aperture
ApertureMask = double((XPupil.^2+YPupil.^2)<1.0);
Amplitude = ApertureMask.*apoid;

% setting of vectorial functions
Phi = atan2(YPupil,XPupil);
CosPhi = cos(Phi);
SinPhi = sin(Phi);
CosTheta = sqrt(1-(XPupil.^2+YPupil.^2)*NA^2/refmed^2);
SinTheta = sqrt(1-CosTheta.^2);

pvec{1} = FresnelP.*CosTheta.*CosPhi;
pvec{2} = FresnelP.*CosTheta.*SinPhi;
pvec{3} = -FresnelP.*SinTheta;
svec{1} = -FresnelS.*SinPhi;
svec{2} = FresnelS.*CosPhi;
svec{3} = 0;

PolarizationVector = cell(2,3);
for jtel = 1:3
  PolarizationVector{1,jtel} = CosPhi.*pvec{jtel}-SinPhi.*svec{jtel};
  PolarizationVector{2,jtel} = SinPhi.*pvec{jtel}+CosPhi.*svec{jtel};
end

wavevector = cell(1,3);
wavevector{1} = (2*pi*NA/lambda)*XPupil;
wavevector{2} = (2*pi*NA/lambda)*YPupil;
wavevector{3} = (2*pi*refimm/lambda)*CosThetaImm;
wavevectorzmed = (2*pi*refmed/lambda)*CosThetaMed; 

% calculation aberration function
Waberration = zeros(size(XPupil));
orders = parameters.aberrations(:,1:2);
zernikecoefs = squeeze(parameters.aberrations(:,3));
normfac = sqrt(2*(orders(:,1)+1)./(1+double(orders(:,2)==0)));
zernikecoefs = normfac.*zernikecoefs;
allzernikes = get_zernikefunctions(orders,XPupil,YPupil);
for j = 1:numel(zernikecoefs)
  Waberration = Waberration+zernikecoefs(j)*squeeze(allzernikes(j,:,:));  
end

PhaseFactor = exp(2*pi*1i*(Waberration+WaveFront)/lambda);

PupilMatrix = cell(2,3);
for itel = 1:2
  for jtel = 1:3
    PupilMatrix{itel,jtel} = Amplitude.*PhaseFactor.*PolarizationVector{itel,jtel};
  end
end

% pupil and image size (in diffraction units)
% PupilSize = NA/lambda;
ImageSizex = xrange*NA/lambda;
ImageSizey = yrange*NA/lambda;

% calculate auxiliary vectors for chirpz
[Ax,Bx,Dx] = prechirpz(PupilSize,ImageSizex,Npupil,sizeX);
[Ay,By,Dy] = prechirpz(PupilSize,ImageSizey,Npupil,sizeY);

FieldMatrix = cell(2,3,sizeZ);

for jz = 1:sizeZ
    % xyz induced phase contribution
    if zemit(jz)+zemit0>=0
        Wxyz= (-1*xemit(jz))*wavevector{1}+(-1*yemit(jz))*wavevector{2}+(zemit(jz)+zemit0)*wavevectorzmed;
        %   zemitrun = objStage(jz);
        PositionPhaseMask = exp(1i*(Wxyz+(objStage(jz)+objStage0)*wavevector{3}));
    else
        Wxyz= (-1*xemit(jz))*wavevector{1}+(-1*yemit(jz))*wavevector{2};
        %   zemitrun = objStage(jz);
        PositionPhaseMask = exp(1i*(Wxyz+(objStage(jz)+objStage0+zemit(jz)+zemit0)*wavevector{3}));
    end
  
  for itel = 1:2
    for jtel = 1:3
      
      % pupil functions and FT to matrix elements
      PupilFunction =PositionPhaseMask.*PupilMatrix{itel,jtel};
      IntermediateImage = transpose(cztfunc(PupilFunction,Ay,By,Dy));
      FieldMatrix{itel,jtel,jz} = transpose(cztfunc(IntermediateImage,Ax,Bx,Dx));
    end
  end
end

%calculates the free dipole PSF given the field matrix.
PSFs = zeros(sizeX,sizeY,sizeZ);
for jz = 1:sizeZ
    for jtel = 1:3 %new
        for itel = 1:2
            PSFs(:,:,jz) = PSFs(:,:,jz) + (1/3)*abs(FieldMatrix{itel,jtel,jz}).^2;
        end
    end
end
% %%
% azumuthal acc is about 10 degree, polar should be 5 degree
% so azumuthal is 360/10=36, polar is 180/5=36
extend=3;
Azim=linspace(0,2*pi,Nazim+1);
Azim=[Azim(end-extend:end-1)-2*pi,Azim,2*pi+Azim(2:4)];
Pola=linspace(0,pi/2,Npolar+1);
Pola=[-(pi/2-Pola(end-extend:end-1)),Pola,pi/2+Pola(2:4)];

FixedPSF = zeros(sizeX,sizeY,sizeZ,size(Azim,2),size(Pola,2));
for j=1:size(FixedPSF,5)   %pola
    for i=1:size(FixedPSF,4)  %azim
        pola = Pola(j);
        azim = Azim(i);
        dipor(1) = sin(pola)*cos(azim);
        dipor(2) = sin(pola)*sin(azim);
        dipor(3) = cos(pola);
        
        % calculation of fixed PSF
        for jz =1:sizeZ
            Ex = dipor(1)*FieldMatrix{1,1,jz}+dipor(2)*FieldMatrix{1,2,jz}+dipor(3)*FieldMatrix{1,3,jz};
            Ey = dipor(1)*FieldMatrix{2,1,jz}+dipor(2)*FieldMatrix{2,2,jz}+dipor(3)*FieldMatrix{2,3,jz};
            FixedPSF(:,:,jz,i,j) = abs(Ex).^2+abs(Ey).^2;
            FixedPSF(:,:,jz,i,j) = FixedPSF(:,:,jz,i,j)/sum(FixedPSF(:,:,jz,i,j),'all');
        end
    end
end
% figure;
% image(FixedPSF(:,:,11),'CDataMapping', 'scaled');colormap jet,view(2);axis square
% calculate intensity normalization function using the PSFs at focus
% position without any aberration. It might not work when sizex and sizeY
% are very small
FieldMatrix = cell(2,3);
for itel = 1:2
  for jtel = 1:3
    PupilFunction = Amplitude.*PolarizationVector{itel,jtel};
    IntermediateImage = transpose(cztfunc(PupilFunction,Ay,By,Dy));
    FieldMatrix{itel,jtel} = transpose(cztfunc(IntermediateImage,Ax,Bx,Dx));
  end
end

intFocus = zeros(sizeX,sizeY);
for jtel = 1:3
    for itel = 1:2
        intFocus = intFocus + (1/3)*abs(FieldMatrix{itel,jtel}).^2;
    end
end

normIntensity = sum(intFocus(:));

PSFs = PSFs./normIntensity;
FixedPSF=FixedPSF./normIntensity;

% sigma = 0.5;
% if sigma ~=0
%     h = fspecial('gaussian',5,sigma);
%     PSFs = convn(PSFs,h,'same');
% end

hx = fspecial('gaussian',5,0.6);
hy = fspecial('gaussian',5,0.5);
hxs = sum(hx,1);
hys = sum(hy,1);

hgauss = hys'*hxs;
hgauss = hgauss/(sum(hgauss(:)));
FixedPSF= convn(FixedPSF,hgauss,'same');
PSFs = convn(PSFs,hgauss,'same');







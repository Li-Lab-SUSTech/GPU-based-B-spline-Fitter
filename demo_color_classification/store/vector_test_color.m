function modlepsf = vector_test_color(color,Npixels,Nmol,delt,PSFtype,pixelsz,xyrand)

%% hyper parameters for PSF model used for fit
if ~exist("PSFtype")
    PSFtype=[];
end
if ~exist("xyrand")
    xyrand=false;
end

if ~exist("pixelsz")
    pixelsz=[110 110];
end

depth = 00;

paraSim.NA = 1.5;                                                % numerical aperture of obj             
paraSim.refmed = 1.406;                                          % refractive index of sample medium
% paraSim.refmed = 1.5;
paraSim.refcov = 1.525;                                           % refractive index of converslip
paraSim.refimm = 1.518;                                           % refractive index of immersion oil
paraSim.lambda = color;                                             % wavelength of emission
paraSim.zemit0 = 1000;  % reference emitter z position, nm, distance of molecule to coverslip
paraSim.objStage0 = -1.1*paraSim.refimm/paraSim.refmed*(paraSim.zemit0);       % nm, initial objStage0 position,relative to focus at coverslip

% paraSim.zemit0 = -1*deltobjStage+2000;
% paraSim.objStage0 = -1*paraSim.refimm/paraSim.refmed*(paraSim.zemit0);
paraSim. pixelSizeX = pixelsz(1);                                        % nm, pixel size of the image
paraSim. pixelSizeY = pixelsz(2);                                        % nm, pixel size of the image
paraSim.Npupil = 64;                                             % sampling at the pupil plane

switch PSFtype
    case 'DMO'
        paraSim.aberrations = [2,-2,80.94; 2,2,-0.0; 3,-1,0.0; 3,1,0.0; 4,0,0.0; 3,-3,0.0; 3,3,0.0; 4,-2,-71.54; 4,2,0.00; 5,-1,0.0; 5,1,0.0; 6,0,0.0; 4,-4,0.0; 4,4,0.0;  5,-3,0.0; 5,3,0.0;  6,-2,10.09; 6,2,0.0; 7,1,0.0; 7,-1,0.00; 8,0,0.0];
    case 'Ast'
        %using cylindrical lens,magnitude of abberation will depend on it's
        %wave length
        paraSim.aberrations = [2,-2,0.0; 2,2,80; 3,-1,0.0; 3,1,0.0; 4,0,0.0; 3,-3,0.0; 3,3,0.0; 4,-2,0.0; 4,2,0.0; 5,-1,0.0; 5,1,0.0; 6,0,0.0; 4,-4,0.0; 4,4,0.0;  5,-3,0.0; 5,3,0.0;  6,-2,0.0; 6,2,0.0; 7,1,0.0; 7,-1,0.00; 8,0,0.0];
%         paraSim.aberrations(:,3) =  paraSim.aberrations(:,3)*paraSim.lambda;

    otherwise
        paraSim.aberrations = [2,-2,0.0; 2,2,-0.0; 3,-1,0.0; 3,1,0.0; 4,0,0.0; 3,-3,0.0; 3,3,0.0; 4,-2,0.0; 4,2,0.0; 5,-1,0.0; 5,1,0.0; 6,0,0.0; 4,-4,0.0; 4,4,0.0;  5,-3,0.0; 5,3,0.0;  6,-2,0.0; 6,2,0.0; 7,1,0.0; 7,-1,0.00; 8,0,0.0];

end



%% parameters for molecules for simulation
% Nmol = 50;
% Npixels = 31;
%Nphotons = 5000 +10000*rand(1,Nmol);
zshift = 250;
Nphotons =1;
% bg = 10 + 10*rand(1,Nmol);
bg = 0;
paraSim.Nmol = Nmol;
paraSim.sizeX = Npixels;
paraSim.sizeY = Npixels;

paraSim.xemit = xyrand*(100*rand(1,Nmol)-50);                               %nm
paraSim.yemit = xyrand*(100*rand(1,Nmol)-50);                               %nm
paraSim.zemit = linspace(-1000,1000,Nmol);                                     %nm
paraSim.objStage = linspace(-1000+delt,1000+delt,Nmol)*0;                                  %nm
%paraSim.objStage = deltobjStage*ones(1,Nmol);
    
[PSFs,Waberration] = vectorPSF_Final(paraSim);


data = PSFs*Nphotons+bg;

modlepsf=data;
% data = poissrnd(data,size(data));

%%
% for i=1:10
%     figure;mesh(modlepsf(:,:,i));
% end



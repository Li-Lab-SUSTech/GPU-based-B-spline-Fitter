%Copyright (c)2017 Ries Lab, European Molecular Biology Laboratory,
%  Heidelberg.
%
%  This file is part of GPUmleFit_LM Fitter.
%
%  GPUmleFit_LM Fitter is free software: you can redistribute it and/or modify
%  it under the terms of the GNU General Public License as published by
%  the Free Software Foundation, either version 3 of the License, or
%  (at your option) any later version.
%
%  GPUmleFit_LM Fitter is distributed in the hope that it will be useful,
%  but WITHOUT ANY WARRANTY; without even the implied warranty of
%  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%  GNU General Public License for more details.
%
%  You should have received a copy of the GNU General Public License
%  along with GPUmleFit_LM Fitter.  If not, see <http://www.gnu.org/licenses/>.
%
%
%  Additional permission under GNU GPL version 3 section 7

function simplefitter_cspline(p)
%parameters:
% p.imagefile: fiename of data (char);
% p.calfile: filename of calibration data (char);
% p.offset=ADU offset of data;
% p.conversion=conversion e-/ADU;
% p.preview: true if preview mode (fit only current image and display
% results).
% p.previewframe=frame to preview;
% p.peakfilter=filtersize (sigma, Gaussian filter) for peak finding;
% p.peakcutoff=cutoff for peak finding
% p.roifit=size of the ROI in pixels
% p.bidirectional= use bi-directional fitting for 2D data
% p.mirror=mirror images if bead calibration was taken without EM gain
% p.status=handle to a GUI object to display the status;
% p.outputfile=file to write the localization table to;
% p.outputformat=Format of file;
% p.pixelsize=pixel size in nm;

% p.loader which loader to use
% p.mij if loader is fiji: this is the fiji handle
% p.isscmos scmos camera used
% p.scmosfile file containgn scmos varmap


global simplefitter_stop

fittime=0;
if p.fiveD
    fitsperblock=5000;
else
    fitsperblock=20000;
end
imstack=zeros(p.roifit,p.roifit,fitsperblock,'single');
peakcoordinates=zeros(fitsperblock,3);
indstack=0;
resultsind=1;
imstack_all=[];
% bgmode='wavelet';
if contains(p.backgroundmode,'avelet')
    bgmode=2;
elseif contains(p.backgroundmode,'aussian')
    bgmode=1;
elseif contains(p.backgroundmode,'Conv PSF')
    bgmode=3;
else
    bgmode=0;
end

%scmos
varmap=[];
if p.isscmos
    varstack=ones(p.roifit,p.roifit,fitsperblock,'single');
    [~,~,ext]=fileparts(p.scmosfile);
    switch ext
        case '.tif'
            varmap=imread(p.scmosfile);
        case '.mat'
            varmap=load(p.scmosfile);
            if isstruct(varmap)
                fn=fieldnames(varmap);
                varmap=varmap.(fn{1});
            end
        otherwise
            errordlg('could not load variance map. No sCMOS noise model used.')
            p.isscmos=false;
            varstack=0;
    end
else
    varstack=0;
end
varmap=varmap*p.conversion^2;
%results
% frame, x,y,z,phot,bg, errx,erry, errz,errphot, errbg,logLikelihood
%load calibration
if exist(p.calfile,'file')
    cal=load(p.calfile);
    if p.fourD
        p.dz=cal.bspline.dz;  %coordinate system of spline PSF is corner based and in units pixels / planes
        p.z0=cal.bspline.z0;
        p.coeff=cal.bspline.coeff;
        p.isspline=true;
        try
            cal.PSF=cal.bspline.PSF;
        catch
            warning('Here is not cotain PSF');
        end
        
    elseif p.fiveD
        p.dz=cal.bspline.dz;  %coordinate system of spline PSF is corner based and in units pixels / planes
        p.z0=cal.bspline.z0;
        p.du=cal.bspline.du;
        p.dv=cal.bspline.dv;
        p.coeff5=cal.bspline.coeff5;
        p.coeff3=cal.bspline.coeff3;
        p.isspline=true;
        
    else
        p.dz=cal.cspline.dz;  %coordinate system of spline PSF is corner based and in units pixels / planes
        p.z0=cal.cspline.z0;
        p.coeff=cal.cspline.coeff;
        p.isspline=true;
        try
            cal.PSF=cal.cspline.PSF;
        catch
            warning('Here is not cotain PSF');
        end
    end
else
    %     errordlg('please select 3D calibration file')
    warndlg('3D calibration file could not be loaded. Using Gaussian fitter instead.','Using Gaussian fit','replace');
    p.isspline=false;
end

p.dx=floor(p.roifit/2);
% readerome=bfGetReader(p.imagefile);
p.status.String=['Open tiff file' ]; drawnow
filenumb=0;
% for i=1:10
%     if exist([p.imagefile(1:end-8),'_',num2str(i),p.imagefile(end-7:end)])~=0
%         filenumb=filenumb+1;
%     else
%         break;
%     end
% end

switch p.loader
    case 1
        reader=mytiffreader(p.imagefile);
        numframes(1)=reader.info.numberOfFrames;
        for i=1:11
            tmp=[p.imagefile(1:end-8),'_',num2str(i),p.imagefile(end-7:end)];
            tmp1=[p.imagefile(1:end-5),num2str(i+1),p.imagefile(end-3:end)];
            if exist(tmp)
                %                  eval(['reader',num2str(i),'=mytiffreader(',tmp,');']);
                reader(i+1)=mytiffreader(tmp);
                %                  eval(['numframes',num2str(i),'=reader.info.numberOfFrames;'])
                numframes(i+1)=reader(1+i).info.numberOfFrames;
            elseif exist(tmp1)
                reader(i+1)=mytiffreader(tmp1);
                %                  eval(['numframes',num2str(i),'=reader.info.numberOfFrames;'])
                numframes(i+1)=reader(1+i).info.numberOfFrames;
                
            else
                break;
            end
        end
    case 2
        reader=bfGetReader(p.imagefile);
        numframes=reader.getImageCount;
    case 3 %fiji
        ij=p.mij.imagej;
        ijframes=ij.getFrames;
        for k=1:length(ijframes)
            if strcmp(ijframes(k).class,'ij.gui.StackWindow')&&~isempty(ijframes(k).getImagePlus)
                reader=ijframes(k).getImagePlus.getStack;
                break
            end
        end
        if ~exist('reader','var')
            p.status.String='Error... Check if image is loaded in ImageJ'; drawnow
            return
        end
        %           numframes=reader.size;
        numframes=reader.getSize;
end


if p.preview
    frames=min(p.previewframe,sum(numframes(:)));
else
    frames=1:sum(numframes(:));
end

%loop over frames, do filtering/peakfinding
hgauss=fspecial('gaussian',max(3,ceil(3*p.peakfilter+1)),p.peakfilter);
rsize=max(ceil(6*p.peakfilter+1),3);
if p.conv
    try
        rsize=p.roifit;
        rsz=floor((size(cal.PSF,1)-rsize)/2):floor((size(cal.PSF,1)+rsize)/2);
        %     hdog=2*cal.PSF(rsz,rsz,floor(p.z0-40))/max(cal.PSF(rsz,rsz,floor(p.z0-40)),[],'all')+cal.PSF(rsz,rsz,floor(p.z0))/max(cal.PSF(rsz,rsz,floor(p.z0)),[],'all')+2*cal.PSF(rsz,rsz,floor(p.z0+40))/max(cal.PSF(rsz,rsz,floor(p.z0+40)),[],'all');
        hdog=mean(cal.PSF(rsz,rsz,:),3);
        hdog=hdog-mean(hdog(:));
        hdog=0.05*hdog/max(hdog(:));
    catch
        error('calibrate file not contain PSF model');
    end
else
    hdog=fspecial('gaussian',rsize,p.peakfilter)-fspecial('gaussian',rsize,max(1,2.5*p.peakfilter));
end
tshow=tic;
for F=frames
    if F<=numframes(1)
        image=getimage(F,reader(1),p);
    else
        for j=2:i
            if sum(numframes(1:j-1))<F && sum(numframes(1:j))>=F
                image=getimage(F-sum(numframes(1:j-1)),reader(j),p);
            end
        end
    end
    
    sim=size(image);
    imphot=(single(image)-p.offset)*p.conversion;
    
    %background determination
    if 0 %bgmode==3% wavelet
        %         bg=mywaveletfilter(imphot,3,false,true);
        %
        %         impf=filter2(hgauss,(imphot)-(bg));
    elseif bgmode==3       %add 0507,old version conv disabled
        imtmp=zeros([size(imphot),size(cal.PSF,3)]);
        for ii=1:size(cal.PSF,3)
            hdog=5*cal.PSF(:,:,ii)-mean(5*cal.PSF(:,:,ii));
            imtmp(:,:,ii)=filter2(hdog,(imphot- min(imphot(:,1))));
        end
        impf=mean(imtmp,3);
    elseif bgmode==1
        %         impf=filter2(hdog,sqrt(imphot));
        %          impf=filter2(hdog,(imphot));
        impf=filter2(hdog,(imphot- min(imphot(:,1))));
    elseif bgmode==0
        impf=filter2(hgauss,(imphot));
        
    end
    maxima=maximumfindcall(impf);
    indmgood=maxima(:,3)>(p.peakcutoff);
    indmgood=indmgood&maxima(:,1)>p.dx &maxima(:,1)<=sim(2)-p.dx;
    indmgood=indmgood&maxima(:,2)>p.dx &maxima(:,2)<=sim(1)-p.dx;
    maxgood=maxima(indmgood,:);
    if p.agg
        [maxgood,~]=aggre(maxgood,p);
    end
    %     ind=1;
    %     D=zeros(size(maxgood,1)-1);
    %     for ii=1:size(maxgood,1)-1
    %         for jj=ii:size(maxgood,1)
    %         D(ii,jj)=sqrt(maxgood(ii,1)-maxgood(jj,1))^2+(maxgood(ii,2)-maxgood(jj,2))^2);
    %         end
    %     end
    %     while(1)
    %         if ind>size(maxgood,1)-1
    %             break;
    %         end
    %         for ind1=ind+1:size(maxgood,1)
    %             if (abs((maxgood(ind,1)-maxgood(ind1,1)))<3*p.roifit/2 &&abs((maxgood(ind,2)-maxgood(ind1,2)))<3*p.roifit/2)
    %                 maxgood(ind,:)=round((maxgood(ind,:)+maxgood(ind+1,:))/2);
    %                 maxgood(ind1,:)=[];
    %                 ind=ind-1;
    %             end
    %         end
    %         ind=ind+1;
    %     end
    if p.preview && size(maxgood,1)>2000
        p.status.String=('increase cutoff');
        return
    elseif p.preview && size(maxgood,1)==0
        p.status.String=('No localizations found, decrease cutoff');
        return
    end
    
    %cut out images
    if p.conv
        indout=[];
        for k=1:size(maxgood,1)
            if maxgood(k,1)<size(cal.PSF,1)/2||maxgood(k,1)>(sim(1)-size(cal.PSF,1)/2)||...
                    maxgood(k,2)<size(cal.PSF,1)/2||maxgood(k,2)>(sim(1)-size(cal.PSF,1)/2)
                indout=cat(1,indout,k);
            end
        end
        maxgood(indout,:)=[];
    end
    for k=1:size(maxgood,1)
        if maxgood(k,1)>p.dx && maxgood(k,2)>p.dx && maxgood(k,1)<= sim(2)-p.dx && maxgood(k,2)<=sim(1)-p.dx
            indstack=indstack+1;
            if p.mirror
                imstack(:,:,indstack)=imphot(maxgood(k,2)-p.dx:maxgood(k,2)+p.dx,maxgood(k,1)+p.dx:-1:maxgood(k,1)-p.dx);
            else
                imstack(:,:,indstack)=imphot(maxgood(k,2)-p.dx:maxgood(k,2)+p.dx,maxgood(k,1)-p.dx:maxgood(k,1)+p.dx);
            end
            if p.isscmos
                varstack(:,:,indstack)=varmap(maxgood(k,2)-p.dx:maxgood(k,2)+p.dx,maxgood(k,1)-p.dx:maxgood(k,1)+p.dx);
            end
            peakcoordinates(indstack,1:2)=maxgood(k,1:2);
            peakcoordinates(indstack,3)=F;
            
            if indstack==fitsperblock
                p.status.String=['Fitting frame ' num2str(F) ' of ' num2str(frames(end)) ', Fitting...' ]; drawnow
                t=tic;
                resultsh=fitspline(imstack,peakcoordinates,p,varstack);
                fittime=fittime+toc(t);
                
                results(resultsind:resultsind+fitsperblock-1,:)=resultsh;
                resultsind=resultsind+fitsperblock;
                
                indstack=0;
                imstack_all=cat(3,imstack_all,imstack);
            end
        end
        
    end
    if toc(tshow)>1
        tshow=tic;
        p.status.String=['Loading frame ' num2str(F) ' of ' num2str(frames(end))]; drawnow
    end
    if  simplefitter_stop
        break
    end
end

for j=1:i
    closereader(reader(j),p);
end
p.status.String=['Fitting last stack...' ]; drawnow
if indstack<1
    p.status.String=['No localizations found. Increase cutoff?' ]; drawnow
else
    
    t=tic;
    if p.isscmos
        varh=varstack(:,:,1:indstack);
    else
        varh=0;
    end
    resultsh=fitspline(imstack(:,:,1:indstack),peakcoordinates(1:indstack,:),p,varh); %fit all the rest
    fittime=fittime+toc(t);
    %     imstack_all=cat(3,imstack_all,imstack(:,:,1:indstack));
    %save(['H:\test\result' datestr(now,'mmdd') '.mat'],'imstack_all');
    
    results(resultsind:resultsind+indstack-1,:)=resultsh;
end
if p.preview
%     f1=figure(201);
%      %     imagesc(impf.^2);
%     imagesc(impf);
%     title('Filtered image')
%     f1.Position=[180   358   560   420];
%     colorbar
%     hold on
%     plot(maxgood(:,1),maxgood(:,2),'wo')
%     for ii=1:size(maxgood,1)
%         rectangle('Position',[maxgood(ii,1)-(p.roifit/2),maxgood(ii,2)-(p.roifit/2),(p.roifit),(p.roifit)],'EdgeColor','r');
%     end
%     
%     plot(results(:,2),results(:,3),'k+')
%     
    
    f2=figure(202);
    imagesc(imphot);
%     f2.Position=[780   358   560   420];
    
    for ii=1:size(maxgood,1)
        rectangle('Position',[maxgood(ii,1)-(p.roifit/2),maxgood(ii,2)-(p.roifit/2),(p.roifit),(p.roifit)],'EdgeColor','r');
    end
    colorbar
    hold on
    title('Raw image');
    if p.fourD
        for ii=1:size(results,1)
            if results(ii,5)<2.5
                plot(results(ii,2),results(ii,3),'g+','LineWidth',1.2)
            else
                plot(results(ii,2),results(ii,3),'r+','LineWidth',1.2)
            end
        end
    else
        plot(results(:,2),results(:,3),'k+')
    end
    %     plot(maxgood(:,1),maxgood(:,2),'wo')
    %     plot(results(:,2),results(:,3),'k+')
    if p.fourD&&(size(results,1)>100)
        for jj=1:size(results,1)
            disp([num2str(results(jj,5)),' ',num2str(results(jj,6))]);
        end
    end
    hold off
    
    p.status.String=['Preview done. ' num2str(size(results,1)/fittime,'%3.0f') ' fits/s. ' num2str(size(results,1),'%3.0f') ' localizations.','F:',num2str(F),'/',num2str(numframes)]; drawnow
else
    p.status.String=['Fitting done. ' num2str(size(results,1)/fittime,'%3.0f') ' fits/s. ' num2str(size(results,1),'%3.0f') ' localizations. Saving now.']; drawnow
    
    if p.fiveD
        results(:,[20,22]+3*p.fourD)=results(:,[2, 10])*p.pixelsize(1);
        results(:,[21,23]+3*p.fourD)=results(:,[3,11])*p.pixelsize(end);
    elseif p.fourD
        results(:,[20,21])=results(:,[16,17]);
        results(:,[13,15]+3*p.fourD)=results(:,[2, 7+p.fourD])*p.pixelsize(1);
        results(:,[14,16]+3*p.fourD)=results(:,[3, 8+p.fourD])*p.pixelsize(end);
    else
        results(:,[13,15]+3*p.fourD)=results(:,[2, 7+p.fourD])*p.pixelsize(1);
        results(:,[14,16]+3*p.fourD)=results(:,[3, 8+p.fourD])*p.pixelsize(end);
        
    end
    
    if  p.fourD
        resultstable=array2table(results,'VariableNames',{'frame','x_pix','y_pix','z_nm','fourth_variable','photons','background','crlb_x','crlb_y','crlb_z','crlb_4th','crlb_photons','crlb_background','logLikelyhood','iter','x_nm','y_nm','crlb_xnm','crlb_ynm','xfit','yfit'});
        %save(['H:\test\result' datestr(now,'mmdd') '.mat'],'results','-append')
    elseif p.fiveD
        resultstable=array2table(results,'VariableNames',{'frame',...
            'x_pix','y_pix','z_nm','fourth_variable','fiveth_variable','g2','photons','background',...
            'crlb_x','crlb_y','crlb_z','crlb_4th','crlb_5th','crlb_g2','crlb_photons','crlb_background',...
            'logLikelyhood','iter','x_nm','y_nm','crlb_xnm','crlb_ynm'});
        
        
        
    else
        if p.isspline
            resultstable=array2table(results,'VariableNames',{'frame','x_pix','y_pix','z_nm','photons','background',' crlb_x','crlb_y','crlb_z','crlb_photons','crlb_background','logLikelyhood','x_nm','y_nm','crlb_xnm','crlb_ynm'});
        else
            resultstable=array2table(results,'VariableNames',{'frame','x_pix','y_pix','sx_pix','sy_pix','photons','background',' crlb_x','crlb_y','crlb_photons','crlb_background','logLikelyhood','x_nm','y_nm','crlb_xnm','crlb_ynm'});
        end
        %
    end
    writenames=true;
    if contains(p.outputformat,'pointcloud')
        resultstable=resultstable(:,[1 13 14 4 15 9]); %for pointcloud-loader
        del='\t';
        disp('Load in http://www.cake23.de/pointcloud-loader/')
    elseif contains(p.outputformat,'ViSP')
        writenames=false;
        del='\t';
        resultstable=resultstable(:,[13 14 4 15 16 9 5 1]);
        [path,file]=fileparts(p.outputfile);
        p.outputfile=fullfile(path, [file '.3dlp']);
        disp('Load in Visp: https://science.institut-curie.org/research/multiscale-physics-biology-chemistry/umr168-physical-chemistry/team-dahan/softwares/visp-software-2/')
    else
        del=',';
        disp('Generic output. Can be imported e.g. in PALMsiever: https://github.com/PALMsiever/palm-siever')
    end
    if p.imagefile(1)~='X'
        p.imagefile=['X:\SSD_sync_files\Li_STORM_Experiments', p.imagefile(3:end)];
    end
    %     writetable(resultstable,[p.imagefile(1:end-3),'csv'],'Delimiter',del,'FileType','text','WriteVariableNames',writenames);
    
    writetable(resultstable,p.outputfile,'Delimiter',del,'FileType','text','WriteVariableNames',writenames);
    p.status.String=['Fitting done. ' num2str(size(results,1)/fittime,'%3.0f') ' fits/s. ' num2str(size(results,1),'%3.0f') ' localizations. Saved.']; drawnow
end

end

function results=fitspline(imstack,peakcoordinates,p,varstack)
if p.fourD
    %
    disp('4d_fitting');

%     
%     initP=[1.2*p.z0,2.5;
%         p.z0,2.5;
%         0.8*p.z0,2.5;]';
        initP=[p.z0,2.5]';
    %     initP=[p.z0,2.3;p.z0,2.7];
    %    save(['H:\test\result' datestr(now,'mmdd') '.mat'],'imstack');
    iteration=100;
    %     [Pcspline, CRLB, LL, ~, iter]=GPUmleFit_LM_bSpline_v2(single(imstack),single(p.coeff),iteration,6,0);
    inittmp=initP(:,1);
    
    [Pcspline CRLB LL]=GPUmleFit_LM_multiD(single(imstack),...
        7,single(iteration),single(p.coeff),0,0,single(inittmp));
   
    if size(initP,2)>1
        for i=2:size(initP,2)
            inittmp=initP(:,i);
            %             [Ph4 CRLBh LogLh5 deg iterh] = GPUmleFit_LM_bSpline_v2(single(imstack),single(p.coeff),iteration,6,0,single(inittmp));
            [Ph4 CRLBh LogLh5]=GPUmleFit_LM_multiD(single(imstack),...
                7,single(iteration),single(p.coeff),0,0,single(inittmp));
            indbetter=LogLh5-LL>1e-4; %copy only everything if LogLh increases by more than rounding error.
            Pcspline(indbetter,:)=Ph4(indbetter,:);
            CRLB(indbetter,:)=CRLBh(indbetter,:);
            LL(indbetter)=LogLh5(indbetter);
        end
    end
   
    results=zeros(size(imstack,3),17);
    %
elseif p.fiveD
    disp('5d_fitting');
    csz=size(p.coeff5);
    iteration=50;
    initP=[%z,azumathual,polar,g2
        csz(1)/2,csz(2)/3,csz(3)/2,0.75;
        csz(1)/2,2*csz(2)/3,csz(3)/2,0.75;
        ]';
    
    iter=50;
    inittmp=initP(:,1);
    %     [Pcspline CRLB LL deg iter] = GPUmleFit_LM_bSpline_v21(single(imstack),single(p.coeff5),iteration,8,0,single(p.coeff3),single(inittmp));
    [Pcspline, CRLB,LL]=GPUmleFit_LM_multiD(single(imstack),single(8),iteration,...
        single(p.coeff5),(0),0,single(inittmp),single(p.coeff3));
    %cycle fitting
    if size(initP,2)>1
        for i=2:size(initP,2)
            inittmp=initP(:,i);
            %             [Ph5 CRLBh LogLh5 deg iter] = GPUmleFit_LM_bSpline_v21(single(imstack),single(p.coeff5),iteration,8,0,single(p.coeff3),single(inittmp));
            [Ph5, CRLBh,LogLh5]=GPUmleFit_LM_multiD(single(imstack),single(8),iteration,...
                single(p.coeff5),(0),0,single(inittmp),single(p.coeff3));
            
            indbetter=LogLh5-LL>=1e-4; %copy only everything if LogLh increases by more than rounding error.
            Pcspline(indbetter,:)=Ph5(indbetter,:);
            CRLB(indbetter,:)=CRLBh(indbetter,:);
            LL(indbetter)=LogLh5(indbetter);
            
        end
    end
    
else
    if p.isspline
        if p.bidirectional
            fitmode=6;
        else
            fitmode=5;
        end
        fitpar=single(p.coeff);
    else
        if p.bidirectional
            fitmode=2;
        else
            fitmode=4;
        end
        fitpar=single(1);
    end
    
%     [Pcspline,CRLB,LL]=mleFit_LM(imstack,fitmode,50,fitpar,varstack,0);
        [Pcspline,CRLB,LL]=GPUmleFit_LM_multiD(single(imstack),single(6),50,...
                    single(p.coeff),0,0);
    results=zeros(size(imstack,3),12);
end

results(:,1)=peakcoordinates(:,3);
if  p.mirror
    results(:,2)=p.dx-Pcspline(:,2)+peakcoordinates(:,1);
else
    if p.fourD|p.fiveD
        %4d or 5d fit and x/y exchange
        results(:,2)=Pcspline(:,1)-p.dx+peakcoordinates(:,1);
        
    else
        results(:,2)=Pcspline(:,2)-p.dx+peakcoordinates(:,1);
    end
end

if  ~(p.fourD |p.fiveD)
    results(:,3)=Pcspline(:,1)-p.dx+peakcoordinates(:,2); %x,y in pixels
    results(:,4)=-(Pcspline(:,5)-p.z0)*p.dz;
    results(:,5:6)=Pcspline(:,3:4);
    results(:,7:8)=real(sqrt(CRLB(:,[2 1])));
    results(:,9)=real(sqrt(CRLB(:,5)*p.dz));
    results(:,10:11)=real(sqrt(CRLB(:,3:4)));
    results(:,12)=LL;
    
    
    if p.isspline
        % frame, x,y,z,phot,bg, errx,erry, errz,errphot, errbg,logLikelihood
        results(:,3)=Pcspline(:,1)-p.dx+peakcoordinates(:,2); %x,y in pixels
        results(:,4)=-(Pcspline(:,5)-p.z0)*p.dz;
        results(:,5:6)=Pcspline(:,3:4);
        results(:,7:8)=real(sqrt(CRLB(:,[2 1])));
        results(:,9)=real(sqrt(CRLB(:,5)*p.dz));
        results(:,10:11)=real(sqrt(CRLB(:,3:4)));
        results(:,12)=LL;
    else
        % frame, x,y,sx,sy,phot,bg, errx,erry,errphot, errbg,logLikelihood
        results(:,3)=Pcspline(:,1)-p.dx+peakcoordinates(:,2); %x,y in pixels
        results(:,4)=Pcspline(:,5);
        if p.bidirectional
            results(:,5)=results(:,4);
        else
            results(:,5)=Pcspline(:,6);
        end
        results(:,6:7)=Pcspline(:,3:4);
        
        results(:,8:9)=real(sqrt(CRLB(:,[2 1])));
        
        results(:,10:11)=real(sqrt(CRLB(:,3:4)));
        results(:,12)=LL;
    end
elseif p.fourD
    %4d fit
    results(:,3)=Pcspline(:,2)-p.dx+peakcoordinates(:,2); %x,y in pixels
    results(:,4)=-(Pcspline(:,5)-p.z0)*p.dz;
    results(:,5)=Pcspline(:,6);%4D variable
    results(:,6:7)=Pcspline(:,3:4);
    results(:,8:9)=real(sqrt(CRLB(:,[2 1])));
    results(:,10)=real(sqrt(CRLB(:,5)*p.dz));
    results(:,11)=real(sqrt(CRLB(:,6)));
    results(:,12:13)=real(sqrt(CRLB(:,3:4)));
    results(:,14)=LL;
    results(:,15)=Pcspline(:,end);
    results(:,16)=Pcspline(:,2);
    results(:,17)=Pcspline(:,1);
    
    %
elseif p.fiveD
    results(:,3)=Pcspline(:,2)-p.dx+peakcoordinates(:,2); %x,y in pixels
    results(:,4)=(Pcspline(:,5)-p.z0)*p.dz;
    results(:,5)=(Pcspline(:,6)-1)*p.du;%4th variable
    results(:,6)=(Pcspline(:,7)-1)*p.dv;%5th
    results(:,7)=Pcspline(:,8);%6th
    
    results(:,8:9)=Pcspline(:,3:4);
    results(:,10:11)=real(sqrt(CRLB(:,[2 1])));
    results(:,12)=real(sqrt(CRLB(:,5)*p.dz));
    results(:,13)=real(sqrt(CRLB(:,6))*p.du);
    results(:,14)=real(sqrt(CRLB(:,7))*p.dv);
    results(:,15)=real(sqrt(CRLB(:,7)));
    
    results(:,16:17)=real(sqrt(CRLB(:,3:4)));
    results(:,18)=LL;
    results(:,19)=Pcspline(:,end);
    
    
end
end

function img=getimage(F,reader,p)
switch p.loader
    case 1
        img=reader.read(F);
    case 2
        img=bfGetPlane(reader,F);
    case 3
        ss=[reader.getWidth reader.getHeight reader.getSize];
        if F>0&&F<=ss(3)
            pixel=reader.getPixels(F);
            img=reshape(pixel,ss(1),ss(2))';
        else
            img=[];
        end
end

end
function [data,D]=aggre(data,p)
flag=0;
while(~flag)
    flag=0;
    D=zeros(size(data,1)-1);
    for ii=1:size(data,1)-1
        for jj=ii+1:size(data,1)
            D(ii,jj)=sqrt((data(ii,1)-data(jj,1))^2+(data(ii,2)-data(jj,2))^2);
        end
    end
    D(D==0)=nan;
    agind=D<p.roifit*p.aggsz;
    if ismember(1,agind)
        [ind1,ind2]=find(agind==1);
        data(ind1(1),:)=round((data(ind1(1),:)+data(ind2(1),:))/2);
        data(ind2(1),:)=[];
    else
        flag=1;
    end
end
end


function closereader(reader,p)
switch p.loader
    case 1
        reader.close;
    case 2
    case 3
        
end

end
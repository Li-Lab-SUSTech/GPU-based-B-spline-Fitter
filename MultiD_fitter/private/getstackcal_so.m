function [splinefit,indgood,shift]=getstackcal_so(beads,p)
global stackcal_testfit
isastig=contains(p.modality,'astig'); %||contains(p.modality,'2D');
alignzastig=isastig&contains(p.zcorr,'astig');
zcorr=contains(p.zcorr,'corr');
sstack=size(beads(1).stack.image);
    halfstoreframes=round((size(beads(1).stack.image,3)-1)/2);
    if isastig    
        for B=length(beads):-1:1
            if  halfstoreframes<length(beads(B).stack.framerange)
                dframe(B)=beads(B).stack.framerange(halfstoreframes+1)-beads(B).f0;
            else
                dframe(B)=NaN;
            end
        end
        
    %remove outliers:
        badind=abs(dframe-nanmedian(dframe))>10|isnan(dframe);
        beads(badind)=[];
    

        psfx=[beads(:).psfx0];psfy=[beads(:).psfy0];
        dpsfx=(psfx-median(psfx(~isnan(psfx))))*10;
        dpsfy=(psfy-median(psfy(~isnan(psfy))))*10;
    else
        dframe=0;
        dpsfx=0;dpsfy=0;
    end
    
    allstacks=zeros(sstack(1),sstack(2),sstack(3),length(beads))+NaN;
    goodvs=[];
    for B=length(beads):-1:1
        stackh=beads(B).stack.image;
        allstacks(:,:,1:size(stackh,3),B)=stackh;
        stackh=allstacks(:,:,:,B);
        goodvs(B)=sum(~isnan(stackh(:)))/numel(stackh);
    end
    
    mstack=nanmean(allstacks,4);
    mstacks=mstack(3:end-2);
    mstack=mstack-nanmin(mstacks(:));
    mstack=mstack/nansum(mstack(:));
    for k=length(beads):-1:1
    	stackh=(allstacks(:,:,:,k));
        stackh=stackh-nanmin(stackh(:));
        stackh=stackh/nansum(stackh(:));
        dstack(k)=sum((stackh(:)-mstack(:)).^2);
    end
    dstack=dstack/mean(dstack);    
    devs=(dpsfx.^2+dpsfy.^2+dstack)./goodvs;

    if zcorr
        
        fw2=round((p.zcorrframes-1)/2);
        
    else
        fw2=2;
    end
   
%     ax=axes('Parent',uitab(p.tabgroup,'Title','scatter'));

    [~,sortinddev]=sort(devs);
%     allrois=allstacks(:,:,:,sortinddev);
    
    if alignzastig
        zshift=dframe(sortinddev)-round(median(dframe));
    else
        zshift=[];
    end
    
%     focusreference=round(median(dframe));
    midrange=halfstoreframes+1-round(median(dframe));
     framerange=max(1,midrange-fw2):min(midrange+fw2,size(stackh,3));
    p.status.String='calculate shift of individual PSFs';drawnow
    filenumber=[beads(:).filenumber];
    [corrPSF,shiftedstack,shift,beadgood]=registerPSF3D_so(allstacks,struct('sortind',sortinddev,'framerange',framerange,'alignz',zcorr,'zshiftf0',zshift,'beadfilterf0',false,'status',p.status),{},filenumber(sortinddev));
    
    
    %undo sorting by deviation to associate beads again to their
    %bead number
%     [~,sortback]=sort(sortinddev);
%     shiftedstack=shiftedstack(:,:,:,sortback);
%     beadgood=beadgood(sortback);

    indgood=beadgood;
    allrois=allstacks;
  

        %cut out the central part of the PSF correspoinding to the set
        %Roisize in x,y and z

        scorrPSF=size(corrPSF);
        x=round((scorrPSF(1)+1)/2);y=round((scorrPSF(2)+1)/2);

        dRx=round((p.ROIxy-1)/2);
        if ~isfield(p,'ROIz') || isnan(p.ROIz)
            p.ROIz=size(corrPSF,3);
        end
            dzroi=round((p.ROIz-1)/2);
        
        rangex=x-dRx:x+dRx;
        rangey=y-dRx:y+dRx;

        z=midrange;%always same reference: z=f0
        rangez=max(1,z-dzroi):min(size(corrPSF,3),z+dzroi);
        z0reference=find(rangez>=z,1,'first');
        
        %normalize PSF
        centpsf=corrPSF(rangex,rangey,z-1:z+1); %cut out rim from shift
%         centpsf=corrPSF(2:end-1,2:end-1,2:end-1); %cut out rim from shift
        minPSF=nanmin(centpsf(:));
        corrPSFn=corrPSF-minPSF;
%         corrPSFn=corrPSF;
        intglobal=nanmean(nansum(nansum(corrPSFn(rangex,rangey,z-1:z+1),1),2));
        corrPSFn=corrPSFn/intglobal;

        shiftedstack=shiftedstack/intglobal;
        corrPSFn(isnan(corrPSFn))=0;
        corrPSFn(corrPSFn<0)=0;
        corrPSFs=corrPSFn(rangex,rangey,rangez);
        
        PSFgood=true;

        %calculate effective smoothing factor. For dz=10 nm, pixelsize= 130
        %nm, a value around 1 produces visible but not too much smoothing.
        lambdax=p.smoothxy/p.cam_pixelsize_um(1)/100000;
        lambdaz=p.smoothz/p.dz*100;
        lambda=[lambdax lambdax lambdaz];
        %calculate smoothed bsplines
        b3_0=bsarray(double(corrPSFs),'lambda',lambda);

        %calculate smoothed volume
        zhd=1:1:b3_0.dataSize(3);
        dxxhd=1;
        [XX,YY,ZZ]=meshgrid(1:dxxhd:b3_0.dataSize(1),1:dxxhd:b3_0.dataSize(2),zhd);
        p.status.String='calculating cspline coefficients in progress';drawnow
        corrPSFhd = interp3_0(b3_0,XX,YY,ZZ,0);
        
        %calculate cspline coefficients
%         spline = Spline3D_v2(corrPSFhd);
%         coeff = spline.coeff;
        coeff = Spline3D_interp(corrPSFhd);
       
        %assemble output structure for saving
        bspline.bslpine=b3_0;
        cspline.coeff=coeff;
        cspline.z0=z0reference;%round((b3_0.dataSize(3)+1)/2);
        cspline.dz=p.dz;
        cspline.x0=dRx+1;
        bspline.z0=round((b3_0.dataSize(3)+1)/2);
        bspline.dz=p.dz;            
        splinefit.bspline=bspline;
        p.z0=cspline.z0;
        
        splinefit.PSF=corrPSF;
        
        splinefit.PSFsmooth=corrPSFhd;
        splinefit.cspline=cspline;

        %plot graphs
        if PSFgood       
            ax=axes(uitab(p.tabgroup,'Title','PSFz'));
             framerange0=max(p.fminmax(1)):min(p.fminmax(2));
             halfroisizebig=(size(shiftedstack,1)-1)/2;         
            ftest=z;
            xt=x;
            yt=y;
            zpall=squeeze(shiftedstack(xt,yt,:,beadgood));
            zpall2=squeeze(allrois(xt,yt,:,beadgood));
            xpall=squeeze(shiftedstack(:,yt,ftest,beadgood));
            xpall2=squeeze(allrois(:,yt,ftest,beadgood));
            for k=1:size(zpall,2)
                zpall2(:,k)=zpall2(:,k)/nanmax(zpall2(:,k));
                xpall2(:,k)=xpall2(:,k)/nanmax(xpall2(:,k));                
            end           
            zprofile=squeeze(corrPSFn(xt,yt,:));
%             mphd=round((size(corrPSFhd,1)+1)/2);
                 
            xprofile=squeeze(corrPSFn(:,yt,ftest));
            mpzhd=round((size(corrPSFhd,3)+1)/2+1);
            dzzz=round((size(corrPSFn,3)-1)/2+1)-mpzhd;
            dxxx=0.1;
            xxx=1:dxxx:b3_0.dataSize(1);
%             zzzt=0*xxx+mpzhd+dzzz-1;
            zzzt=0*xxx+ftest;
            xbs= interp3_0(b3_0,0*xxx+b3_0.dataSize(1)/2+.5,xxx,zzzt);
            zzz=1:dxxx:b3_0.dataSize(3);xxxt=0*zzz+b3_0.dataSize(1)/2+.5;
            zbs= interp3_0(b3_0,xxxt,xxxt,zzz); 
            hold(ax,'off')
             h1=plot(ax,framerange0,zpall(1:length(framerange0),:),'c');
             hold(ax,'on')
            h2=plot(ax,framerange0',zprofile(1:length(framerange0)),'k*');
            h3=plot(ax,zzz+rangez(1)+framerange0(1)-2,zbs,'b','LineWidth',2);
            xlabel(ax,'frames')
            ylabel(ax,'normalized intensity')
            ax.XLim(2)=max(framerange0);ax.XLim(1)=min(framerange0);
            title(ax,'Profile along z for x=0, y=0');
            
            legend([h1(1),h2,h3],'individual PSFs','average PSF','smoothed spline')
            
            xrange=-halfroisizebig:halfroisizebig;
             ax=axes(uitab(p.tabgroup,'Title','PSFx'));
            hold(ax,'off')
            h1=plot(ax,xrange,xpall,'c');
            hold(ax,'on')
            h2=plot(ax,xrange,xprofile,'k*');
            h3=plot(ax,(xxx-(b3_0.dataSize(1)+1)/2),xbs,'b','LineWidth',2);
            xlabel(ax,'x (pixel)')
            ylabel(ax,'normalized intensity')
            title(ax,'Profile along x for y=0, z=0');
             legend([h1(1),h2,h3],'individual PSFs','average PSF','smoothed spline')
            
            drawnow
            
            %quality control: refit all beads
            if isempty(stackcal_testfit)||stackcal_testfit
                ax=axes(uitab(p.tabgroup,'Title','validate'));
                testallrois=allrois(:,:,:,beadgood);
                testallrois(isnan(testallrois))=0;
                zall=testfit(testallrois,cspline.coeff,p,{},ax);
                corrPSFfit=corrPSF/max(corrPSF(:))*max(testallrois(:)); %bring back to some reasonable photon numbers;
                zref=testfit(corrPSFfit,cspline.coeff,p,{'k','LineWidth',2},ax);
                drawnow
            end
        end 
end

function zs=testfit(teststack,coeff,p,linepar,ax)
if nargin<4
    linepar={};
elseif ~iscell(linepar)
    linepar={linepar};
end
fitsize=min(p.ROIxy,21);
d=round((size(teststack,1)-fitsize)/2);
            range=d+1:d+fitsize;

numstack=size(teststack,4);
t=tic;
% f=figure(989);ax2=gca;hold off
    for k=1:size(teststack,4)
        if toc(t)>1
            p.status.String=['fitting test stacks: ' num2str(k/numstack,'%1.2f')];drawnow
            t=tic;
        end
        if contains(p.modality,'2D')
            fitmode=6;
        else
            fitmode=5;
        end

        [P] =  mleFit_LM(single(squeeze(teststack(range,range,:,k))),fitmode,100,single(coeff),0,1);
        
        z=(1:size(P,1))'-1;

        znm=(P(:,5)-p.z0)*p.dz;
        plot(ax,z,znm,linepar{:})
        hold(ax,'on')
        xlabel(ax,'frame')
        ylabel(ax,'zfit (nm)')
        zs(:,k)=P(:,5);
% test for the returned photons and photons in the raw image        
%         phot=P(:,3); bg=P(:,4);
%         totsum=squeeze(nansum( nansum(teststack(range,range,:,k),1),2));
%         totsum=totsum-squeeze(min(min(teststack(range,range,:,k),[],1),[],2))*length(range)^2;
%         photsum=phot+0*bg*length(range)^2;
%         plot(ax2,z,(photsum-totsum)./totsum,'.')
%         hold(ax2,'on')
    end
    
end

function teststripes(coeff,p,ax)
%not used, can be called to test for stripe artifacts.
tt=tic;

zr=0:0.2:p.ROIz;
xr=0:0.05:p.ROIxy;
hz=zeros(1,length(zr)-1);
hx=zeros(1,length(xr)-1);
hy=hx;
while toc(tt)<30
    nn=rand(11,11,10000,'single');
%     P=callYimingFitter(nn,single(coeff),50,5,0,1);
    [P] =  mleFit_LM(nn,5,50,single(coeff),0,1);
    hz=histcounts(P(:,5),zr)+hz;
    hx=histcounts(P(:,1),xr)+hx;
    hy=histcounts(P(:,2),xr)+hy;
    
end

hz(1)=[];hz(end)=[];
hz(1)=0;hz(end)=0;

indx=(hx==0);
hx(indx)=[];
indy=(hy==0);
hy(indy)=[];
hx(1)=[];hx(end)=[];
hy(1)=[];hy(end)=[];
hzx=myxcorr(hz-mean(hz),hz-mean(hz));
hxx=myxcorr(hx-mean(hx),hx-mean(hx));
hyx=myxcorr(hy-mean(hy),hy-mean(hy));
hzx(1)=0;hzx(end)=0;
ax2=axes(ax.Parent);
subplot(1,2,1,ax);
subplot(1,2,2,ax2);
findx=find(~indx);findy=find(~indy);
plot(ax,zr(2:end-2),hz,zr(2:end-2),hzx/max(hzx)*(quantile(hz,.99)));
ax.YLim(2)=(quantile(hz,.99))*1.1;
ax.YLim(1)=min(quantile(hz,.01),quantile(hzx/max(hzx)*(quantile(hz,.99)),.01));
plot(ax2,xr(findx(2:end-1)),hx,xr(findx(2:end-1)),hxx/max(hxx)*max(hx),xr(findy(2:end-1)),hy,xr(findy(2:end-1)),hyx/max(hyx)*max(hy));
end

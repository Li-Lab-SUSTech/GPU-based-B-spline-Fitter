% Copyright (c) 2021 Li Lab, Southern University of Science and Technology, Shenzhen
% author: Yiming Li
% email: liym2019@sustech.edu.cn
% date: 2022.10.25
% Tested with CUDA 11.1 (Express installation) and Matlab 2021b
%%
addpath('.\store');
addpath(genpath('..\MultiD_bspline'))
clear
close all
clc
%% generate two color psf
lambda=[580,680];
pixelsz=[110 110];
PSF2 = color_generation(lambda,31,101,2,"DMO",pixelsz);
% PSF2 = color_generation(lambda,31,101,2,"Ast",pixelsz);
%lambda test
%repeat the psf for continuous interpolation
PSFsz=size(PSF2);
tmp = zeros([PSFsz(1:3),size(PSF2,4)+3]);
tmp(:,:,:,1)=PSF2(:,:,:,1);
tmp(:,:,:,2)=PSF2(:,:,:,1);
tmp(:,:,:,end-2)=PSF2(:,:,:,end);
tmp(:,:,:,end-1)=PSF2(:,:,:,end);
tmp(:,:,:,end)=PSF2(:,:,:,end);


bc = bsarray(squeeze(tmp(:,:,:,:)),'lambda',[0 0 0.1 0.2],'degree',[3 3 3 3]);

ByteSize(bc)
%% 4D CRLB
addpath(genpath('C:\MATLAB\R2019a\bin\public'));
photon=2000;
bg=20;
zpos4 = linspace(-1000,1000,bc.dataSize(3));
zsze=80;sze = 20;
off = floor((bc.dataSize(1)-sze)/2);
f=0;
dfx=0;
dfy=0;
dfz=0;
zrange=30:70;
newTheta = ones(6,1);newTheta(3)=photon;newTheta(4)=bg;
xc = 0;yc = 0;zc = 50;NV = 6;jacobian = zeros(1,6);CRLB4 = zeros(length(zrange),NV);
dataSize = bc.dataSize;wc = (dataSize(4))/2;
k=1;
for kk = zrange
    zc = kk;
    xs= single(xc-floor(xc));
    ys= single(yc-floor(yc));
    zs= single(zc-floor(zc));
    ws= single(wc-floor(wc));
    [delta_f,delta_dfx,delta_dfy,delta_dfz,delta_dfw] = Fan_computeDelta4D_bSpline(xs,ys,zs,ws);
    hessian = zeros(6);
    for ii = 0:sze-1
        for jj = 0:sze-1
            [newDudt,model] = kernel_Derivative_bSpline4D(xc+jj+off,yc+ii+off,zc,wc,dataSize(1),dataSize(2),dataSize(3),dataSize(4),bc.coeffs,delta_f,delta_dfx,delta_dfy,delta_dfz,delta_dfw,newTheta);% nwe
            test(jj+1,ii+1) = model;
            t2 = 1/model;
            for l = 0:NV-1
                for m =l:NV-1
                    hessian(l*NV+m+1) = hessian(l*NV+m+1)+t2*newDudt(l+1)*newDudt(m+1);
                    hessian(m*NV+l+1) = hessian(l*NV+m+1);
                end
            end
        end
    end
    CRLB4(k,:) = sqrt(diag(inv(hessian)))';
    CRLB3(k,:)=sqrt(diag(inv(hessian(1:end-1,1:end-1))));
    k=k+1;
end
CRLB4(:,5) = 2000/100*CRLB4(:,5);CRLB4(:,6) = (lambda(2)-lambda(1))*CRLB4(:,6);
CRLB4(:,1) = pixelsz(1)*CRLB4(:,1);CRLB4(:,2) = pixelsz(2)*CRLB4(:,2);
% figure('name','4D CRLB','NumberTitle','off');plot(zpos4(zrange),CRLB4(:,[1,2,5]));grid;
% legend('x','y','z');xlabel('z(nm)');ylabel('CRLB(nm)')
% zrange
figure('name','Color CRLB','NumberTitle','off');
plot((zrange-50)*20,CRLB4(:,[1,2,5,6]),'LineWidth',2);grid;
legend('x','y','z','lambda','FontSize',15,'FontWeight','bold');
xlabel('z(nm)','FontSize',15,'FontWeight','bold');ylabel('CRLB(nm)','FontSize',15,'FontWeight','bold');ylim([0 60])
title('4D CRLB','FontSize',15,'FontWeight','bold');




CRLB3(:,5) = 2000/100*CRLB3(:,5);
CRLB3(:,1) = pixelsz(1)*CRLB3(:,1);CRLB3(:,2) = pixelsz(2)*CRLB3(:,2);

figure('name','Color CRLB','NumberTitle','off');plot((zrange-50)*20,CRLB3(:,[1,2,5]),'LineWidth',2);grid;
legend('x','y','z','FontSize',15,'FontWeight','bold');xlabel('z(nm)','FontSize',15,'FontWeight','bold');
ylabel('CRLB(nm)','FontSize',15,'FontWeight','bold');ylim([0 60])
title('3D CRLB','FontSize',15,'FontWeight','bold');

%% Fitting simulate  data
% photon=2000;
% bg=10;

coco1=PSF2*photon+bg;
rep=500;
zlength=zrange(1:end);
coco2=repmat(coco1(9:21,9:21,zlength,:),[1 1 rep]);
iteration=60;
% result=struct('x',[],'y',[],'class',[]);
for i=1:2
    % tmptest=zeros(size(coco(9:21,9:21,40:80,i)) +[0 0 size(coco(9:21,9:21,40:80,i),3)]);
    % tmptest() = poissrnd(coco(9:21,9:21,40:80,i));
    resulttmp=zeros(2);
    reject=0;
    result=struct('x',[],'y',[],'class',[],'z',[],'LL',[]);

   
    [P , CRLB, LL ]=GPUmleFit_LM_multiD(single(poissrnd(coco2(:,:,:,i))),...
        7,iteration,single(bc.coeffs),0,0,single([50,2.5]));

    P(:,6)=max(2,P(:,6));
    P(:,6)=min(3,P(:,6));
    P(:,6)=(P(:,6)-2.5)*2;
    adddata=P;
    result.class=[result.class; adddata(:,6)];
    result.x=[result.x; adddata(:,1)];
    result.y=[result.y; adddata(:,2)];
    result.z=[result.z; adddata(:,5)];
    result.LL=[result.LL; LL];
    reject=reject+size(P,1)-length(adddata);
    clear GPUmleFit_LM_multiD
    if i==1
%         figure;
        result1=result;
    else
%         hold;
        result2=result;
    end
    % result=sigmoid(result.class);
%     histogram(result.class,-1:0.1:1);
end

% legend([num2str(lambda(1)),'nm'],[num2str(lambda(2)),'nm'] );
% xlabel('Probability');ylabel('Count')
% title(['photon:',num2str(photon),', bg:',num2str(bg) ],'FontSize',15,'FontWeight','bold')
z1=reshape(result1.z,length(zlength),rep);
Zgt=(zlength-50)*20;
col=1:size(z1,1);
figure;plot(Zgt,Zgt,'LineWidth',5,'Color','black')
hold;scatter(Zgt,20*(z1-50),25,'filled')
xlabel('z/nm'),ylabel('fit z/nm')


rz2=reshape(result2.z,length(zlength),rep);
stdz2=std(rz2');
rz1=reshape(result1.z,length(zlength),rep);
stdz1=std(rz1');

figure;plot(Zgt,CRLB4(:,5),'LineWidth',2,'Color','black')
hold on;
scatter(Zgt,stdz1*20,25,'filled','b')
scatter(Zgt,stdz2*20,25,'filled','r')
xlim([-400 400]);
xlabel('z/nm','FontSize',15,'FontWeight','bold');ylabel('CRLBz/nm','FontSize',15,'FontWeight','bold');
legend('CRLBz','STDz1','STDz2')
ylim([0 2*max(CRLB4(:,5),[],'all')])
title(['photon:',num2str(photon),', bg:',num2str(bg) ],'FontSize',15,'FontWeight','bold')
%% Performance at different signal-to-noise ratios
silent=1;
photoncomp=struct;
rep=500;
bg=20;
for ii=1:9  % photon range
    photon=1000*ii;
    coco1=PSF2*photon+bg;
    %     result=struct('x',[],'y',[],'class',[]);
    for i=1:2
        result=struct('x',[],'y',[],'class',[],'z',[],'LL',[]);
        inputdata=coco1(9:21,9:21,zrange,i);
        Znum=size(inputdata,3);
        fit_data=repmat(inputdata,[1 1 rep]);
%                 [P CRLB LL deg iter]=GPUmleFit_LM_bSpline_v3(single(poissrnd(fit_data)),single(bc.coeffs),70,6,0);
        [P , CRLB, LL ]=GPUmleFit_LM_multiD(single(poissrnd(fit_data)),7,iteration,single(bc.coeffs),0,silent,single([50,2.5]));
%         pause(10)
        %             [P CRLB LL]=CPUmleFit_LM(single(poissrnd(coco1(9:21,9:21,30:70,i))),8,100,single(bc.coeffs));
%         clear GPUmleFit_LM_multiD
        P(:,6)=max(2,P(:,6));
        P(:,6)=min(3,P(:,6));
        P(:,6)=(P(:,6)-2.5)*2;
        %         adddata=P(find(LL>(mean(LL)*10)),:);
        adddata=P;
        result.class=[result.class; adddata(:,6)];
        result.x=[result.x; adddata(:,1)];
        result.y=[result.y; adddata(:,2)];
        result.z=[result.z; adddata(:,5)];
        result.LL=[result.LL LL];

        if i==1
            result1=result;
        else
            result2=result;
        end
    end
    photoncomp(ii).result1=result1;
    photoncomp(ii).result2=result2;
end

%% Accuracy ploting
% 2k photon and 20 bg
% figure;subplot(1,2,1);histogram(photoncomp(2).result1.class,10,'FaceColor','#0072BD');
% title(num2str(lambda(1)))
% ylabel('count')
% subplot(1,2,2);histogram(photoncomp(2).result2.class,10,'FaceColor','#D95319')
% title(num2str(lambda(2)))
% ylabel('count')


statis=struct('error',zeros(5,2),'accuracy',zeros(5,2));
% reject=[0,0.4,0.8];
reject=0;
rej1=zeros(size(photoncomp,2),size(reject,2),2);
rej2=zeros(size(zrange,2),size(reject,2),2);

for ri=1:size(reject,2)
    reject1=-reject(ri);
    reject2=reject(ri);
    for ii=1:size(photoncomp,2)
        tmpstr=photoncomp(ii);
        statis.error(ii,1)=sum(tmpstr.result1.class>reject1,'all');
        statis.error(ii,2)=sum(tmpstr.result2.class<reject2,'all');

        %         statis.accuracy(ii,1)=sum(tmpstr.result1.class<reject1,'all')/(sum(tmpstr.result1.class<reject1,'all')+sum(tmpstr.result1.class>reject2,'all'));
        %         statis.accuracy(ii,2)=sum(tmpstr.result2.class>reject2,'all')/(sum(tmpstr.result2.class<reject1,'all')+sum(tmpstr.result2.class>reject2,'all'));

        %         statis.accuracy(ii,1)=sum(tmpstr.result1.class<reject1,'all')/size(tmpstr.result1.class);
        %         statis.accuracy(ii,2)=sum(tmpstr.result2.class>reject2,'all')/size(tmpstr.result2.class);


        statis.accuracy(ii,1)=sum(tmpstr.result1.class<reject1,'all')/(sum(tmpstr.result1.class<reject1,'all')+sum(tmpstr.result1.class>reject2,'all'));
        statis.accuracy(ii,2)=sum(tmpstr.result2.class>reject2,'all')/(sum(tmpstr.result2.class>reject2,'all')+sum(tmpstr.result2.class<reject1,'all'));
        rej1(ii,ri,1)=sum(tmpstr.result1.class>reject1&tmpstr.result1.class<reject2,'all')/size(tmpstr.result1.class,1);
        rej1(ii,ri,2)=sum(tmpstr.result2.class>reject1&tmpstr.result2.class<reject2,'all')/size(tmpstr.result2.class,1);
    end


    numb={'1k','2k','3k','4k','5k','6k','7k','8k','9k','10k','11k','12k','13k','14k','15k'};
    n=size(statis.accuracy,1);
    figure;plot(1:1:n,100*statis.accuracy(:,:),'linewidth',2.5);
    ylim([0 100])
    legend([num2str(lambda(1)),'nm'],[num2str(lambda(2)),'nm'],'FontSize',12,'FontWeight','bold');
    xlabel(['photon count in ',num2str(bg),' bg'],'FontSize',12,'FontWeight','bold');
    ylabel('Accuracy','FontSize',12,'FontWeight','bold');
    set(gca,'xtick',1:n)
    set(gca,'xticklabel',numb(1:n),'FontSize',15,'FontWeight','bold','LineWidth',2)
    xlim([1 n]);


    % z acc
    % sastis1=struct('acc',zeros(41,2));
    classz1=reshape(photoncomp(5).result1.class,Znum,rep);%test in 2k photon
    classz2=reshape(photoncomp(5).result2.class,Znum,rep);
    Z=20*(zrange-50);
    for ii=1:size(zrange,2)
        satis1.acc(ii,1)=sum(classz1(ii,:)<reject1,'all')/(sum(classz1(ii,:)<reject1,'all')+sum(classz1(ii,:)>reject2,'all'));
        satis1.acc(ii,2)=sum(classz2(ii,:)>reject2,'all')/(sum(classz2(ii,:)<reject1,'all')+sum(classz2(ii,:)>reject2,'all'));
        rej2(ii,ri,1)=sum(classz1(ii,:)>reject1&classz1(ii,:)<reject2,'all')/size(classz1(ii,:),2);
        rej2(ii,ri,2)=sum(classz2(ii,:)>reject1&classz2(ii,:)<reject2,'all')/size(classz1(ii,:),2);
    end
%     figure;
%     % yyaxis left;
%     plot(Z,100*satis1.acc,'linewidth',2);
%     ylabel('Accuracy','FontSize',12,'FontWeight','bold');
%     ylim([0 100]);
%     % yyaxis right;plot(-400:20:400,CRLB4(10:50,6),'linewidth',2);ylabel('CRLB\lambda/nm','FontSize',12,'FontWeight','bold');
%     xlabel('z/nm','FontSize',12,'FontWeight','bold');
%     legend([num2str(lambda(1)),'nm'],[num2str(lambda(2)),'nm'],'FontSize',12,'FontWeight','bold','Location','best')
%     set(gca,'FontSize',15,'FontWeight','bold','linewidth',2)

    if ri==1
        resfilter=struct;
        resfilter.accuracy=statis.accuracy(:,:);
        resfilter.accz=satis1.acc;
        rejectz=rej1;
        rejectp=rej2;
    else
        resfilter.accuracy=cat(3,resfilter.accuracy,statis.accuracy(:,:));
        resfilter.accz=cat(3,resfilter.accz,satis1.acc);
        rejectz=cat(3,rejectz,rej1);
        rejectp=cat(3,rejectp,rej2);

    end
end

% ztmp1=reshape(photoncomp(5).result1.z,size(zrange,2),rep)*20;
% ztmp2=reshape(photoncomp(5).result2.z,size(zrange,2),rep)*20;
% 
% zstd1=std(ztmp1,0,2);
% zstd2=std(ztmp2,0,2);

% figure;plot(Z,CRLB4(:,5),'linewidth',2);hold on;
% scatter(Z,zstd1,'filled','MarkerFaceColor','#0072BD')
% scatter(Z,zstd2,'filled','MarkerFaceColor','#D95319')
% 
% xlabel('z (nm)')
% legend('CRLBz^{1/2}',['\sigma_{z',num2str(lambda(1)),'}'],['\sigma_{z',num2str(lambda(2)),'}'])
% ylim([0 max(CRLB4(:,5))*2])
% set(gca,'FontSize',15,'FontWeight','bold','linewidth',2)


% mz1=mean(ztmp1-1000,2);
% figure;
% errorbar(Z,mz1,zstd1,'LineWidth',2);
% 
% set(gca,'FontSize',15,'FontWeight','bold','linewidth',2)

%% smoothing
tmp=satis1.acc;
fz1 = fit((1:size(squeeze(tmp),1))',squeeze(tmp(:,1)),'smoothingspline','SmoothingParam',5e-1);
fz2 = fit((1:size(squeeze(tmp),1))',squeeze(tmp(:,2)),'smoothingspline','SmoothingParam',5e-1);

smooth1=fz1((1:size(squeeze(tmp),1))');

smooth2=fz2((1:size(squeeze(tmp),1))');

smooth1(smooth1>1)=1;
smooth2(smooth2>1)=1;

figure;
% yyaxis left;
plot(Z,[smooth1,smooth2],'linewidth',2);
ylabel('Accuracy','FontSize',12,'FontWeight','bold');
ylim([0 1]);
% yyaxis right;plot(-400:20:400,CRLB4(10:50,6),'linewidth',2);ylabel('CRLB\lambda/nm','FontSize',12,'FontWeight','bold');
xlabel('z (nm)','FontSize',12,'FontWeight','bold');
legend([num2str(lambda(1)),'nm'],[num2str(lambda(2)),'nm'],'FontSize',12,'FontWeight','bold','Location','best')
set(gca,'FontSize',15,'FontWeight','bold','linewidth',2)
title('photon:5k, bg:20')

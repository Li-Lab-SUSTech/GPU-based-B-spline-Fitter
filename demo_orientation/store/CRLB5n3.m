function [variable,CRLB53,PSFstack] = CRLB5n3(photon,bg,g2,b5,b3,dim,sze)
dataSize = b5.dataSize;
if dim==6
    theta4=0:0.05:1;
    theta4(1)=0.001;
elseif dim==3
    theta4=4:1:dataSize(dim)-4;
else
    theta4=4:1:dataSize(dim)-3;
end
coord=round((dataSize(1)-sze)/2);
Theta=repmat([coord,coord,photon,bg,26,18,22,g2]',size(theta4));
Theta(dim+2,:)=theta4;

Plength=size(Theta,2);

% off = floor((b5.dataSize(1)-sze));
NV = 8;% x,y,z,photon.bg,u,v,
% jacobian = zeros(1,NV);
CRLB53 = zeros(Plength,NV);
newDudtAll=zeros(1,NV);
PSFstack=zeros(sze,sze,Plength);
k=1;
for kk = 1:Plength
    newTheta=Theta(:,kk);
    g2=Theta(8,kk);
    xc=Theta(1,kk);
    yc=Theta(2,kk);
    zc=Theta(5,kk);
    uc=Theta(6,kk);
    vc=Theta(7,kk);

    xs= single(xc-floor(xc));
    ys= single(yc-floor(yc));
    zs= single(zc-floor(zc));
    us= single(uc-floor(uc));
    vs= single(vc-floor(vc));
    [delta_f,delta_dfx,delta_dfy,delta_dfz,delta_dfu,delta_dfv] = Fan_computeDelta5D_bSpline(xs,ys,zs,us,vs);
    [delta_f3,delta_dfx3,delta_dfy3,delta_dfz3] = computeDelta3D_bSpline_v2_single(xs,ys,zs);
    hessian = zeros(NV);
    %     tmp=[];
    for ii = 0:sze-1
        for jj = 0:sze-1
            [newDudt,model] = kernel_Derivative_bSpline5D(xc+jj,yc+ii,zc,uc,vc,dataSize(1),dataSize(2),dataSize(3),dataSize(4),dataSize(5),b5.coeffs,delta_f,delta_dfx,delta_dfy,delta_dfz,delta_dfu,delta_dfv,newTheta);% nwe

            [newDudt3,model3] = kernel_Derivative_bSpline_single(xc+jj,yc+ii,zc,dataSize(1),dataSize(2),dataSize(3),b3.coeffs,delta_f3,delta_dfx3,delta_dfy3,delta_dfz3,newTheta);

            modelAll=g2*model+(1-g2)*model3;
            for i=1:5
                newDudtAll(i)=g2*newDudt(i)+(1-g2)*newDudt3(i);
            end
            for i=6:7
                newDudtAll(i)=g2*newDudt(i);
            end
            %             tmp(jj+1,ii+1)=modelAll;
            PSFstack(ii+1,jj+1,kk)=modelAll;
            newDudtAll(8)=model-model3;

            t2 = 1/modelAll;

            for l = 0:NV-1
                for m =l:NV-1
                    hessian(l*NV+m+1) = hessian(l*NV+m+1)+t2*newDudtAll(l+1)*newDudtAll(m+1);
                    hessian(m*NV+l+1) = hessian(l*NV+m+1);
                end
            end


        end
    end
    CRLB53(k,:) = sqrt(diag(inv(hessian)))';
    k=k+1;
end

CRLB53(:,5) = b5.dz*CRLB53(:,5);CRLB53(:,6) = b5.du*CRLB53(:,6);CRLB53(:,7) = b5.dv*CRLB53(:,7);
CRLB53(:,1) = b5.dx*CRLB53(:,1);CRLB53(:,2) = b5.dy*CRLB53(:,2);
fz1 = fit((1:size(CRLB53(:,5),1))',squeeze(CRLB53(:,5)),'smoothingspline','SmoothingParam',5e-1);
CRLB53(:,5)=fz1((1:size(CRLB53(:,5),1))');

switch dim
    case 3
        variable=(theta4-dataSize(3)/2)*b5.dz;
    case 4
        variable=(theta4-4)*b5.du;
    case 5
        variable=(theta4-4)*b5.dv;
    case 6
        variable=theta4;
    otherwise
        disp('dim should be one of 3,4,5');
end

figure('name','Color CRLB','NumberTitle','off');plot(variable,CRLB53(:,[5,6,7]),'LineWidth',2);grid;
legend('z','\phi','\theta');ylabel('CRLB (nm/\circ)')
xlim([variable(1) variable(end)]);ylim([0 25])
switch dim
    case 3
        xlabel('z (nm)')
    case 4
        xlabel('\phi (\circ)')
    case 5
        xlabel('\theta (\circ)')
    case 6
        xlabel('g_2')
end
set(gca,'FontSize',12,'FontWeight','bold');
% figure('name','g2 CRLB','NumberTitle','off')
% plot(variable,CRLB53(:,8))
% xlim([variable(1)-variable(1) variable(end)+variable(1)]);
% set(gca,'FontSize',12,'FontWeight','bold');



end


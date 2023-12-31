% function [dudt,model] =  kernel_Derivative_bSpline(zc,yc,xc,xsize,ysize,zsize,delta_f,delta_dxf,delta_dyf,delta_dzf,coeff,theta)

function [dudt,model] = kernel_Derivative_bSpline_single(xi,yi,zi,nDatax,nDatay,nDataz,cMat,delta_f,delta_dfx,delta_dfy,delta_dfz,theta)

xc = floor(xi);
% sx= xi-xc;
yc = floor(yi);
% sy= yi-yc;
zc = floor(zi);
% sz= zi-zc;

xc = max(xc,1);
xc = min(xc,nDatax-1);

yc = max(yc,1);
yc = min(yc,nDatay-1);

zc = max(zc,1);
zc = min(zc,nDataz-1);

if xi == nDatax
    xc=nDatax;
end

if yi == nDatay
    yc=nDatay;
end

if zi == nDataz
    zc=nDataz;
end


kx = xc+1;
ky = yc+1;
kz = zc+1;

% dkx = kx+1;
% dky = ky+1;
% dkz = kz+1;

mx = 1;
my = 1;
mz = 1;

% dmx = 0;
% dmy = 0;
% dmz = 0;

f = single(0);
dfx = single(0);
dfy = single(0);
dfz = single(0);
nIndex = 0;
for q = 1:2
%     Bz1 = evalBSpline(sz+q-1,3);
%     Bz2 = evalBSpline(sz-q+0,3);
    for j = 1:2
%         Bx1 = evalBSpline(sx+j-1,3);
%         Bx2 = evalBSpline(sx-j+0,3);
        for i = 1:2
%             By1=evalBSpline(sy+i-1,3);
%             By2=evalBSpline(sy-i+0,3);
         
%             pd = pd+ ...
%                 cMat(ky-i+1,kx-j+1,kz-q+1)* delta_f(8*nIndex+1)+ ...
%                 cMat(ky+i-1+1,kx-j+1,kz-q+1)* delta_f(8*nIndex+2)+ ...
%                 cMat(ky-i+1,kx+j-1+1,kz-q+1)* delta_f(8*nIndex+3)+ ...
%                 cMat(ky+i-1+1,kx+j-1+1,kz-q+1)* delta_f(8*nIndex+4)+ ...
%                 cMat(ky-i+1,kx-j+1,kz+q-1+1)* delta_f(8*nIndex+5)+ ...
%                 cMat(ky+i-1+1,kx-j+1,kz+q-1+1)* delta_f(8*nIndex+6) + ...
%                 cMat(ky-i+1,kx+j-1+1,kz+q-1+1)* delta_f(8*nIndex+7) + ...
%                 cMat(ky+i-1+1,kx+j-1+1,kz+q-1+1)* delta_f(8*nIndex+8);
            
            f = f+ ...
                cMat(ky-i+my,kx-j+mx,kz-q+mz)* delta_f(8*nIndex+1)+ ...
                cMat(ky+i-1+my,kx-j+mx,kz-q+mz)* delta_f(8*nIndex+2)+ ...
                cMat(ky-i+my,kx+j-1+mx,kz-q+mz)* delta_f(8*nIndex+3)+ ...
                cMat(ky+i-1+my,kx+j-1+mx,kz-q+mz)* delta_f(8*nIndex+4)+ ...
                cMat(ky-i+my,kx-j+mx,kz+q-1+mz)* delta_f(8*nIndex+5)+ ...
                cMat(ky+i-1+my,kx-j+mx,kz+q-1+mz)* delta_f(8*nIndex+6) + ...
                cMat(ky-i+my,kx+j-1+mx,kz+q-1+mz)* delta_f(8*nIndex+7) + ...
                cMat(ky+i-1+my,kx+j-1+mx,kz+q-1+mz)* delta_f(8*nIndex+8);
            
%             dfx = dfx+ ...
%                 cMat_dx(ky-i+my,kx-j+mx,kz-q+mz)* delta_dfx(8*nIndex+1)+ ...
%                 cMat_dx(ky+i-1+my,kx-j+mx,kz-q+mz)* delta_dfx(8*nIndex+2)+ ...
%                 cMat_dx(ky-i+my,kx+j-1+mx,kz-q+mz)* delta_dfx(8*nIndex+3)+ ...
%                 cMat_dx(ky+i-1+my,kx+j-1+mx,kz-q+mz)* delta_dfx(8*nIndex+4)+ ...
%                 cMat_dx(ky-i+my,kx-j+mx,kz+q-1+mz)* delta_dfx(8*nIndex+5)+ ...
%                 cMat_dx(ky+i-1+my,kx-j+mx,kz+q-1+mz)* delta_dfx(8*nIndex+6) + ...
%                 cMat_dx(ky-i+my,kx+j-1+mx,kz+q-1+mz)* delta_dfx(8*nIndex+7) + ...
%                 cMat_dx(ky+i-1+my,kx+j-1+mx,kz+q-1+mz)* delta_dfx(8*nIndex+8);
            
            dfx = dfx+ ...
                (-1*cMat(ky-i+my,kx-j+mx,kz-q+mz)+cMat(ky-i+my+1,kx-j+mx,kz-q+mz))* delta_dfx(8*nIndex+1)+ ...
                (-1*cMat(ky+i-1+my,kx-j+mx,kz-q+mz)+cMat(ky+i-1+my+1,kx-j+mx,kz-q+mz))* delta_dfx(8*nIndex+2)+ ...
                (-1*cMat(ky-i+my,kx+j-1+mx,kz-q+mz)+cMat(ky-i+my+1,kx+j-1+mx,kz-q+mz))* delta_dfx(8*nIndex+3)+ ...
                (-1*cMat(ky+i-1+my,kx+j-1+mx,kz-q+mz)+cMat(ky+i-1+my+1,kx+j-1+mx,kz-q+mz))* delta_dfx(8*nIndex+4)+ ...
                (-1*cMat(ky-i+my,kx-j+mx,kz+q-1+mz)+cMat(ky-i+my+1,kx-j+mx,kz+q-1+mz))* delta_dfx(8*nIndex+5)+ ...
                (-1*cMat(ky+i-1+my,kx-j+mx,kz+q-1+mz)+cMat(ky+i-1+my+1,kx-j+mx,kz+q-1+mz))* delta_dfx(8*nIndex+6) + ...
                (-1*cMat(ky-i+my,kx+j-1+mx,kz+q-1+mz)+cMat(ky-i+my+1,kx+j-1+mx,kz+q-1+mz))* delta_dfx(8*nIndex+7) + ...
                (-1*cMat(ky+i-1+my,kx+j-1+mx,kz+q-1+mz)+cMat(ky+i-1+my+1,kx+j-1+mx,kz+q-1+mz))* delta_dfx(8*nIndex+8);
            
            
%             dfy = dfy+ ...
%                 cMat_dy(ky-i+my,kx-j+mx,kz-q+mz)* delta_dfy(8*nIndex+1)+ ...
%                 cMat_dy(ky+i-1+my,kx-j+mx,kz-q+mz)* delta_dfy(8*nIndex+2)+ ...
%                 cMat_dy(ky-i+my,kx+j-1+mx,kz-q+mz)* delta_dfy(8*nIndex+3)+ ...
%                 cMat_dy(ky+i-1+my,kx+j-1+mx,kz-q+mz)* delta_dfy(8*nIndex+4)+ ...
%                 cMat_dy(ky-i+my,kx-j+mx,kz+q-1+mz)* delta_dfy(8*nIndex+5)+ ...
%                 cMat_dy(ky+i-1+my,kx-j+mx,kz+q-1+mz)* delta_dfy(8*nIndex+6) + ...
%                 cMat_dy(ky-i+my,kx+j-1+mx,kz+q-1+mz)* delta_dfy(8*nIndex+7) + ...
%                 cMat_dy(ky+i-1+my,kx+j-1+mx,kz+q-1+mz)* delta_dfy(8*nIndex+8);
            
            dfy = dfy+ ...
                (-1*cMat(ky-i+my,kx-j+mx,kz-q+mz)+cMat(ky-i+my,kx-j+mx+1,kz-q+mz))* delta_dfy(8*nIndex+1)+ ...
                (-1*cMat(ky+i-1+my,kx-j+mx,kz-q+mz)+cMat(ky+i-1+my,kx-j+mx+1,kz-q+mz))* delta_dfy(8*nIndex+2)+ ...
                (-1*cMat(ky-i+my,kx+j-1+mx,kz-q+mz)+cMat(ky-i+my,kx+j-1+mx+1,kz-q+mz))* delta_dfy(8*nIndex+3)+ ...
                (-1*cMat(ky+i-1+my,kx+j-1+mx,kz-q+mz)+ cMat(ky+i-1+my,kx+j-1+mx+1,kz-q+mz))* delta_dfy(8*nIndex+4)+ ...
                (-1*cMat(ky-i+my,kx-j+mx,kz+q-1+mz)+cMat(ky-i+my,kx-j+mx+1,kz+q-1+mz))* delta_dfy(8*nIndex+5)+ ...
                (-1*cMat(ky+i-1+my,kx-j+mx,kz+q-1+mz)+cMat(ky+i-1+my,kx-j+mx+1,kz+q-1+mz))* delta_dfy(8*nIndex+6) + ...
                (-1*cMat(ky-i+my,kx+j-1+mx,kz+q-1+mz)+cMat(ky-i+my,kx+j-1+mx+1,kz+q-1+mz))* delta_dfy(8*nIndex+7) + ...
                (-1*cMat(ky+i-1+my,kx+j-1+mx,kz+q-1+mz)+cMat(ky+i-1+my,kx+j-1+mx+1,kz+q-1+mz))* delta_dfy(8*nIndex+8);
            
            
%             dfz = dfz+ ...
%                 cMat_dz(ky-i+my,kx-j+mx,kz-q+mz)* delta_dfz(8*nIndex+1)+ ...
%                 cMat_dz(ky+i-1+my,kx-j+mx,kz-q+mz)* delta_dfz(8*nIndex+2)+ ...
%                 cMat_dz(ky-i+my,kx+j-1+mx,kz-q+mz)* delta_dfz(8*nIndex+3)+ ...
%                 cMat_dz(ky+i-1+my,kx+j-1+mx,kz-q+mz)* delta_dfz(8*nIndex+4)+ ...
%                 cMat_dz(ky-i+my,kx-j+mx,kz+q-1+mz)* delta_dfz(8*nIndex+5)+ ...
%                 cMat_dz(ky+i-1+my,kx-j+mx,kz+q-1+mz)* delta_dfz(8*nIndex+6) + ...
%                 cMat_dz(ky-i+my,kx+j-1+mx,kz+q-1+mz)* delta_dfz(8*nIndex+7) + ...
%                 cMat_dz(ky+i-1+my,kx+j-1+mx,kz+q-1+mz)* delta_dfz(8*nIndex+8);
            
            dfz = dfz+ ...
                (-1*cMat(ky-i+my,kx-j+mx,kz-q+mz)+cMat(ky-i+my,kx-j+mx,kz-q+mz+1))* delta_dfz(8*nIndex+1)+ ...
                (-1*cMat(ky+i-1+my,kx-j+mx,kz-q+mz)+cMat(ky+i-1+my,kx-j+mx,kz-q+mz+1))* delta_dfz(8*nIndex+2)+ ...
                (-1*cMat(ky-i+my,kx+j-1+mx,kz-q+mz)+cMat(ky-i+my,kx+j-1+mx,kz-q+mz+1))* delta_dfz(8*nIndex+3)+ ...
                (-1*cMat(ky+i-1+my,kx+j-1+mx,kz-q+mz)+cMat(ky+i-1+my,kx+j-1+mx,kz-q+mz+1))* delta_dfz(8*nIndex+4)+ ...
                (-1*cMat(ky-i+my,kx-j+mx,kz+q-1+mz)+cMat(ky-i+my,kx-j+mx,kz+q-1+mz+1))* delta_dfz(8*nIndex+5)+ ...
                (-1*cMat(ky+i-1+my,kx-j+mx,kz+q-1+mz)+cMat(ky+i-1+my,kx-j+mx,kz+q-1+mz+1))* delta_dfz(8*nIndex+6) + ...
                (-1*cMat(ky-i+my,kx+j-1+mx,kz+q-1+mz)+cMat(ky-i+my,kx+j-1+mx,kz+q-1+mz+1))* delta_dfz(8*nIndex+7) + ...
                (-1*cMat(ky+i-1+my,kx+j-1+mx,kz+q-1+mz)+cMat(ky+i-1+my,kx+j-1+mx,kz+q-1+mz+1))* delta_dfz(8*nIndex+8);
            
            
            
            
            nIndex = nIndex+1;
        end
    end
end

















% [temp,dfx,dfy,dfz] = fAt3D_bs_v6(xc+jj+off,yc+ii+off,zc,dataSize(1),dataSize(2),dataSize(3),b3.coeffs,delta_f,delta_dfx,delta_dfy,delta_dfz);




dudt(1) = -1*theta(3)*dfy;
dudt(2) = -1*theta(3)*dfx;



dudt(5) = theta(3)*dfz;
dudt(3) = f;
dudt(4) = 1;
model = single(theta(4)+theta(3)*f);








% dudt = zeros(5,1);
% xc = max(xc,0);
% xc = min(xc,xsize-1);
% 
% yc = max(yc,0);
% yc = min(yc,ysize-1);
% 
% zc = max(zc,0);
% zc = min(zc,zsize-1);
% 
% temp = coeff(xc+1,yc+1,zc+1,:);
% dudt(1)=-1*theta(3)*sum(delta_dxf.*(temp(:)));
% dudt(2)=-1*theta(3)*sum(delta_dyf.*(temp(:)));
% dudt(5)=theta(2)*sum(delta_dzf.*(temp(:)));
% dudt(3)=sum(delta_f.*(temp(:)));
% dudt(4)=1;
% model = theta(4)+theta(3)*dudt(3);


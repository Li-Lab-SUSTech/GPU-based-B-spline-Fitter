function [delta_f,delta_dfx,delta_dfy,delta_dfz,delta_dfw] = Fan_computeDelta4D_bSpline(x_delta,y_delta,z_delta,w_delta)
%UNTITLED4 此处显示有关此函数的摘要
%   此处显示详细说明
dx_delta = x_delta-1;
dy_delta = y_delta-1;
dz_delta = z_delta-1;
dw_delta = w_delta-1;

delta_f = zeros(256,1);
delta_dfx = zeros(256,1);
delta_dfy = zeros(256,1);
delta_dfz = zeros(256,1);
delta_dfw = zeros(256,1);
nIndex = 0;


for r=1:2
    Bw1=evalBSpline_single(w_delta+r-1,3);
    Bw2=evalBSpline_single(w_delta-r+0,3);
    
    dBw1=evalBSpline_single(dw_delta+r-1/2,2);
    dBw2=evalBSpline_single(dw_delta-r+1/2,2);
    
    Bw1 = single(Bw1);
    Bw2 = single(Bw2);
    dBw1 = single(dBw1);
    dBw2 = single(dBw2);
    for q = 1:2
        Bz1 = evalBSpline_single(z_delta+q-1,3);
        Bz2 = evalBSpline_single(z_delta-q+0,3);
        
        dBz1 = evalBSpline_single(dz_delta+q-1/2,2);
        dBz2 = evalBSpline_single(dz_delta-q+1/2,2);
        
        Bz1 = single(Bz1);
        Bz2 = single(Bz2);
        dBz1 = single(dBz1);
        dBz2 = single(dBz2);
        for j = 1:2
            Bx1 = evalBSpline_single(x_delta+j-1,3);
            Bx2 = evalBSpline_single(x_delta-j+0,3);
            
            dBx1 = evalBSpline_single(dx_delta+j-1/2,2);
            dBx2 = evalBSpline_single(dx_delta-j+1/2,2);
            
            Bx1 = single(Bx1);
            Bx2 = single(Bx2);
            dBx1 = single(dBx1);
            dBx2 = single(dBx2);
            
            for i = 1:2
                By1=evalBSpline_single(y_delta+i-1,3);
                By2=evalBSpline_single(y_delta-i+0,3);
                
                dBy1=evalBSpline_single(dy_delta+i-1/2,2);
                dBy2=evalBSpline_single(dy_delta-i+1/2,2);
                
                By1 = single(By1);
                By2 = single(By2);
                dBy1 = single(dBy1);
                dBy2 = single(dBy2);
                %             dxf(8*(q-1)*i*j+8*j*i+8*(i-1)+1)
                
                delta_f(16*nIndex+1) = By1*Bx1*Bz1*Bw1;
                delta_f(16*nIndex+2) = By2*Bx1*Bz1*Bw1;
                delta_f(16*nIndex+3) = By1*Bx2*Bz1*Bw1;
                delta_f(16*nIndex+4) = By2*Bx2*Bz1*Bw1;
                delta_f(16*nIndex+5) = By1*Bx1*Bz2*Bw1;
                delta_f(16*nIndex+6) = By2*Bx1*Bz2*Bw1;
                delta_f(16*nIndex+7) = By1*Bx2*Bz2*Bw1;
                delta_f(16*nIndex+8) = By2*Bx2*Bz2*Bw1;
                
                delta_f(16*nIndex+9) = By1*Bx1*Bz1*Bw2;
                delta_f(16*nIndex+10) = By2*Bx1*Bz1*Bw2;
                delta_f(16*nIndex+11) = By1*Bx2*Bz1*Bw2;
                delta_f(16*nIndex+12) = By2*Bx2*Bz1*Bw2;
                delta_f(16*nIndex+13) = By1*Bx1*Bz2*Bw2;
                delta_f(16*nIndex+14) = By2*Bx1*Bz2*Bw2;
                delta_f(16*nIndex+15) = By1*Bx2*Bz2*Bw2;
                delta_f(16*nIndex+16) = By2*Bx2*Bz2*Bw2;
                %dx
                delta_dfx(16*nIndex+1) = dBy1*Bx1*Bz1*Bw1;
                delta_dfx(16*nIndex+2) = dBy2*Bx1*Bz1*Bw1;
                delta_dfx(16*nIndex+3) = dBy1*Bx2*Bz1*Bw1;
                delta_dfx(16*nIndex+4) = dBy2*Bx2*Bz1*Bw1;
                delta_dfx(16*nIndex+5) = dBy1*Bx1*Bz2*Bw1;
                delta_dfx(16*nIndex+6) = dBy2*Bx1*Bz2*Bw1;
                delta_dfx(16*nIndex+7) = dBy1*Bx2*Bz2*Bw1;
                delta_dfx(16*nIndex+8) = dBy2*Bx2*Bz2*Bw1;
                
                delta_dfx(16*nIndex+9) = dBy1*Bx1*Bz1*Bw2;
                delta_dfx(16*nIndex+10) = dBy2*Bx1*Bz1*Bw2;
                delta_dfx(16*nIndex+11) = dBy1*Bx2*Bz1*Bw2;
                delta_dfx(16*nIndex+12) = dBy2*Bx2*Bz1*Bw2;
                delta_dfx(16*nIndex+13) = dBy1*Bx1*Bz2*Bw2;
                delta_dfx(16*nIndex+14) = dBy2*Bx1*Bz2*Bw2;
                delta_dfx(16*nIndex+15) = dBy1*Bx2*Bz2*Bw2;
                delta_dfx(16*nIndex+16) = dBy2*Bx2*Bz2*Bw2;
                %dy
                delta_dfy(16*nIndex+1) = By1*dBx1*Bz1*Bw1;
                delta_dfy(16*nIndex+2) = By2*dBx1*Bz1*Bw1;
                delta_dfy(16*nIndex+3) = By1*dBx2*Bz1*Bw1;
                delta_dfy(16*nIndex+4) = By2*dBx2*Bz1*Bw1;
                delta_dfy(16*nIndex+5) = By1*dBx1*Bz2*Bw1;
                delta_dfy(16*nIndex+6) = By2*dBx1*Bz2*Bw1;
                delta_dfy(16*nIndex+7) = By1*dBx2*Bz2*Bw1;
                delta_dfy(16*nIndex+8) = By2*dBx2*Bz2*Bw1;
                
                delta_dfy(16*nIndex+9) = By1*dBx1*Bz1*Bw2;
                delta_dfy(16*nIndex+10) = By2*dBx1*Bz1*Bw2;
                delta_dfy(16*nIndex+11) = By1*dBx2*Bz1*Bw2;
                delta_dfy(16*nIndex+12) = By2*dBx2*Bz1*Bw2;
                delta_dfy(16*nIndex+13) = By1*dBx1*Bz2*Bw2;
                delta_dfy(16*nIndex+14) = By2*dBx1*Bz2*Bw2;
                delta_dfy(16*nIndex+15) = By1*dBx2*Bz2*Bw2;
                delta_dfy(16*nIndex+16) = By2*dBx2*Bz2*Bw2;
                %dz
                delta_dfz(16*nIndex+1) = By1*Bx1*dBz1*Bw1;
                delta_dfz(16*nIndex+2) = By2*Bx1*dBz1*Bw1;
                delta_dfz(16*nIndex+3) = By1*Bx2*dBz1*Bw1;
                delta_dfz(16*nIndex+4) = By2*Bx2*dBz1*Bw1;
                delta_dfz(16*nIndex+5) = By1*Bx1*dBz2*Bw1;
                delta_dfz(16*nIndex+6) = By2*Bx1*dBz2*Bw1;
                delta_dfz(16*nIndex+7) = By1*Bx2*dBz2*Bw1;
                delta_dfz(16*nIndex+8) = By2*Bx2*dBz2*Bw1;
                
                delta_dfz(16*nIndex+9) = By1*Bx1*dBz1*Bw2;
                delta_dfz(16*nIndex+10) = By2*Bx1*dBz1*Bw2;
                delta_dfz(16*nIndex+11) = By1*Bx2*dBz1*Bw2;
                delta_dfz(16*nIndex+12) = By2*Bx2*dBz1*Bw2;
                delta_dfz(16*nIndex+13) = By1*Bx1*dBz2*Bw2;
                delta_dfz(16*nIndex+14) = By2*Bx1*dBz2*Bw2;
                delta_dfz(16*nIndex+15) = By1*Bx2*dBz2*Bw2;
                delta_dfz(16*nIndex+16) = By2*Bx2*dBz2*Bw2;
                
                %dw
                delta_dfw(16*nIndex+1) = By1*Bx1*Bz1*dBw1;
                delta_dfw(16*nIndex+2) = By2*Bx1*Bz1*dBw1;
                delta_dfw(16*nIndex+3) = By1*Bx2*Bz1*dBw1;
                delta_dfw(16*nIndex+4) = By2*Bx2*Bz1*dBw1;
                delta_dfw(16*nIndex+5) = By1*Bx1*Bz2*dBw1;
                delta_dfw(16*nIndex+6) = By2*Bx1*Bz2*dBw1;
                delta_dfw(16*nIndex+7) = By1*Bx2*Bz2*dBw1;
                delta_dfw(16*nIndex+8) = By2*Bx2*Bz2*dBw1;
                
                delta_dfw(16*nIndex+9) = By1*Bx1*Bz1*dBw2;
                delta_dfw(16*nIndex+10) = By2*Bx1*Bz1*dBw2;
                delta_dfw(16*nIndex+11) = By1*Bx2*Bz1*dBw2;
                delta_dfw(16*nIndex+12) = By2*Bx2*Bz1*dBw2;
                delta_dfw(16*nIndex+13) = By1*Bx1*Bz2*dBw2;
                delta_dfw(16*nIndex+14) = By2*Bx1*Bz2*dBw2;
                delta_dfw(16*nIndex+15) = By1*Bx2*Bz2*dBw2;
                delta_dfw(16*nIndex+16) = By2*Bx2*Bz2*dBw2;
                
                nIndex = nIndex+1;
            end
        end
    end
end

end


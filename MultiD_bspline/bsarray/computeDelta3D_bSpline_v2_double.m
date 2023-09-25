function [delta_f,delta_dfx,delta_dfy,delta_dfz] = computeDelta3D_bSpline_v2_double(x_delta,y_delta,z_delta)

dx_delta = x_delta-1;
dy_delta = y_delta-1;
dz_delta = z_delta-1;

delta_f = (zeros(64,1));
delta_dfx = (zeros(64,1));
delta_dfy = (zeros(64,1));
delta_dfz = (zeros(64,1));
nIndex = 0;
for q = 1:2
    Bz1 = evalBSpline(z_delta+q-1,3);
    Bz2 = evalBSpline(z_delta-q+0,3);
    
    dBz1 = evalBSpline(dz_delta+q-1/2,2);
    dBz2 = evalBSpline(dz_delta-q+1/2,2);
    
    Bz1 = (Bz1);
    Bz2 = (Bz2);
    dBz1 = (dBz1);
    dBz2 = (dBz2);
    
    for j = 1:2
        Bx1 = evalBSpline(x_delta+j-1,3);
        Bx2 = evalBSpline(x_delta-j+0,3);
        
        dBx1 = evalBSpline(dx_delta+j-1/2,2);
        dBx2 = evalBSpline(dx_delta-j+1/2,2);
        
        Bx1 = (Bx1);
        Bx2 = (Bx2);
        dBx1 = (dBx1);
        dBx2 = (dBx2);
        for i = 1:2
            By1=evalBSpline(y_delta+i-1,3);
            By2=evalBSpline(y_delta-i+0,3);
            
            dBy1=evalBSpline(dy_delta+i-1/2,2);
            dBy2=evalBSpline(dy_delta-i+1/2,2);
            
            By1 = (By1);
            By2 = (By2);
            dBy1 = (dBy1);
            dBy2 = (dBy2);
            
%             dxf(8*(q-1)*i*j+8*j*i+8*(i-1)+1)
            delta_f(8*nIndex+1) = By1*Bx1*Bz1;
            delta_f(8*nIndex+2) = By2*Bx1*Bz1;
            delta_f(8*nIndex+3) = By1*Bx2*Bz1;
            delta_f(8*nIndex+4) = By2*Bx2*Bz1;
            delta_f(8*nIndex+5) = By1*Bx1*Bz2;
            delta_f(8*nIndex+6) = By2*Bx1*Bz2;
            delta_f(8*nIndex+7) = By1*Bx2*Bz2;
            delta_f(8*nIndex+8) = By2*Bx2*Bz2;
            
            delta_dfx(8*nIndex+1) = dBy1*Bx1*Bz1;
            delta_dfx(8*nIndex+2) = dBy2*Bx1*Bz1;
            delta_dfx(8*nIndex+3) = dBy1*Bx2*Bz1;
            delta_dfx(8*nIndex+4) = dBy2*Bx2*Bz1;
            delta_dfx(8*nIndex+5) = dBy1*Bx1*Bz2;
            delta_dfx(8*nIndex+6) = dBy2*Bx1*Bz2;
            delta_dfx(8*nIndex+7) = dBy1*Bx2*Bz2;
            delta_dfx(8*nIndex+8) = dBy2*Bx2*Bz2;
            
            delta_dfy(8*nIndex+1) = By1*dBx1*Bz1;
            delta_dfy(8*nIndex+2) = By2*dBx1*Bz1;
            delta_dfy(8*nIndex+3) = By1*dBx2*Bz1;
            delta_dfy(8*nIndex+4) = By2*dBx2*Bz1;
            delta_dfy(8*nIndex+5) = By1*dBx1*Bz2;
            delta_dfy(8*nIndex+6) = By2*dBx1*Bz2;
            delta_dfy(8*nIndex+7) = By1*dBx2*Bz2;
            delta_dfy(8*nIndex+8) = By2*dBx2*Bz2;
            
            delta_dfz(8*nIndex+1) = By1*Bx1*dBz1;
            delta_dfz(8*nIndex+2) = By2*Bx1*dBz1;
            delta_dfz(8*nIndex+3) = By1*Bx2*dBz1;
            delta_dfz(8*nIndex+4) = By2*Bx2*dBz1;
            delta_dfz(8*nIndex+5) = By1*Bx1*dBz2;
            delta_dfz(8*nIndex+6) = By2*Bx1*dBz2;
            delta_dfz(8*nIndex+7) = By1*Bx2*dBz2;
            delta_dfz(8*nIndex+8) = By2*Bx2*dBz2;
            
            nIndex = nIndex+1;
            
        end
    end
end
function [delta_f,delta_dfx,delta_dfy,delta_dfz] = computeDelta3D_bSpline_v2_single(x_delta,y_delta,z_delta)

dx_delta = x_delta-1;
dy_delta = y_delta-1;
dz_delta = z_delta-1;

delta_f = single(zeros(64,1));
delta_dfx = single(zeros(64,1));
delta_dfy = single(zeros(64,1));
delta_dfz = single(zeros(64,1));
nIndex = 0;
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
            delta_f(8*nIndex+1) = By1*Bx1*Bz1;
            delta_f(8*nIndex+2) = By2*Bx1*Bz1;
            delta_f(8*nIndex+3) = By1*Bx2*Bz1;
            delta_f(8*nIndex+4) = By2*Bx2*Bz1;
            delta_f(8*nIndex+5) = By1*Bx1*Bz2;
            delta_f(8*nIndex+6) = By2*Bx1*Bz2;
            delta_f(8*nIndex+7) = By1*Bx2*Bz2;
            delta_f(8*nIndex+8) = By2*Bx2*Bz2;
            
%             delta_dfx(8*nIndex+1) = dBy1;
%             delta_dfx(8*nIndex+2) = dBy2;
%             delta_dfx(8*nIndex+3) = Bx1;
%             delta_dfx(8*nIndex+4) = By1;
%             delta_dfx(8*nIndex+5) = Bz1;
%             delta_dfx(8*nIndex+6) = Bx2;
%             delta_dfx(8*nIndex+7) = By2;
%             delta_dfx(8*nIndex+8) = Bz2;
            
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
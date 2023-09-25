function [delta_f,delta_dfx,delta_dfy,delta_dfz,delta_dfu,delta_dfv] = Fan_computeDelta5D_bSpline(x_delta,y_delta,z_delta,u_delta,v_delta)
%UNTITLED4 此处显示有关此函数的摘要
%   此处显示详细说明
dx_delta = x_delta-1;
dy_delta = y_delta-1;
dz_delta = z_delta-1;
du_delta = u_delta-1;
dv_delta = v_delta-1;

delta_f = zeros(1024,1);
delta_dfx = zeros(1024,1);
delta_dfy = zeros(1024,1);
delta_dfz = zeros(1024,1);
delta_dfu = zeros(1024,1);
delta_dfv = zeros(1024,1);

nIndex = 0;

for s=1:2
    Bv1=evalBSpline_single(v_delta+s-1,3);
    Bv2=evalBSpline_single(v_delta-s+0,3);
    
    dBv1=evalBSpline_single(dv_delta+s-1/2,2);
    dBv2=evalBSpline_single(dv_delta-s+1/2,2);
    
    Bv1 = single(Bv1);
    Bv2 = single(Bv2);
    dBv1 = single(dBv1);
    dBv2 = single(dBv2);
    
    for r=1:2
        Bu1=evalBSpline_single(u_delta+r-1,3);
        Bu2=evalBSpline_single(u_delta-r+0,3);

        dBu1=evalBSpline_single(du_delta+r-1/2,2);
        dBu2=evalBSpline_single(du_delta-r+1/2,2);

        Bu1 = single(Bu1);
        Bu2 = single(Bu2);
        dBu1 = single(dBu1);
        dBu2 = single(dBu2);
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

                    delta_f(32*nIndex+1) = By1*Bx1*Bz1*Bu1*Bv1;
                    delta_f(32*nIndex+2) = By2*Bx1*Bz1*Bu1*Bv1;
                    delta_f(32*nIndex+3) = By1*Bx2*Bz1*Bu1*Bv1;
                    delta_f(32*nIndex+4) = By2*Bx2*Bz1*Bu1*Bv1;
                    delta_f(32*nIndex+5) = By1*Bx1*Bz2*Bu1*Bv1;
                    delta_f(32*nIndex+6) = By2*Bx1*Bz2*Bu1*Bv1;
                    delta_f(32*nIndex+7) = By1*Bx2*Bz2*Bu1*Bv1;
                    delta_f(32*nIndex+8) = By2*Bx2*Bz2*Bu1*Bv1;
                    delta_f(32*nIndex+9) = By1*Bx1*Bz1*Bu2*Bv1;
                    delta_f(32*nIndex+10) = By2*Bx1*Bz1*Bu2*Bv1;
                    delta_f(32*nIndex+11) = By1*Bx2*Bz1*Bu2*Bv1;
                    delta_f(32*nIndex+12) = By2*Bx2*Bz1*Bu2*Bv1;
                    delta_f(32*nIndex+13) = By1*Bx1*Bz2*Bu2*Bv1;
                    delta_f(32*nIndex+14) = By2*Bx1*Bz2*Bu2*Bv1;
                    delta_f(32*nIndex+15) = By1*Bx2*Bz2*Bu2*Bv1;
                    delta_f(32*nIndex+16) = By2*Bx2*Bz2*Bu2*Bv1;
                    
                    delta_f(32*nIndex+17) = By1*Bx1*Bz1*Bu1*Bv2;
                    delta_f(32*nIndex+18) = By2*Bx1*Bz1*Bu1*Bv2;
                    delta_f(32*nIndex+19) = By1*Bx2*Bz1*Bu1*Bv2;
                    delta_f(32*nIndex+20) = By2*Bx2*Bz1*Bu1*Bv2;
                    delta_f(32*nIndex+21) = By1*Bx1*Bz2*Bu1*Bv2;
                    delta_f(32*nIndex+22) = By2*Bx1*Bz2*Bu1*Bv2;
                    delta_f(32*nIndex+23) = By1*Bx2*Bz2*Bu1*Bv2;
                    delta_f(32*nIndex+24) = By2*Bx2*Bz2*Bu1*Bv2;
                    delta_f(32*nIndex+25) = By1*Bx1*Bz1*Bu2*Bv2;
                    delta_f(32*nIndex+26) = By2*Bx1*Bz1*Bu2*Bv2;
                    delta_f(32*nIndex+27) = By1*Bx2*Bz1*Bu2*Bv2;
                    delta_f(32*nIndex+28) = By2*Bx2*Bz1*Bu2*Bv2;
                    delta_f(32*nIndex+29) = By1*Bx1*Bz2*Bu2*Bv2;
                    delta_f(32*nIndex+30) = By2*Bx1*Bz2*Bu2*Bv2;
                    delta_f(32*nIndex+31) = By1*Bx2*Bz2*Bu2*Bv2;
                    delta_f(32*nIndex+32) = By2*Bx2*Bz2*Bu2*Bv2;

                    %dx
                    delta_dfx(32*nIndex+1) = dBy1*Bx1*Bz1*Bu1*Bv1;
                    delta_dfx(32*nIndex+2) = dBy2*Bx1*Bz1*Bu1*Bv1;
                    delta_dfx(32*nIndex+3) = dBy1*Bx2*Bz1*Bu1*Bv1;
                    delta_dfx(32*nIndex+4) = dBy2*Bx2*Bz1*Bu1*Bv1;
                    delta_dfx(32*nIndex+5) = dBy1*Bx1*Bz2*Bu1*Bv1;
                    delta_dfx(32*nIndex+6) = dBy2*Bx1*Bz2*Bu1*Bv1;
                    delta_dfx(32*nIndex+7) = dBy1*Bx2*Bz2*Bu1*Bv1;
                    delta_dfx(32*nIndex+8) = dBy2*Bx2*Bz2*Bu1*Bv1;
                    delta_dfx(32*nIndex+9) = dBy1*Bx1*Bz1*Bu2*Bv1;
                    delta_dfx(32*nIndex+10) = dBy2*Bx1*Bz1*Bu2*Bv1;
                    delta_dfx(32*nIndex+11) = dBy1*Bx2*Bz1*Bu2*Bv1;
                    delta_dfx(32*nIndex+12) = dBy2*Bx2*Bz1*Bu2*Bv1;
                    delta_dfx(32*nIndex+13) = dBy1*Bx1*Bz2*Bu2*Bv1;
                    delta_dfx(32*nIndex+14) = dBy2*Bx1*Bz2*Bu2*Bv1;
                    delta_dfx(32*nIndex+15) = dBy1*Bx2*Bz2*Bu2*Bv1;
                    delta_dfx(32*nIndex+16) = dBy2*Bx2*Bz2*Bu2*Bv1;
                    
                    delta_dfx(32*nIndex+17) = dBy1*Bx1*Bz1*Bu1*Bv2;
                    delta_dfx(32*nIndex+18) = dBy2*Bx1*Bz1*Bu1*Bv2;
                    delta_dfx(32*nIndex+19) = dBy1*Bx2*Bz1*Bu1*Bv2;
                    delta_dfx(32*nIndex+20) = dBy2*Bx2*Bz1*Bu1*Bv2;
                    delta_dfx(32*nIndex+21) = dBy1*Bx1*Bz2*Bu1*Bv2;
                    delta_dfx(32*nIndex+22) = dBy2*Bx1*Bz2*Bu1*Bv2;
                    delta_dfx(32*nIndex+23) = dBy1*Bx2*Bz2*Bu1*Bv2;
                    delta_dfx(32*nIndex+24) = dBy2*Bx2*Bz2*Bu1*Bv2;
                    delta_dfx(32*nIndex+25) = dBy1*Bx1*Bz1*Bu2*Bv2;
                    delta_dfx(32*nIndex+26) = dBy2*Bx1*Bz1*Bu2*Bv2;
                    delta_dfx(32*nIndex+27) = dBy1*Bx2*Bz1*Bu2*Bv2;
                    delta_dfx(32*nIndex+28) = dBy2*Bx2*Bz1*Bu2*Bv2;
                    delta_dfx(32*nIndex+29) = dBy1*Bx1*Bz2*Bu2*Bv2;
                    delta_dfx(32*nIndex+30) = dBy2*Bx1*Bz2*Bu2*Bv2;
                    delta_dfx(32*nIndex+31) = dBy1*Bx2*Bz2*Bu2*Bv2;
                    delta_dfx(32*nIndex+32) = dBy2*Bx2*Bz2*Bu2*Bv2;

                    %dy
                    delta_dfy(32*nIndex+1) = By1*dBx1*Bz1*Bu1*Bv1;
                    delta_dfy(32*nIndex+2) = By2*dBx1*Bz1*Bu1*Bv1;
                    delta_dfy(32*nIndex+3) = By1*dBx2*Bz1*Bu1*Bv1;
                    delta_dfy(32*nIndex+4) = By2*dBx2*Bz1*Bu1*Bv1;
                    delta_dfy(32*nIndex+5) = By1*dBx1*Bz2*Bu1*Bv1;
                    delta_dfy(32*nIndex+6) = By2*dBx1*Bz2*Bu1*Bv1;
                    delta_dfy(32*nIndex+7) = By1*dBx2*Bz2*Bu1*Bv1;
                    delta_dfy(32*nIndex+8) = By2*dBx2*Bz2*Bu1*Bv1;
                    delta_dfy(32*nIndex+9) = By1*dBx1*Bz1*Bu2*Bv1;
                    delta_dfy(32*nIndex+10) = By2*dBx1*Bz1*Bu2*Bv1;
                    delta_dfy(32*nIndex+11) = By1*dBx2*Bz1*Bu2*Bv1;
                    delta_dfy(32*nIndex+12) = By2*dBx2*Bz1*Bu2*Bv1;
                    delta_dfy(32*nIndex+13) = By1*dBx1*Bz2*Bu2*Bv1;
                    delta_dfy(32*nIndex+14) = By2*dBx1*Bz2*Bu2*Bv1;
                    delta_dfy(32*nIndex+15) = By1*dBx2*Bz2*Bu2*Bv1;
                    delta_dfy(32*nIndex+16) = By2*dBx2*Bz2*Bu2*Bv1;
                    
                    delta_dfy(32*nIndex+17) = By1*dBx1*Bz1*Bu1*Bv2;
                    delta_dfy(32*nIndex+18) = By2*dBx1*Bz1*Bu1*Bv2;
                    delta_dfy(32*nIndex+19) = By1*dBx2*Bz1*Bu1*Bv2;
                    delta_dfy(32*nIndex+20) = By2*dBx2*Bz1*Bu1*Bv2;
                    delta_dfy(32*nIndex+21) = By1*dBx1*Bz2*Bu1*Bv2;
                    delta_dfy(32*nIndex+22) = By2*dBx1*Bz2*Bu1*Bv2;
                    delta_dfy(32*nIndex+23) = By1*dBx2*Bz2*Bu1*Bv2;
                    delta_dfy(32*nIndex+24) = By2*dBx2*Bz2*Bu1*Bv2;
                    delta_dfy(32*nIndex+25) = By1*dBx1*Bz1*Bu2*Bv2;
                    delta_dfy(32*nIndex+26) = By2*dBx1*Bz1*Bu2*Bv2;
                    delta_dfy(32*nIndex+27) = By1*dBx2*Bz1*Bu2*Bv2;
                    delta_dfy(32*nIndex+28) = By2*dBx2*Bz1*Bu2*Bv2;
                    delta_dfy(32*nIndex+29) = By1*dBx1*Bz2*Bu2*Bv2;
                    delta_dfy(32*nIndex+30) = By2*dBx1*Bz2*Bu2*Bv2;
                    delta_dfy(32*nIndex+31) = By1*dBx2*Bz2*Bu2*Bv2;
                    delta_dfy(32*nIndex+32) = By2*dBx2*Bz2*Bu2*Bv2;

                    %dz
                    delta_dfz(32*nIndex+1) = By1*Bx1*dBz1*Bu1*Bv1;
                    delta_dfz(32*nIndex+2) = By2*Bx1*dBz1*Bu1*Bv1;
                    delta_dfz(32*nIndex+3) = By1*Bx2*dBz1*Bu1*Bv1;
                    delta_dfz(32*nIndex+4) = By2*Bx2*dBz1*Bu1*Bv1;
                    delta_dfz(32*nIndex+5) = By1*Bx1*dBz2*Bu1*Bv1;
                    delta_dfz(32*nIndex+6) = By2*Bx1*dBz2*Bu1*Bv1;
                    delta_dfz(32*nIndex+7) = By1*Bx2*dBz2*Bu1*Bv1;
                    delta_dfz(32*nIndex+8) = By2*Bx2*dBz2*Bu1*Bv1;
                    delta_dfz(32*nIndex+9) = By1*Bx1*dBz1*Bu2*Bv1;
                    delta_dfz(32*nIndex+10) = By2*Bx1*dBz1*Bu2*Bv1;
                    delta_dfz(32*nIndex+11) = By1*Bx2*dBz1*Bu2*Bv1;
                    delta_dfz(32*nIndex+12) = By2*Bx2*dBz1*Bu2*Bv1;
                    delta_dfz(32*nIndex+13) = By1*Bx1*dBz2*Bu2*Bv1;
                    delta_dfz(32*nIndex+14) = By2*Bx1*dBz2*Bu2*Bv1;
                    delta_dfz(32*nIndex+15) = By1*Bx2*dBz2*Bu2*Bv1;
                    delta_dfz(32*nIndex+16) = By2*Bx2*dBz2*Bu2*Bv1;

                    delta_dfz(32*nIndex+17) = By1*Bx1*dBz1*Bu1*Bv2;
                    delta_dfz(32*nIndex+18) = By2*Bx1*dBz1*Bu1*Bv2;
                    delta_dfz(32*nIndex+19) = By1*Bx2*dBz1*Bu1*Bv2;
                    delta_dfz(32*nIndex+20) = By2*Bx2*dBz1*Bu1*Bv2;
                    delta_dfz(32*nIndex+21) = By1*Bx1*dBz2*Bu1*Bv2;
                    delta_dfz(32*nIndex+22) = By2*Bx1*dBz2*Bu1*Bv2;
                    delta_dfz(32*nIndex+23) = By1*Bx2*dBz2*Bu1*Bv2;
                    delta_dfz(32*nIndex+24) = By2*Bx2*dBz2*Bu1*Bv2;
                    delta_dfz(32*nIndex+25) = By1*Bx1*dBz1*Bu2*Bv2;
                    delta_dfz(32*nIndex+26) = By2*Bx1*dBz1*Bu2*Bv2;
                    delta_dfz(32*nIndex+27) = By1*Bx2*dBz1*Bu2*Bv2;
                    delta_dfz(32*nIndex+28) = By2*Bx2*dBz1*Bu2*Bv2;
                    delta_dfz(32*nIndex+29) = By1*Bx1*dBz2*Bu2*Bv2;
                    delta_dfz(32*nIndex+30) = By2*Bx1*dBz2*Bu2*Bv2;
                    delta_dfz(32*nIndex+31) = By1*Bx2*dBz2*Bu2*Bv2;
                    delta_dfz(32*nIndex+32) = By2*Bx2*dBz2*Bu2*Bv2;
                    
                    %du
                    delta_dfu(32*nIndex+1) = By1*Bx1*Bz1*dBu1*Bv1;
                    delta_dfu(32*nIndex+2) = By2*Bx1*Bz1*dBu1*Bv1;
                    delta_dfu(32*nIndex+3) = By1*Bx2*Bz1*dBu1*Bv1;
                    delta_dfu(32*nIndex+4) = By2*Bx2*Bz1*dBu1*Bv1;
                    delta_dfu(32*nIndex+5) = By1*Bx1*Bz2*dBu1*Bv1;
                    delta_dfu(32*nIndex+6) = By2*Bx1*Bz2*dBu1*Bv1;
                    delta_dfu(32*nIndex+7) = By1*Bx2*Bz2*dBu1*Bv1;
                    delta_dfu(32*nIndex+8) = By2*Bx2*Bz2*dBu1*Bv1;
                    delta_dfu(32*nIndex+9) = By1*Bx1*Bz1*dBu2*Bv1;
                    delta_dfu(32*nIndex+10) = By2*Bx1*Bz1*dBu2*Bv1;
                    delta_dfu(32*nIndex+11) = By1*Bx2*Bz1*dBu2*Bv1;
                    delta_dfu(32*nIndex+12) = By2*Bx2*Bz1*dBu2*Bv1;
                    delta_dfu(32*nIndex+13) = By1*Bx1*Bz2*dBu2*Bv1;
                    delta_dfu(32*nIndex+14) = By2*Bx1*Bz2*dBu2*Bv1;
                    delta_dfu(32*nIndex+15) = By1*Bx2*Bz2*dBu2*Bv1;
                    delta_dfu(32*nIndex+16) = By2*Bx2*Bz2*dBu2*Bv1;

                    delta_dfu(32*nIndex+17) = By1*Bx1*Bz1*dBu1*Bv2;
                    delta_dfu(32*nIndex+18) = By2*Bx1*Bz1*dBu1*Bv2;
                    delta_dfu(32*nIndex+19) = By1*Bx2*Bz1*dBu1*Bv2;
                    delta_dfu(32*nIndex+20) = By2*Bx2*Bz1*dBu1*Bv2;
                    delta_dfu(32*nIndex+21) = By1*Bx1*Bz2*dBu1*Bv2;
                    delta_dfu(32*nIndex+22) = By2*Bx1*Bz2*dBu1*Bv2;
                    delta_dfu(32*nIndex+23) = By1*Bx2*Bz2*dBu1*Bv2;
                    delta_dfu(32*nIndex+24) = By2*Bx2*Bz2*dBu1*Bv2;
                    delta_dfu(32*nIndex+25) = By1*Bx1*Bz1*dBu2*Bv2;
                    delta_dfu(32*nIndex+26) = By2*Bx1*Bz1*dBu2*Bv2;
                    delta_dfu(32*nIndex+27) = By1*Bx2*Bz1*dBu2*Bv2;
                    delta_dfu(32*nIndex+28) = By2*Bx2*Bz1*dBu2*Bv2;
                    delta_dfu(32*nIndex+29) = By1*Bx1*Bz2*dBu2*Bv2;
                    delta_dfu(32*nIndex+30) = By2*Bx1*Bz2*dBu2*Bv2;
                    delta_dfu(32*nIndex+31) = By1*Bx2*Bz2*dBu2*Bv2;
                    delta_dfu(32*nIndex+32) = By2*Bx2*Bz2*dBu2*Bv2;
                   
                    %dv
                    delta_dfv(32*nIndex+1) = By1*Bx1*Bz1*Bu1*dBv1;
                    delta_dfv(32*nIndex+2) = By2*Bx1*Bz1*Bu1*dBv1;
                    delta_dfv(32*nIndex+3) = By1*Bx2*Bz1*Bu1*dBv1;
                    delta_dfv(32*nIndex+4) = By2*Bx2*Bz1*Bu1*dBv1;
                    delta_dfv(32*nIndex+5) = By1*Bx1*Bz2*Bu1*dBv1;
                    delta_dfv(32*nIndex+6) = By2*Bx1*Bz2*Bu1*dBv1;
                    delta_dfv(32*nIndex+7) = By1*Bx2*Bz2*Bu1*dBv1;
                    delta_dfv(32*nIndex+8) = By2*Bx2*Bz2*Bu1*dBv1;
                    delta_dfv(32*nIndex+9) = By1*Bx1*Bz1*Bu2*dBv1;
                    delta_dfv(32*nIndex+10) = By2*Bx1*Bz1*Bu2*dBv1;
                    delta_dfv(32*nIndex+11) = By1*Bx2*Bz1*Bu2*dBv1;
                    delta_dfv(32*nIndex+12) = By2*Bx2*Bz1*Bu2*dBv1;
                    delta_dfv(32*nIndex+13) = By1*Bx1*Bz2*Bu2*dBv1;
                    delta_dfv(32*nIndex+14) = By2*Bx1*Bz2*Bu2*dBv1;
                    delta_dfv(32*nIndex+15) = By1*Bx2*Bz2*Bu2*dBv1;
                    delta_dfv(32*nIndex+16) = By2*Bx2*Bz2*Bu2*dBv1;
                    
                    delta_dfv(32*nIndex+17) = By1*Bx1*Bz1*Bu1*dBv2;
                    delta_dfv(32*nIndex+18) = By2*Bx1*Bz1*Bu1*dBv2;
                    delta_dfv(32*nIndex+19) = By1*Bx2*Bz1*Bu1*dBv2;
                    delta_dfv(32*nIndex+20) = By2*Bx2*Bz1*Bu1*dBv2;
                    delta_dfv(32*nIndex+21) = By1*Bx1*Bz2*Bu1*dBv2;
                    delta_dfv(32*nIndex+22) = By2*Bx1*Bz2*Bu1*dBv2;
                    delta_dfv(32*nIndex+23) = By1*Bx2*Bz2*Bu1*dBv2;
                    delta_dfv(32*nIndex+24) = By2*Bx2*Bz2*Bu1*dBv2;
                    delta_dfv(32*nIndex+25) = By1*Bx1*Bz1*Bu2*dBv2;
                    delta_dfv(32*nIndex+26) = By2*Bx1*Bz1*Bu2*dBv2;
                    delta_dfv(32*nIndex+27) = By1*Bx2*Bz1*Bu2*dBv2;
                    delta_dfv(32*nIndex+28) = By2*Bx2*Bz1*Bu2*dBv2;
                    delta_dfv(32*nIndex+29) = By1*Bx1*Bz2*Bu2*dBv2;
                    delta_dfv(32*nIndex+30) = By2*Bx1*Bz2*Bu2*dBv2;
                    delta_dfv(32*nIndex+31) = By1*Bx2*Bz2*Bu2*dBv2;
                    delta_dfv(32*nIndex+32) = By2*Bx2*Bz2*Bu2*dBv2;
                    

                    nIndex = nIndex+1;
                end
            end
        end
    end
end
end


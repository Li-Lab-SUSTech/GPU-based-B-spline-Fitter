function delta_f = computeDelta3D_bSpline(sx,sy,sz)


delta_f = zeros(64,1);
nIndex = 0;
for q = 1:2
    Bz1 = evalBSpline(sz+q-1,3);
    Bz2 = evalBSpline(sz-q+0,3);
    
    for j = 1:2
        Bx1 = evalBSpline(sx+j-1,3);
        Bx2 = evalBSpline(sx-j+0,3);
        for i = 1:2
            By1=evalBSpline(sy+i-1,3);
            By2=evalBSpline(sy-i+0,3);
%             dxf(8*(q-1)*i*j+8*j*i+8*(i-1)+1)
            delta_f(8*nIndex+1) = By1*Bx1*Bz1;
            delta_f(8*nIndex+2) = By2*Bx1*Bz1;
            delta_f(8*nIndex+3) = By1*Bx2*Bz1;
            delta_f(8*nIndex+4) = By2*Bx2*Bz1;
            delta_f(8*nIndex+5) = By1*Bx1*Bz2;
            delta_f(8*nIndex+6) = By2*Bx1*Bz2;
            delta_f(8*nIndex+7) = By1*Bx2*Bz2;
            delta_f(8*nIndex+8) = By2*Bx2*Bz2;
            nIndex = nIndex+1;
            
        end
    end
end
alpha = 10:10:180;
z = linspace(-1000,1000,size(FixedPSF,3)+1)+20;
f = FixedPSF(:,:,[20 40 60 80],[1 3 5 7]);
f=reshape(f,31,31,16);
alpha1 = alpha([1 3 5 7]);alpha1 = repmat(alpha1',1,4)';
z1 = repmat(z([20 40 60 80])',1,4);
xN = 4;
yN = 4;
h = figure;
h.Position = [200,300,100*xN,105*yN];
for n = 1:16
    if (n<=xN*yN)
        ha = axes;
        ii = mod(n,xN);
        jj = floor(n/xN)+1;
        if ii == 0
            ii = xN;
            jj = n/xN;
        end
        ha.Position = [(ii-1)/xN,(yN-jj)/yN,1/xN,1/yN];
        imshow(f(:,:,n),[]);axis off;axis equal;
        text(2,3, ['\alpha=',num2str(alpha1(n),3),'^o'],'color',[1,1,1],'fontsize',10);
        text(2,size(f,2)-3, ['z=',num2str(z1(n),3),'nm'],'color',[1,1,1],'fontsize',12);
    end
end

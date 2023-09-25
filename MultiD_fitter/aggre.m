function [out] = aggre(data)

ind=1;
D=zeros(size(data,1)-1);
for ii=1:size(data,1)-1
    for jj=ii:size(data,1)
        D(ii,jj)=sqrt(data(ii,1)-data(jj,1))^2+(data(ii,2)-data(jj,2))^2;
    end
end






end


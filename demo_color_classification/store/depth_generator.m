function [output] = depth_generator(Npixels,Nmol,Depth,depNum)
%Npixels is pixel number in camera,Nmol is distance of molecule to
%coverslip,deltobjStage is position relative to focus at coverslip
%example input = depth_generator(13,100,1000,100);
output=zeros(Npixels,Npixels,Nmol,depNum);
depth=linspace(2000,2000+1*Depth,depNum);
for i=1:depNum
    
    output(:,:,:,i)=vector_test(depth(i),Npixels,Nmol);
    
end

end


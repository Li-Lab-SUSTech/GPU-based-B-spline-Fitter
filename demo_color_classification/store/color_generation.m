function [output1] = color_generation(lambda,Npixels,Nmol,Ncolor,PSFtype,pixelsz)
%Npixels is pixel number in camera,Nmol is distance of molecule to
%coverslip,deltobjStage is position relative to focus at coverslip
%example input = depth_generator(13,100,1000,100);
output1=zeros(Npixels,Npixels,Nmol,Ncolor);
% output2=zeros(Npixels,Npixels,Nmol,depNum);

color=linspace(lambda(1),lambda(2),Ncolor);
for i=1:Ncolor
    
    output1(:,:,:,i)=vector_test_color(color(i),Npixels,Nmol,0,PSFtype,pixelsz);
%     output2(:,:,:,i)=vector_test_color(1,depth(i),Npixels,Nmol,delt);

end

end
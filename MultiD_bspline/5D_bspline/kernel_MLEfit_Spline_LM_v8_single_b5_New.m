function [P,update,fmodel] =  kernel_MLEfit_Spline_LM_v8_single_b5_New(d_data,b5,initP,sz,iterations)
% update computeDelta3Dj and fAt3Dj. Add kernel_derivativeSpline
% with bSpline
% with only one coeff
 update = [];
pi = single(3.141592);
% spline_xsize = size(coeff,1);
% spline_ysize = size(coeff,1);
% spline_zsize = size(coeff,1);
dataSize = b5.dataSize;
%b5.coeffs(end+1,end+1,end+1,end+1,end+1)=0;
PSFSigma = single(1.5);
xc = single(0);
yc = single(0);
zc = single(0);
uc = single(0);
vc = single(0);

NV = 7;% x,y,z,photon.bg,u,v,
fmodel=zeros(size(d_data));
dudt = single(zeros(sz,sz,NV));

newTheta = single(zeros(NV,1));
oldTheta = newTheta;

M = zeros(NV*NV,1);
Minv = zeros(NV*NV,1);

Nfits = size(d_data,3);
CRLB = zeros(Nfits,NV);
RUNNING = 0;
CONVERGED = 1;
CHOLERRER = 2;
BADPEAK = 3;
tolerance=1e-6;
%%
for tx = 1:Nfits
    tx
    %newpeak
    [newTheta(1),newTheta(2)]=kernel_CenterofMass2D(sz,single(d_data(sz*sz*(tx-1)+1:sz*sz*(tx))));
    [Nmax,newTheta(4)] = kernel_GaussFMaxMin2D(sz,PSFSigma,single(d_data(sz*sz*(tx-1)+1:sz*sz*(tx))));
    newTheta(3)=max(0,(Nmax-newTheta(4))*2*pi*PSFSigma*PSFSigma);
    
    if ~isempty(initP)
        newTheta(5:7)=initP;
    else
        newTheta(5) = dataSize(3)/2;
        newTheta(6) = dataSize(4)*1/2;
        newTheta(7) = dataSize(5)*1/2;
    end
    newLambda = single(1);
    newStatus = RUNNING;
    newSign = zeros(NV,1);
    newUpdate = zeros(NV,1);
    newClamp = [1 1 10000 20 dataSize(3)/10 dataSize(4)/10 dataSize(5)/10];
    fade = [0.5 0.5 0.4 0.5 0.5 0.5 0.5];
    %resetFit
    newFit = single(zeros(sz,sz));
    fit = single(zeros(sz,sz));
    
    %updateFitValues3D
    %     xc = single(-2*(newTheta(1) - 6.5+0.5));
    %     yc = single(-2*(newTheta(2) - 6.5+0.5));
    %     zc = single(newTheta(5)-floor(newTheta(5)));
    xc = single(7-newTheta(1));
    yc = single(7-newTheta(2));
    zc = single(newTheta(5));
    uc = single(newTheta(6));
    vc = single(newTheta(7));
    off = (dataSize(1)- sz)/2;
    %     xstart = xc;
    %     ystart = yc;
    %     zstart = zc;
    %
    %     xs= xstart-floor(xstart);
    %     ys= ystart-floor(ystart);
    %     zs= zstart-floor(zstart);
    
    xs= single(xc-floor(xc));
    ys= single(yc-floor(yc));
    zs= single(zc-floor(zc));
    us= single(uc-floor(uc));
    vs= single(vc-floor(vc));
    %     xstart = floor(xc);
    %     xc = xc-floor(xc);
    %
    %     ystart = floor(yc);
    %     yc = yc-floor(yc);
    
    
    
    newErr = 0;
    jacobian = single(zeros(NV,1));
    hessian = single(zeros(NV,NV));
    [delta_f,delta_dfx,delta_dfy,delta_dfz,delta_dfu,delta_dfv] = Fan_computeDelta5D_bSpline(xs,ys,zs,us,vs);
%     [delta_f,delta_dfx,delta_dfy,delta_dfz] = computeDelta3D_bSpline_v2_single(0.5,0.5,0.5);
    
    for ii = 0:sz-1
        for jj = 0:sz-1

            data = single(d_data(sz*sz*(tx-1)+sz*jj+ii+1));

            [newDudt,model] = kernel_Derivative_bSpline5D(xc+jj+off,yc+ii+off,zc,uc,vc,dataSize(1),dataSize(2),dataSize(3),dataSize(4),dataSize(5),b5.coeffs,delta_f,delta_dfx,delta_dfy,delta_dfz,delta_dfu,delta_dfv,newTheta);% nwe

            
            
            
            test(ii+1,jj+1)=newDudt(3);
            if data>0
                newErr = newErr +2*((model-data)-data*log(model/data));
            elseif data ==0
                newErr = newErr + 2*model;
            end
            
            newDudt(6) = newDudt(5);%remove it
            t1 = single(1-data/model);
            for l = 1:NV
                jacobian(l) = jacobian(l)+t1*newDudt(l);
            end
            t2 = data/model^2;
            
            for l = 0:NV-1
                for m =l:NV-1
                    hessian(l*NV+m+1) = hessian(l*NV+m+1)+t2*newDudt(l+1)*newDudt(m+1);
                    hessian(m*NV+l+1) = hessian(l*NV+m+1);
                end
            end
            
        end
    end
    %         dipshow(newDudt1);
    %     newErr = err;
    oldErr = 1e13;
    %addPeak
    
    %copyFitData
    
    oldClamp = newClamp;
    oldLambda = newLambda;
    oldSign = newSign;
    oldTheta = newTheta;
    
%     turn = 1;
    
    for kk =1:iterations
        
        
        
        %calcError
        %             err = 0;
        %             for ii = 0:sz-1
        %                 for jj = 0:sz-1
        %                     model = newFit(ii+1,jj+1);%fit to newFit
        %                     data = single(d_data(sz*sz*(tx-1)+sz*jj+ii+1));
        % %                     residual(ii+1,jj+1,kk)=model-data;
        %                     err = err +2*((model-data)-data*log(model/data));
        %
        %                 end
        %             end
        %             newErr = err;
        %             errout(tx,1)=err;
        newErr-oldErr;
        if abs((newErr-oldErr)/newErr)<tolerance
            %                 newStatus = CONVERGED;
            break;
        else
            if newErr>1.5*oldErr
                %subtractPeak
                fit = fit-newFit;
                %copyfitdata
                
                
                newClamp = oldClamp;
                newLambda = oldLambda;
                
                newSign = oldSign;
                
                newTheta = oldTheta;
                newErr = oldErr;
                
                
                newLambda = 10*newLambda;
                %addpeak
                
                
            elseif newErr<oldErr
                if newLambda >1
                    newLambda = newLambda*0.9;
                elseif newLambda <1
                    newLambda = 1;
                end
            end
            
           
            for i = 0:NV-1
                hessian(i*NV+i+1) = hessian(i*NV+i+1)*newLambda;
            end
            
            
            
            
            [L U info] = kernel_cholesky(hessian,NV);
            if info ==0
                newUpdate = kernel_luEvaluate(L,U,jacobian,NV);
                fit = fit -newFit;
                %newUpdate(6)=newUpdate(6)*10;
                %copyFitData
                
                oldClamp = newClamp;
                oldLambda = newLambda;
                
                oldSign = newSign;
                
                oldTheta = newTheta;
                oldErr = newErr;
                
%                 if (abs(newUpdate(5))<5e-3) && turn 
%                     newClamp(6) = 0.2;oldClamp(6) = 0.2;turn = 0;
%                     newClamp(5) = 2;oldClamp(5) = 2;
%                 end
                    
                %updatePeakParameters
                for ll =1:NV
                    if newSign(ll)~=0
                        if newSign(ll)==1&&newUpdate(ll)<0
                            newClamp(ll)=newClamp(ll)*fade(ll);
                        elseif newSign(ll)==-1&&newUpdate(ll)>0
                            newClamp(ll)=newClamp(ll)*fade(ll);
                        end
                        
                    end
                    
                    
                    
                    if newUpdate(ll)>0
                        newSign(ll)=1;
                    else
                        newSign(ll)=-1;
                    end
                    
                    newTheta(ll) = newTheta(ll)-newUpdate(ll)/(1+abs(newUpdate(ll)/newClamp(ll)));
%                     update(kk,ll) = newUpdate(ll)/(1+abs(newUpdate(ll)/newClamp(ll)));
                end
%                 newTheta-oldTheta;
                update(kk,1) = newUpdate(2)/(1+abs(newUpdate(2)/newClamp(2)));
                update(kk,2) = newTheta(2);
                update(kk,3)= oldErr;
%                 update(kk,NV+1)= oldErr;

                newTheta(3) = max(newTheta(3),1);
                newTheta(4) = max(newTheta(4),0.01);
                newTheta(5) = max(newTheta(5),0);
                newTheta(5) = min(newTheta(5),dataSize(3));
                
%                 newTheta(6) = max(newTheta(6),0);
%                 newTheta(6) = min(newTheta(6),dataSize(4));
                if newTheta(6)<0
                    newTheta(6)=dataSize(4)+newTheta(6);
                elseif newTheta(6)>dataSize(4)
                    newTheta(6)=-dataSize(4)+newTheta(6);
                end
                
                newTheta(7) = max(newTheta(7),0);
                newTheta(7) = min(newTheta(7),dataSize(5)-1);
                
                %updateFitValues3D
                xc = -newTheta(1)+6+1;
                yc = -newTheta(2)+6+1;
                zc = newTheta(5);
                uc = newTheta(6);
                vc = newTheta(7);

                off = (dataSize(1)- sz)/2;

                
                xs= xc-floor(xc);
                ys= yc-floor(yc);
                zs= zc-floor(zc);
                us= uc-floor(uc);
                vs= vc-floor(vc);
                
                
                
                
                
                newErr = 0;
                jacobian = single(zeros(NV,1));
                hessian = single(zeros(NV,NV));
               
                [delta_f,delta_dfx,delta_dfy,delta_dfz,delta_dfu,delta_dfv] = Fan_computeDelta5D_bSpline(xs,ys,zs,us,vs);
                for ii = 0:sz-1
                    for jj = 0:sz-1
                        %[newDudt,model] = kernel_Derivative_bSpline(xstart+jj+off,ystart+ii+off,zstart,dataSize(1),dataSize(2),dataSize(3),b4.coeffs,delta_f,delta_dfx,delta_dfy,delta_dfz,newTheta);
                        [newDudt,model] = kernel_Derivative_bSpline5D(xc+jj+off,yc+ii+off,zc,uc,vc,dataSize(1),dataSize(2),dataSize(3),dataSize(4),dataSize(5),b5.coeffs,delta_f,delta_dfx,delta_dfy,delta_dfz,delta_dfu,delta_dfv,newTheta);% nwe
                        test(ii+1,jj+1)=model;
                        data = single(d_data(sz*sz*(tx-1)+sz*jj+ii+1));
                        %                         [temp,dfx,dfy,dfz] = fAt3D_bs_v6(xstart+jj+off,ystart+ii+off,zstart,dataSize(1),dataSize(2),dataSize(3),b4.coeffs,delta_f,delta_dfx,delta_dfy,delta_dfz);
                        %                         model = single(newTheta(4)+newTheta(3)*temp);
                        %                         newDudt(1) = -1*newTheta(3)*dfy;
                        %                         newDudt(2) = -1*newTheta(3)*dfx;
                        %                         newDudt(5) = newTheta(3)*dfz;
                        %                         newDudt(3) = temp;
                        %                         newDudt(4) = 1;
                        fmodel(sz*sz*(tx-1)+sz*jj+ii+1)=model;
                        if data>0
                            newErr = newErr +2*((model-data)-data*log(model/data));
                        elseif data ==0
                            newErr = newErr + 2*model;
                        end
                        
                        t1 = single(1-data/model);
                        for l = 1:NV
                            jacobian(l) = jacobian(l)+t1*newDudt(l);
                        end
                        t2 = data/model^2;
                        
                        for l = 0:NV-1
                            for m =l:NV-1
                                hessian(l*NV+m+1) = hessian(l*NV+m+1)+t2*newDudt(l+1)*newDudt(m+1);
                                hessian(m*NV+l+1) = hessian(l*NV+m+1);
                            end
                        end
                        
                    end
                end
                %                                 dipshow(newDudt1);
            else
                newLambda = 10*newLambda;
                disp('CHOLERRER')
            end
            %                 kk
            
            
        end
        
    end
    iteration = kk;
    if isempty(kk)
        iteration=0;
    end
    for kk = 1:NV
        P(Nfits*(kk-1)+tx) = newTheta(kk);
    end
%     CRLB(tx,:) = sqrt(diag(inv(hessian)))';
    P(Nfits*(NV)+tx) = iteration;
    for ii = 0:sz-1
        for jj = 0:sz-1
            %[newDudt,model] = kernel_Derivative_bSpline(xstart+jj+off,ystart+ii+off,zstart,dataSize(1),dataSize(2),dataSize(3),b4.coeffs,delta_f,delta_dfx,delta_dfy,delta_dfz,newTheta);
            [~,model] = kernel_Derivative_bSpline5D(xc+jj+off,yc+ii+off,zc,uc,vc,dataSize(1),dataSize(2),dataSize(3),dataSize(4),dataSize(5),b5.coeffs,delta_f,delta_dfx,delta_dfy,delta_dfz,delta_dfu,delta_dfv,newTheta);% nwe
            fmodel(ii+1,jj+1,tx)=model;
        end
    end


end
P = reshape(P,Nfits,NV+1);
% update = 0;









































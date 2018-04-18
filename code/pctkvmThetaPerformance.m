%% Theta Performance
% This script shows the performance of the PCTKVM over the Reuters dataset
% with varying thetas.
% Output is plotted and saved as matrix 



addpath(genpath('../libsvm/matlab'));
addpath(genpath('../data'));
addpath(genpath('../result'));
addpath(genpath('../code'));

clear all;

%% Reuters Dataset
options.ker = 'rbf';      % TKL: kernel: 'linear' | 'rbf' | 'lap'
options.eta = 2.0;           % TKL: eigenspectrum damping factor
accResult = [];
for strData = {'org_vs_people','org_vs_place', 'people_vs_place'} %
    accTmp = [];
    for iData = 1:2
         accTmp = [];
        for theta = {0.1,0.5,1,1.5,2,2.5,3,5,10}
            options.theta = theta{1};
            data = char(strData);
            data = strcat(data, '_', num2str(iData));
            load(strcat('../data/Reuters/', data));
            
            fprintf('data=%s\n', data);
            
            Xs=bsxfun(@rdivide, bsxfun(@minus,Xs,mean(Xs)), std(Xs));
            Xt=bsxfun(@rdivide, bsxfun(@minus,Xt,mean(Xt)), std(Xt));
            
            
            model = pctkvm_train(Xs',Ys,Xt',options);
            [erate, nvec, label, y_prob] = pctkvm_predict(Ys,Yt,model);
            
            erate = erate*100;
            acc = 100-erate;
            
            accTmp = [accTmp; acc];
            fprintf('\nPCTKVM %.2f%% \n', acc);
        end
        accResult = [accResult, accTmp];
    end
end
figure;  hold on; plot([0.1,0.5,1,1.5,2,2.5,3,5,10],accResult(:,:)); legend('orgs vs people ','people vs orgs','orgs vs places','places vs orgs','people vs places','places vs people'); ylabel('Accuracy in %'); xlabel('Width of Gaussian Kernel'); hold off;
name = strcat('../result/pctkvmThetaPerformance_Reuters'.mat');
save(name,'accResult');

clear all;
%% OFFICE vs CALLTECH-256 Dataset
options.ker = 'rbf';         % TKL: kernel: 'linear' | 'rbf' | 'lap'
options.eta = 1.1;           % TKL: eigenspectrum damping factor

srcStr = {'Caltech10', 'Caltech10', 'Caltech10', 'amazon', 'amazon', 'amazon', 'webcam', 'webcam', 'webcam', 'dslr', 'dslr', 'dslr'};
tgtStr = {'amazon', 'webcam', 'dslr', 'Caltech10', 'webcam', 'dslr', 'Caltech10', 'amazon', 'dslr', 'Caltech10', 'amazon', 'webcam'};
accResult = [];
for iData = 1:12
    accTmp = [];
    for theta = {0.1,0.5,1,1.5,2.5,3,5,10}
        options.theta = theta{1};
        src = char(srcStr{iData});
        tgt = char(tgtStr{iData});
        data = strcat(src, '_vs_', tgt);
        
        load(['../data/OfficeCaltech/' src '_SURF_L10.mat']);
        fts = fts ./ repmat(sum(fts, 2), 1, size(fts, 2));
        Xs = zscore(fts, 1);
        Ys = labels;
        
        
        load(['../data/OfficeCaltech/' tgt '_SURF_L10.mat']);
        fts = fts ./ repmat(sum(fts, 2), 1, size(fts,2));
        Xt = zscore(fts, 1);
        Yt = labels;
        
        
        fprintf('data=%s\n', data);
        
        
        model = pctkvm_train(Xs,Ys,Xt,options);
        [erate, nvec, label, y_prob] = pctkvm_predict(Ys,Yt,model);
        
        erate = erate*100;
        acc = 100-erate;
        accTmp = [accTmp; acc];
        accResult = [accResult, accTmp];
        fprintf('\nPCTKVM %.2f%% \n', acc);
    end
    name = strcat('../result/pctkvmThetaPerformance_',char(strData),'.mat');
    save(name,'accResult');
end


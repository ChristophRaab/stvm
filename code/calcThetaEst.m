function [] = calcThetaEst()
%CALCTHETAEST: This function calculates the estimate of the theta used for the PCTKVM
% for the test datasets Reuters and the Image datasets
% caltech,webcam,dslr,amazon.

addpath(genpath('../libsvm/matlab'));
addpath(genpath('../data'));
addpath(genpath('../estThetas'));
addpath(genpath('../code'));

clear all;
estThetas = [];

for strData = {'org_vs_people','org_vs_place', 'people_vs_place'} %
    
    for iData = 1:2
        
        data = char(strData);
        data = strcat(data, '_', num2str(iData));
        load(strcat('../data/Reuters/', data));
        
        fprintf('data=%s\n', data);
        
        Xs=bsxfun(@rdivide, bsxfun(@minus,Xs,mean(Xs)), std(Xs));
        Xt=bsxfun(@rdivide, bsxfun(@minus,Xt,mean(Xt)), std(Xt));
        X = [Xs, Xt];
        theta = thetaEstimation(Xs);
        estThetas = [estThetas; theta];
    end
end

srcStr = {'Caltech10', 'Caltech10', 'Caltech10', 'amazon', 'amazon', 'amazon', 'webcam', 'webcam', 'webcam', 'dslr', 'dslr', 'dslr'};
tgtStr = {'amazon', 'webcam', 'dslr', 'Caltech10', 'webcam', 'dslr', 'Caltech10', 'amazon', 'dslr', 'Caltech10', 'amazon', 'webcam'};
for iData = 1:12
    
    src = char(srcStr{iData});
    tgt = char(tgtStr{iData});
    data = strcat(src, '_vs_', tgt);
    
    load(['../data/OfficeCaltech/' src '_SURF_L10.mat']);
    fts = fts ./ repmat(sum(fts, 2), 1, size(fts, 2));
    Xs = zscore(fts, 1);
    
    
    load(['../data/OfficeCaltech/' tgt '_SURF_L10.mat']);
    fts = fts ./ repmat(sum(fts, 2), 1, size(fts,2));
    Xt = zscore(fts, 1);
    X = [Xs', Xt'];
    theta = thetaEstimation(X);
    estThetas = [estThetas; theta];
    fprintf('data=%s\n', data);
end
save('../result/thetaEstResult','estThetas');
end
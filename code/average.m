%% Average Test Script
% This script calculates the error, accuracy and AUC values over the 15
% subsets of selected transfer learning methods. This will be repeated 10
% times to get an additional standard deviation. The datasets are generated
% from preprocessed versions of Reuters-21578, Office and
% Caltech-256.
% Optional: For parallel computing uncomment parfor in line 38, 85


addpath(genpath('../libsvm/ma tlab'));
addpath(genpath('../data'));
addpath(genpath('../result'));
addpath(genpath('../code'));

clear all;

%% Reuters Dataset
options.ker = 'rbf';      % TKL: kernel: 'linear' | 'rbf' | 'lap'
options.eta = 2.0;           % TKL: eigenspectrum damping factor
options.gamma = 1;         % TKL: width of gaussian kernel
options.k = 100;              % JDA: subspaces bases
options.lambda = 1.0;        % JDA: regularization parameter
options.svmc = 10.0;         % SVM: complexity regularizer in LibSVM
options.g = 40;              % GFK: subspace dimension
options.tcaNv = 50;          % TCA: numbers of Vectors after reduction
options.subspaceDim = 80;   %SA: Subspace Dimension
options.theta = -1;
testSize= 5;


for strData = {'org_vs_people','org_vs_place', 'people_vs_place'} %
    
    accResult = [];
    errResult = [];
    aucResult = [];
    timeResult = [];
    nvecResult = [];
    %     parfor (i=1:testSize,2)
    for iData = 1:2
        for i=1:testSize
            data = char(strData);
            data = strcat(data, '_', num2str(iData));
            load(strcat('../data/Reuters/', data));
            
            fprintf('data=%s\n', data);
            
            Xs=bsxfun(@rdivide, bsxfun(@minus,Xs,mean(Xs)), std(Xs));
            Xt=bsxfun(@rdivide, bsxfun(@minus,Xt,mean(Xt)), std(Xt));
            
            [accTmp,errTmp,aucTmp,timeTmp,nvecTmp] = tlfun(Xs,Xt,Ys,Yt,options);
            accResult = [accResult; accTmp]; errResult =[errResult; errTmp]; aucResult = [aucResult; aucTmp]; timeResult = [timeResult; timeTmp]; nvecResult = [nvecResult; nvecTmp];
            
        end
        ormse = sqrt(mean(errResult.^2));
        name = strcat('../result/average/average_Only',data,'.mat');
        save(name,'errResult','accResult','aucResult','timeResult','nvecResult','ormse');
        
    end
end

clear all;
%% OFFICE vs CALLTECH-256 Dataset
options.ker = 'rbf';         % TKL: kernel: 'linear' | 'rbf' | 'lap'
options.eta = 1.1;           % TKL: eigenspectrum damping factor
options.gamma = 1.0;         % TKL: width of gaussian kernel
options.g = 30;              % GFK: subspace dimension
options.k = 100;              % JDA: subspaces bases
options.lambda = 1.0;        % JDA: regularization parameter
options.svmc = 10.0;         % SVM: complexity regularizer in LibSVM
options.tcaNv = 50;          % TCA: numbers of Vectors after reduction
options.subspaceDim = 80;   %SA: Subspace Dimension
options.theta = 1;


srcStr = {'Caltech10', 'Caltech10', 'Caltech10', 'amazon', 'amazon', 'amazon', 'webcam', 'webcam', 'webcam', 'dslr', 'dslr', 'dslr'};
tgtStr = {'amazon', 'webcam', 'dslr', 'Caltech10', 'webcam', 'dslr', 'Caltech10', 'amazon', 'dslr', 'Caltech10', 'amazon', 'webcam'};

for iData = 1:12
    accResult = [];
    errResult = [];
    aucResult = [];
    timeResult = [];
    nvecResult = [];
    
    %     parfor (i=1:testSize,2)
    for i=1:testSize
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
        
        [accTmp,errTmp,aucTmp,timeTmp,nvecTmp] = tlfun(Xs',Xt',Ys,Yt,options);
        accResult = [accResult; accTmp]; errResult =[errResult; errTmp]; aucResult = [aucResult; aucTmp]; timeResult = [timeResult; timeTmp]; nvecResult = [nvecResult; nvecTmp];
        
    end
    ormse = sqrt(mean(errResult.^2));
    
    name = strcat('../result/average/average_Only',data,'.mat');
    save(name,'ormse','errResult','accResult','aucResult','timeResult','nvecResult');
end


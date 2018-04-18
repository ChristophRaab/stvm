function [ accTmp,errTmp,aucTmp,timeTmp,nvecTmp ] = tlfun( Xs,Xt,Ys,Yt,options)
% TLFUN: Function for the test script to encapsulate the classifier.
% It trains and evaluates the classifier and return the results.
% ------------------------------------------------------------------------
%INPUT: 
% Xs - Source Data
% Xt - Target Data
% Ys - Source Label
% Yt - Target Label
% options - Struct for the classifier parameters. For example, the cost
% Parameter C is given with: options.svmc = 10
%OUTPUT: 
% accTmp - The test accuracy of the classifier for the given data
% errTmp - The test error
% aucTmp - The AUC value. Note 1 is positive class. GFK has no prob. est.
% timeTmp - Needed time 
% nvecTmp - Used support vectors. Note GFK uses no vector machine

accTmp = [];
errTmp = [];
aucTmp = [];
timeTmp = [];
nvecTmp = [];


m = size(Xs, 2);
n = size(Xt, 2);

if strcmp(options.ker,'linear')
    libKer = 1;
end

if strcmp(options.ker,'rbf')
    libKer = 2;
end

 
% %% SVM
% tic;
% K = kernel(options.ker, [Xs, Xt], [],options.gamma);
% model = svmtrain(full(Ys), [(1:m)', K(1:m, 1:m)], ['-c ', num2str(options.svmc), ' -t 4 -q 1']);
% [label, acc,scores] = svmpredict(full(Yt), [(1:n)', K(m+1:end, 1:m)], model);
% t = toc;
% [Xauc,Yauc,T,AUC] = perfcurve(Yt,scores(:,1),'1');
% 
% accTmp = [accTmp acc(1)]; %Accuracy
% errTmp = [errTmp 100 - acc(1)]; % Error rate
% nvecTmp = [nvecTmp sum(model.nSV)]; % Number of vectors
% aucTmp  = [aucTmp AUC*100]; % AUC OF ROC
% timeTmp = [timeTmp t]; % Time needed
% 
% fprintf('SVM = %0.4f\n', acc(1));
% 
% %% PCVM
% 
% tic;
% model = pcvm_train(Xs',Ys,options.gamma);
% [erate, nvec, label, y_prob] = pcvm_predict(Xs',Ys,Xt',Yt,model);
% 
% t = toc;
% erate = erate*100;
% acc = 100-erate;
% [Xauc,Yauc,T,AUC] = perfcurve(Yt,y_prob,'1');
% 
% accTmp = [accTmp acc];
% errTmp = [errTmp erate];
% aucTmp = [aucTmp AUC*100];
% nvecTmp = [nvecTmp nvec];
% timeTmp = [timeTmp t];
% 
% fprintf('\nPCVM %.2f%% \n', acc)
% 
% %% TCA
% nt = length(Ys);
% mt = length(Yt);
% tic;
% K = tca(Xs',Xt',options.tcaNv,options.gamma,options.ker);
% model = svmtrain(full(Ys),[(1:nt)',K(1:nt,1:nt)],['-s 0 -c ', num2str(options.svmc), ' -t 4 -q 1']);
% [label,acc,scores] = svmpredict(full(Yt),[(1:mt)',K(nt+1:end,1:nt)],model);
% t = toc;
% [Xauc,Yauc,T,AUC] = perfcurve(Yt,scores(:,1),'1');
% 
% accTmp = [accTmp acc(1)]; 
% errTmp = [errTmp 100 - acc(1)]; 
% nvecTmp = [nvecTmp sum(model.nSV)]; 
% aucTmp  = [aucTmp AUC*100]; 
% timeTmp = [timeTmp t];
% fprintf('\nTCA=%0.4f\n',acc(1));
% 
% %% JDA
% 
% Cls = [];
% % tic;
% [Z,A] = JDA(Xs,Xt,Ys,Cls,options);
% Z = Z*diag(sparse(1./sqrt(sum(Z.^2))));
% Zs = Z(:,1:size(Xs,2));
% Zt = Z(:,size(Xs,2)+1:end);
% K = kernel(options.ker, Z, [],options.gamma);
% model = svmtrain(full(Ys), [(1:m)', K(1:m, 1:m)], ['-c ', num2str(options.svmc), ' -t 4 -q 1']);
% [label, acc,scores] = svmpredict(full(Yt), [(1:n)', K(m+1:end, 1:m)], model);
% t = toc;
% [Xauc,Yauc,T,AUC] = perfcurve(Yt,scores(:,1),'1');
% 
% accTmp = [accTmp acc(1)]; 
% errTmp = [errTmp 100 - acc(1)]; 
% nvecTmp = [nvecTmp sum(model.nSV)]; 
% aucTmp  = [aucTmp AUC*100];
% timeTmp = [timeTmp t];
% fprintf('\nJDA=%0.4f\n',acc(1));
% 
% %% GFK
% tic;
% xs = full(Xs');
% xt = full(Xt');
% Ps = pca(xs);
% Pt = pca(xt);
% nullP = null(Ps');
% 
% G = GFK([Ps,nullP], Pt(:,1:options.g));
% [label, acc] = my_kernel_knn(G, xs, Ys, xt, Yt);
% acc = full(acc)*100;
% t = toc;
% accTmp =[accTmp acc];
% errTmp =[errTmp 100 - acc];
% nvecTmp =[nvecTmp 0];
% timeTmp = [timeTmp t];
% fprintf('\nGFK=%0.4f\n',acc);
% 
% 
% %% TKL SVM
% tic;
% K = TKL(Xs, Xt, options);
% model = svmtrain(full(Ys), [(1:m)', K(1:m, 1:m)], ['-s 0 -c ', num2str(options.svmc), ' -t 4 -q 1']);
% [labels, acc,scores] = svmpredict(full(Yt), [(1:n)', K(m+1:end, 1:m)], model);
% t = toc;
% [Xauc,Yauc,T,AUC] = perfcurve(Yt,scores(:,1),'1');
% 
% accTmp = [accTmp acc(1)];
% errTmp = [errTmp 100 - acc(1)];
% nvecTmp = [nvecTmp sum(model.nSV)]; 
% aucTmp  = [aucTmp AUC*100]; 
% timeTmp = [timeTmp t];
% fprintf('\nTKL=%0.4f\n',acc(1));



%% PCTLVM No Theta Est

tic;
options.theta =2;
model = pctkvm_train(Xs',Ys,Xt',options);
[erate, nvec, label, y_prob] = pctkvm_predict(Ys,Yt,model);
t = toc;
erate = erate*100;
acc = 100-erate;
[Xauc,Yauc,T,AUC] = perfcurve(Yt,y_prob,'1');


accTmp = [accTmp acc];
errTmp = [errTmp erate];
aucTmp = [aucTmp AUC*100];
nvecTmp = [nvecTmp nvec];
timeTmp = [timeTmp t];
fprintf('\nPCTKVM_Theta %.2f%% \n', acc);

%% PCTLVMTheta Est
options.theta =-1;
tic;
model = pctkvm_train(Xs',Ys,Xt',options);
[erate, nvec, label, y_prob] = pctkvm_predict(Ys,Yt,model);
t = toc;
erate = erate*100;
acc = 100-erate;
[Xauc,Yauc,T,AUC] = perfcurve(Yt,y_prob,'1');


accTmp = [accTmp acc];
errTmp = [errTmp erate];
aucTmp = [aucTmp AUC*100];
nvecTmp = [nvecTmp nvec];
timeTmp = [timeTmp t];
fprintf('\nPCTKVM %.2f%% \n', acc);


end


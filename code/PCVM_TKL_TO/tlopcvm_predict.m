function [erate, nvec, y_sign, y_prob] = tlopcvm_predict(trainY,K,testY,w,b,used)

weights = w(used).*trainY(used);

% Compute RVM over test data and calculate error
PHI	= K(:,used);
test_num = size(K,1);

Y_regress = PHI*weights+b*ones(test_num,1);
y_sign	= sign(Y_regress);

% the probablistic output
y_prob = normcdf(Y_regress); 

errs	= sum(y_sign(testY== -1)~=-1) + sum(y_sign(testY==1)~=1);
erate = errs/test_num;
nvec = length(used);


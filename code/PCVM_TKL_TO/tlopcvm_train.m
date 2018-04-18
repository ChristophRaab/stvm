function [w,b,used,theta,errlog,K] = tlopcvm_train(trainX,trainY,testX)
% Implementations of Probabilistic Classification Vector Machines.
% With Theta Opt
% The Algorithm is presented in the following paper:
% Huanhuan Chen, Peter Tino and Xin Yao. Probabilistic Classification Vector Machines. 
% IEEE Transactions on Neural Networks. vol.20, no.6, pp.901-914, June 2009. 
%	Copyright (c) Huanhuan Chen
% Simple integration of the TKL algorithm without solving the issue of theta
% optimization.
%NOTE: No multi-class option available

% the maximal iterations
niters = 500;

pmin=10^-5;
errlog = zeros(1, niters);

ndata= size(trainX,1);

display = 0; % can be zero

% Initial weight vector to let w to be large than zero
w = rand(ndata,1)+ 0.2;

% Initial bias b
b = randn; 

% initialize the auxiliary variables Ht to follow the target labels of the training set
Ht = 10*rand(ndata,1).*trainY + rand(ndata,1);

% Threshold to determine whether this is small
w_minimal = 1e-3;

% Threshold for convergence 
threshold = 1e-3;

% all one vector
I = ones(ndata,1);

y = trainY;

% active vector indicator
nonZero = ones(ndata,1);

% non-zero wegith vector
w_nz = w(logical(nonZero));

wold = w;

repy=repmat(trainY(:)', ndata, 1);

if display
    number_of_RVs = zeros(niters,1);
end

theta = 1.0;
% TKL Parameters
options.ker = 'linear';     % kernel: 'linear' | 'rbf' | 'lap'
options.eta = 2.0;       % eigenspectrum damping factor
options.theta = theta;
% Main loop of algorithm
for n = 1:niters
%     fprintf('\n%d. iteration.\n',n);
    options.theta = theta;
    K = TKL(trainX', testX', options); 

    % Take the link upper square of the K Matrix for the learning algorithm
    Kl = K(1:ndata,1:ndata);
  
    % Note that theta^2
    % scale columns of kernel matrix with label trainY
    Ky = Kl.*repmat(trainY(:)', ndata, 1);
    
    % non-zero vector
    Ky_nz = Ky(:,logical(nonZero));
    
    if n==1
        Ht_nz = Ht;
    else
        Ht_nz = Ky_nz*w_nz + b*ones(ndata,1);
    end

    Z = Ht_nz + y.*normpdf(Ht_nz)./(normcdf(y.*Ht_nz)+ eps);

    % Adjust the new estimates for the parameters
    M = sqrt(2)*diag(w_nz);

    % new weight vector
    Hess = eye(size(M,1))+M*Ky_nz'*Ky_nz*M;
    U    = chol(Hess);
    Ui   = inv(U);

    w(logical(nonZero)) = M*Ui*Ui'*M*(Ky_nz'*Z - b*Ky_nz'*I);

    S = sqrt(2)*abs(b);
    b = S*(1+ S*ndata*S)^(-1)*S*(I'*Z - I'*Ky*w);

    
     % expectation
    A=diag(1./(2*w_nz.^2));
    beta=(0.5+pmin)/(b^2+pmin);
    
    T=diag((0.5+pmin)./(theta^4 + pmin));
    
    [theta, Q] = minimize(theta,'tlthetaOptimization',10,trainX,w,b,nonZero,Kl,Z,A,beta,T,repy);
    theta
    errlog(n) = -1*Q(1);
          

    nonZero	= (w > w_minimal);

    % determine used vectors
    used = find(nonZero==1);
    
    w(~nonZero)	= 0;
    
    % non-zero weight vector
    w_nz = w(nonZero);
   
    if display % && mod(n,10)==0
        number_of_RVs(n) = length(used);
        plot(1:n, number_of_RVs(1:n));
        title('non-zero vectors')
        drawnow;
    end
    
    if (n >1 && max(abs(w - wold))< threshold)
        

         
        break;
    else
        wold = w;
    end 
    
end

if n<niters
    fprintf('PCVM terminates in %d iteration.',n);
    
else
    fprintf('Exceed the maximal iterations (500). \nConsider to increase niters.')
end





function [Q, dQ] = tlthetaOptimization(theta,trainX,w,b,nonZero,K,Z,A,beta,T,repy)
% in this function, we need to calculate the function value of its gradient
% D is the parameter of d

ndata = size(K,1);
I = ones(ndata,1);

% Note that theta^2
w_nz = w(nonZero);

% % scale columns of kernel matrix with label trainY
Ky = K.*repy;
Ky_nz = Ky(:,nonZero);

Q = 2*w_nz'*Ky_nz'*Z - (w_nz'*Ky_nz')*Ky_nz*w_nz + 2*b*I'*Z - 2*b*I'*Ky_nz*w_nz - b*b*ndata ...
    - w_nz'*A*w_nz- theta.^2'*T*theta.^2 -b*b*beta ;

Q = -Q;

X = [trainX', trainX'];
X = X(1:ndata,1:ndata);
n1sq = sum(X.^2, 1);
n1 = size(X, 2);
D = (ones(n1, 1) * n1sq)' + ones(n1, 1) * n1sq -2 * (X' * X);

dker = -(2*theta*D).*Ky;
dtheta = (Z- b*I - Ky*w)*w'.*dker;
dQ= 2*sum(sum(dtheta))-4*T*theta^3;

dQ = -dQ;





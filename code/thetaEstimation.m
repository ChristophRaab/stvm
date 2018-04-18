function [ rmin ] = thetaEstimation( X)
%THETAESTIMATION Calculates the minimum width of the gaussian kernel based on the
%given data matrix X(m,n) with n dimensions or the precalculated distance matrix D(m,m).
%-------------------------------------------------------------------------
%INPUT: Datamatrix
%OUTPUT: rmin - Minimum width

m = size(X, 2);
n = size(X, 1);


n1sq = sum(X.^2, 1);
n1 = size(X, 2);
D = (ones(n1, 1) * n1sq)' + ones(n1, 1) * n1sq -2 * (X' * X);
D = sqrt(D);

djmax = max(D);

rmin =min(bsxfun(@rdivide, djmax, (sqrt(n)*nthroot(m-1,n))));

end


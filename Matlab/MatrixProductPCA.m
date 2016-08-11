function [err_onepass, err_twopass, err_opt] = MatrixProductPCA(A,B,r,k,numSamples)
% This function implements two pass-efficient algorithms for directly
% computing a rank-r approximation of matrix product A^TB.
% Here k is the sketching size of OnePassPCA and the number of expected
% samples is numSamples*nrlogn.
d = size(A,1);
n1 = size(A,2);
n2 = size(B,2);
n = ceil((n1+n2)/2);
T = randn(k,d)/sqrt(k);
M = A'*B;
%-----best rank-r approximation--------------%
[A1,s,B1] = svds(M,r);
M1 = A1*s*B1';
err_opt = norm(M-M1)/s(1,1);
%-----------Sketching and Sampling-----------------%
A_tilde = T*A;
B_tilde = T*B;
A_tilde_colnsum = (A_tilde.*A_tilde)'*ones(k,n2);
B_tilde_colnsum = ones(n1,k)*(B_tilde.*B_tilde);
M_tilde = (A_tilde)'*(B_tilde);
m = numSamples*n*r*log(n); % number of samples;
A_colnsum = (A.*A)'*ones(d,n2);
B_colnsum = ones(n1,d)*(B.*B);
p = 0.5*m*(A_colnsum/(n2*norm(A,'fro')^2)+B_colnsum/(n1*norm(B,'fro')^2)); % the expected number of samples=m
p = min(p,1);
samples = binornd(1, p);
%------err of OnePassPCA-----------%
samples_cosine = M_tilde./sqrt(A_tilde_colnsum)./sqrt(B_tilde_colnsum);
M_samples = samples_cosine.*sqrt(A_colnsum).*sqrt(B_colnsum);
M_samples(samples==0) = 0;
Iterations = 10;
lambda = 0;
[~,~,v] = svds(randn(n1,n2),r);
u = zeros(n,r);
for i = 1:Iterations
    u = weightALS(samples,M_samples,v,ones(n1,n2),r,lambda);
    v = weightALS(samples',M_samples',u,ones(n2,n1),r,lambda);
end
err_onepass = norm(M-u*v')/s(1,1);
%------err of Two-Pass LELA--------------%
M_samples = M;
M_samples(samples==0) = 0;
Iterations = 10;
lambda = 0;
[~,~,v] = svds(randn(n1,n2),r);
u = zeros(n,r);
for i = 1:Iterations
    u = weightALS(samples,M_samples,v,ones(n1,n2),r,lambda);
    v = weightALS(samples',M_samples',u,ones(n2,n1),r,lambda);
end
err_twopass = norm(M-u*v')/s(1,1);

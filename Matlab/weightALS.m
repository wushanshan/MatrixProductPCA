function u = weightALS(samples, M_samples,v,weights,r,lambda)
% This function implements a single iteration of the weighted alternating
% minimization algorithm: 
% min_u \sum_{samples(i,j)==1} weights(i,j)(M_samples(i,j)-e_i*u*v'*e_j)^2 + lambda*||u||^2_F,
% where "samples", "M_samples", and "weights" are n-by-n matrices;
% "samples" is a 0/1 matrix indicating which entry is sampled;
% "M_samples" contains the sampled values; "u" and "v" are n-by-r matrices. 
n = size(M_samples,1);
u = zeros(n,r);
for i = 1:n
    omega = find(samples(i,:)); % omega is the set {j: samples(i,j)=1}
    A = v(omega,:);
    b = M_samples(i,omega)';
    w = weights(i,omega);
    % x is the optimal value for min_x (Ax-b)'*diag(w)*(Ax-b)+lambda*x'*x
    x = (A'*diag(w)*A+lambda*eye(r))\(A'*diag(w)*b); 
    u(i,:) = x';
end
function [rind,w,A] = ckamlfrank(X,labels,A_i)
%cka-based metric learning for feature ranking
%X in R^{N x P} : input matrix N samples P features
%labels in N^{N} : label vector
[n,nl] = size(labels);
if nl > n
    labels = labels';
end

if nargin < 3
    opts.init = 'pca';
else
     opts.init = A_i;
end


L = bsxfun(@eq,labels,labels'); %label kernel.* amplitude-pulse kernel
[~,ind] = sort(labels);
opts.maxiter = 250;
opts.showWindow = false;
opts.showCommandLine = false;
[~, A_i]=ml_pca(X,0.975);
opts.Q = size(A_i,2);
if size(A_i,2) < 3
   opts.Q = 3;
end   
A = kMetricLearningMahalanobis(X(ind,:),L(ind,ind),labels(ind),opts);
N = size(X,1);
H=eye(N)-1/N*ones(N,1)*ones(1,N);
Y = X*A;
M = size(Y,2);
u = zeros(1,M);
for i = 1 : M
    Di = pdist(Y(:,i));
    si = median(Di);
    K = exp(-squareform(Di).^2/(2*si^2));
    trkl=trace(K*H*L*H);
    trkk=trace(K*H*K*H);
    u(i) = real(log(trkl)-log(trkk)/2);
end
u = u/sum(u);

w = sum(abs(repmat(u,size(A,1),1).*A),2);
w = w - min(w);
w = w/max(w);
[~,rind] = sort(w,'descend');



% -------------------------------------------------------------------------
% Algoritmo para una clasificación con HMM utilizando sólo los MFFC
% Tercera parte: Generación de la curva de rendimiento
% Base de datos: Western Case University
% Implementado por: Ing. José Alberto Hernández Muriel
% -------------------------------------------------------------------------

%% Preinicialización
clc; clear all; close all; format compact;

%% Carga de la base de datos
% Nombre del archivo que contiene la base de datos
Data1  = ['[Western][48k][Xdat2][feat6][Rank][c4].mat'];
% Carga de la base de datos
load(Data1);
% Nombre del archivo que contiene la base de datos
Data2  = ['[Western][48k][feat6].mat'];
% Carga de la base de datos
load(Data2);
% Nombre del archivo que contiene la base de datos
Data3  = ['[Western][Xdat2][feat6][BoW][c4].mat'];
% Carga de la base de datos
load(Data3);

%% Only MCF

Mir = [42:53]';

%% nested cross-validation parameters (balanced)
nr = 10; % # of folds
tc = zeros(nr,1); % elapsed time per fold
nC = length(unique(labels));

indices = A_crossval_bal(labels,nr);

%% Curva de aprendizaje
[P,Nw] = size(Mir);     % # of features and # of selection algorithms to test
rr = 2;
vv = 1 : rr : P;
if vv(end) < P
    vv(end+1) = P;
end

% classification initialization
Nc = numel(unique(labels));         % Number of classes
acc_hmm = zeros(Nw,numel(vv),nr);   % HMM accuracy

% for the HMM
Q = 2;                  % Vector with the states to evaluate
M = 1;                  % Number of mixtures
maxiter = 100;

% E-M algorithm
tol = 1e-4;
cov_type = 'spherical';      % 'full', 'diag' or 'spherical'

% gaussians
method = 'kmeans';      % 'rnd' or 'kmeans'

% parámetros óptimos 
% prior_opt = cell(Nw,numel(vv),nr,1);
% transmat_opt = cell(Nw,numel(vv),nr,1);
% mu_opt = cell(Nw,numel(vv),nr,1);
% Sigma_opt = cell(Nw,numel(vv),nr,1);
% mixmat_opt = cell(Nw,numel(vv),nr,1);
% par_opt = zeros(Nw,numel(vv),nr,2);
par_hmm = cell(Nw,numel(vv),nr,1);
% ac = zeros(Nw,numel(vv),numel(Q),numel(M)); 
% acm = cell(Nw,numel(vv),1,nr);

% Matríz de confusión
Cmhmm = zeros(Nc,Nc,Nw,numel(vv),nr);

% %% Matlab 2013
% matlabpool(4)

%% main loop
for i = 1 : nr
    tic
    fprintf('Fold %d/%d\n',i,nr)
    
    %% Partition 
    if iscell(Xdata)
        Xtrain = Xdata(indices~=i);
        ltrain = labels(indices~=i,1); 
        Xtest = Xdata(indices==i);
        ltest = labels(indices==i,1);
    else
        Xtrain = Xdata(:,:,indices~=i);
        ltrain = labels(indices~=i,1); 
        Xtest = Xdata(:,:,indices==i);
        ltest = labels(indices==i,1);
    end
    
    %% Normalization
    Xtrainp = (reshape(Xtrain,size(Xtrain,1)*size(Xtrain,2),[]))';
    Xtestp = (reshape(Xtest,size(Xtest,1)*size(Xtest,2),[]))';
    
    [Xtrainp,mu,st] = zscore(Xtrainp);
    %st(abs(st)<1e-13) = 1; %evitar problemas de normalización
    Xtestp = (Xtestp - repmat(mu,size(Xtestp,1),1))./repmat(st,size(Xtestp,1),1);
    
    Xtrain = reshape(Xtrainp',size(Xtrain,1),size(Xtrain,2),[]);
    Xtest = reshape(Xtestp',size(Xtest,1),size(Xtest,2),[]);
    
    %% Confusion matrix
    nC = unique(ltest);
    for ii = 1 : length(nC)
        nnC(ii,1) = sum(ltest == nC(ii));
    end
    
    %% learning curve feature selection
    for  z = 1 : Nw %loop feature selection approaches
        fprintf('     Method: %d/%d(%s)\n',z,Nw,featselname{z})
        for jj = length(vv) : length(vv) % loop features

            %% hmm-based classification
            fprintf('          set-feat %d/%d \n',jj,numel(vv))
            
            Xt = Xtrain(Mir(1:vv(jj),z),:,:);
            
            if length(Q)==1
                [prior_opt,transmat_opt,mu_opt,Sigma_opt,mixmat_opt] = ...
                A_hmm_mat_multiclass_k(Xt,ltrain,Q,M,cov_type,maxiter,tol,method);
            else
                [prior_opt,transmat_opt,mu_opt,Sigma_opt,mixmat_opt,par_opt,ac,acm]...
                = findparhmm_nested(Xtrain(Mir(1:vv(jj),z),:,:),ltrain,maxiter,tol,cov_type,method,Q,M);
            end
            
            % Testing over test set
            lteste = A_hmmclassify_multi_k(Xtest(Mir(1:vv(jj),z),:,:), prior_opt, transmat_opt,mu_opt,...
            Sigma_opt, mixmat_opt);
            % accuracy measure
            
            acc = 100*sum(lteste==ltest)/numel(ltest);
            acc_hmm(z,jj,i) = acc;
            if length(Q)==1
                par_hmm{z,jj,i,:} = {prior_opt,transmat_opt,mu_opt,Sigma_opt,mixmat_opt};

            else    
                par_hmm{z,jj,i,:} = {prior_opt,transmat_opt,mu_opt,Sigma_opt,mixmat_opt,par_opt,ac,acm};
            end                  
            try
                Cmhmm(:,:,z,jj,i) = confusionmat(ltest,lteste);
                Cmhmm(:,:,z,jj,i) = 100*squeeze(Cmhmm(:,:,z,jj,i))./repmat(nnC,1,size(squeeze(Cmhmm(:,:,z,jj,i)),2));
            end
            
            fprintf('          -accHMM=%.2f  (set-feat %d/%d)\n',acc,jj,numel(vv));
        end
    end
    tc(i) =  toc;
    fprintf('e.time=%.4f[s]\n',tc(i));
end

%% Accuracy
Acc = mean(acc_hmm(1,7,:));
Std = std(acc_hmm(1,7,:));
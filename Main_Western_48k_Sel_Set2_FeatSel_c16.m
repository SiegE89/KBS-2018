% -------------------------------------------------------------------------
% Algoritmo para realizar la selección de características con HMM
% Tercera parte: Generación de la curva de rendimiento
% Base de datos: Western Case University
% Implementado por: Ing. José Alberto Hernández Muriel
% -------------------------------------------------------------------------

%% Preinicialización
clc; clear all; close all; format compact;

%% Carga de la base de datos
% Nombre del archivo que contiene la base de datos
Data1  = ['[Western][48k][Xdat2][feat6][Rank][c16].mat'];
% Carga de la base de datos
load(Data1);
% Nombre del archivo que contiene la base de datos
Data2  = ['[Western][48k][feat6].mat'];
% Carga de la base de datos
load(Data2);
% Nombre del archivo que contiene la base de datos
Data3  = ['[Western][Xdat2][feat6][BoW][c16].mat'];
% Carga de la base de datos
load(Data3);

%% Selección de los datos a trabajar
Xdata = Xdat2(:,:,inds);

%% Agregar la carpeta que contiene las funciones necesarias para proceso 
% addpath(genpath('/home/SiegE89/MatToolsAM'))
% addpath(genpath('/home/SiegE89/HMMall_D'))

% addpath(genpath('C:\Users\Usuario UTP\Google Drive\Proyecto_Vibraciones\Codigos\HMMall_D'))
% addpath(genpath('C:\Users\Usuario UTP\Google Drive\Proyecto_Vibraciones\Codigos\matToolsAm'))

addpath(genpath('HMMall_D'))
addpath(genpath('MatToolsAm'))

%%
fprintf('\nExtracción de los indices de relevancia: ')

%% Generación del ranking de la matriz con PCA ............................
M = 53;     % Número de filas
N = 7;      % Número de columnas
% Inicialización de la matriz de pesos
W_PCA = zeros(M,N);

Init  = 1;
Final = M;

for i = 1:N
    W_PCA(:,i) = w_pca(Init:Final);
    Init  = Init  + M;
    Final = Final + M;
end
   
w_pca2  = sum(W_PCA,2);
[Aux i_pca] = sort(w_pca2,'descend');

%% Generación del ranking de la matriz con Self-weigth ....................
% Inicialización de la matriz de pesos
W_SW  = zeros(M,N);

Init  = 1;
Final = M;

for i = 1:N
    W_SW(:,i) = w_sw(Init:Final);
    Init  = Init  + M;
    Final = Final + M;
end
   
w_sw2  = sum(W_SW,2);
[Aux i_sw] = sort(w_sw2,'descend');

%% Generación del ranking de la matriz con Laplacian-score ................
% Inicialización de la matriz de pesos
W_LS  = zeros(M,N);

Init  = 1;
Final = M;

for i = 1:N
    W_LS(:,i) = w_ls(Init:Final);
    Init  = Init  + M;
    Final = Final + M;
end
   
w_ls2  = sum(W_LS,2);
[Aux i_ls] = sort(w_ls2,'descend');

%% Generación del ranking de la matriz con Distance-weigth ................
% Inicialización de la matriz de pesos
W_DR  = zeros(M,N);

Init  = 1;
Final = M;

for i = 1:N
    W_DR(:,i) = w_dr(Init:Final);
    Init  = Init  + M;
    Final = Final + M;
end
   
w_dr2  = sum(W_DR,2);
[Aux i_dr] = sort(w_dr2,'descend');

%% Generación del ranking de la matriz con Relief-F .......................
% Inicialización de la matriz de pesos
W_RF  = zeros(M,N);

Init  = 1;
Final = M;

for i = 1:N
    W_RF(:,i) = w_rf(Init:Final);
    Init  = Init  + M;
    Final = Final + M;
end
   
w_rf2  = sum(W_RF,2);
[Aux i_rf] = sort(w_rf2,'descend');

%% Generación del ranking de la matriz con CKAML ..........................
% Inicialización de la matriz de pesos
W_ML  = zeros(M,N);

Init  = 1;
Final = M;

for i = 1:N
    W_ML(:,i) = w_ml(Init:Final);
    Init  = Init  + M;
    Final = Final + M;
end
   
w_ml2  = sum(W_ML,2);
[Aux i_ml] = sort(w_ml2,'descend');

%% Generación de la matriz MIR
Mir = [i_pca,i_sw,i_ls,i_dr,i_rf,i_ml];
Wr  = [w_pca2,w_sw2,w_ls2,w_dr2,w_rf2,w_ml2];

fprintf('Completo\n')

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
        parfor jj = 1 : length(vv) % loop features
            %% hmm-based classification
            fprintf('          set-feat %d/%d \n',jj,numel(vv))
            
            if length(Q)==1
%                 disp('1a')
                [prior_opt,transmat_opt,mu_opt,Sigma_opt,mixmat_opt] = ...
                A_hmm_mat_multiclass_k(Xtrain(Mir(1:vv(jj),z),:,:),ltrain,Q,M,cov_type,maxiter,tol,method);
            else
%                 disp('1b')
                [prior_opt,transmat_opt,mu_opt,Sigma_opt,mixmat_opt,par_opt,ac,acm]...
                = findparhmm_nested(Xtrain(Mir(1:vv(jj),z),:,:),ltrain,maxiter,tol,cov_type,method,Q,M);
            end
%             disp('done1')
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
    save('[Western][FeatSel][Set2][c16]','Mir','Wr','acc_hmm','par_hmm','indices','Nw','Cmhmm','vv');
end
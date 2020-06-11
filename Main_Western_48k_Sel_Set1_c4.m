% -------------------------------------------------------------------------
% Algoritmo para realizar la selección de características con KNN
% Base de datos: Western Case University
% Implementado por: Ing. José Alberto Hernández Muriel
% -------------------------------------------------------------------------
%% Preinicialización
clc; clear all; close all; format compact;

%% Agregar la carpeta que contiene las funciones necesarias para proceso 
addpath(genpath('/home/SiegE89/MatToolsAM'));

%% Cargar la base de datos a utilizar
load('[Western][Xdat1][BoW][c4].mat')

%% Normalización de los datos
nor = 'z';
fprintf('\n Data normalization...')
if strcmp(nor,'z')
    %zscore
    [Xn,uz,sz] = zscore(X);
    [Xn,maxX,minX] = drnormalization(Xn);
else
    %dynamic range normalization
    [Xn,maxX,minX] = drnormalization(X);
end

fprintf('done\n')

%% Métodos de selección de características
featselname = {'PCA';'Self-weigth';'LaplacianScore';'Dist.-weigth';'Relieff';'CKAML'};
fprintf('Feature selection:\n')
%usupervised
%pca-based
tic
fprintf('PCA...')
[i_pca,w_pca] = pcafrank(Xn);
fprintf('done\n')
toc
%self-weigth algorithm
tic
fprintf('self-weigth...')
[i_sw,w_sw] = sweigthfrank(Xn);
fprintf('done\n')
toc
%laplacian score algorithm
tic
fprintf('laplacian-weigth...')
[i_ls,w_ls,Kg] = laplacianscorefrank(Xn);
fprintf('done\n')
toc
%supervised
%distance-based algorithm
tic
fprintf('distance-based weigth...')
[i_dr,w_dr] = distsupfrank(Xn,labels);
fprintf('done\n')
toc
%Relieff-based using 1-KNN prediction
tic
fprintf('Relieff...')
[i_rf,w_rf] = reliefnor(Xn,labels);
fprintf('done\n')
toc
%cka metric learning
tic
fprintf('CKA-ML...')
[i_ml,w_ml,A] = ckamlfrank(Xn,labels);
fprintf('done\n')
toc
%save weights and relevant indices
Mir = [i_pca,i_sw,i_ls,i_dr,i_rf,i_ml];
Wr = [w_pca,w_sw,w_ls,w_dr,w_rf,w_ml];
Nw = size(Mir,2);   %number of tested approaches
save('[Western][FeatSel][Set1][c4][CheckPoint]')

%% Curva de aprendizaje
[P,Nw] = size(Mir); %# of features and # of selection algorithms to test
rr = 2;
%P = round(P/rr);
vv = 1 : rr : P;
if vv(end) < P
    vv(end+1) = P;
end
nr = 10; % # of folds
tc = zeros(nr,1); % elapsed time per fold
% classification initialization
acc_knn = zeros(Nw,numel(vv),nr); % knn accuracy
par_opt = zeros(Nw,numel(vv),nr,2); %knn parameters (#knn and sigma gaussian kernel)
Nc = numel(unique(labels));
Cmknn = zeros(Nc,Nc,Nw,numel(vv),nr); %knn parameters (#knn and sigma gaussian kernel)

%% nested cross-validation partition
%balanced cross-validation partition
kv = [1,3,5,7,9];
indices = A_crossval_bal(labels,nr);

%% main loop
for i = 1 : nr
    tic
    fprintf('Fold %d/%d\n',i,nr)
    %% Partition
    %[indices_train,indices_test] = crossvalind('Resubstitution',size(Xdata,1),[1-ptrain,ptrain]);
    indices_train = indices ~= i;
    indices_test = indices == i;
    Xtrain = X(indices_train,:);
    Xtest = X(indices_test,:);
    ltrain = labels(indices_train);
    ltest = labels(indices_test);
    %% Normalization
    if strcmp(nor,'z')
        [Xtrain,mu,st] = zscore(Xtrain);
        st(abs(st)<1e-13) = 1; %evitar problemas de normalización
        Xtest = (Xtest - repmat(mu,size(Xtest,1),1))./repmat(st,size(Xtest,1),1);
        [Xtrain,maxX,minX] = drnormalization(Xtrain);
        Xtest = (Xtest - minX)/(maxX-minX);
    else
        %% dynamic range normalization
        [Xtrain,maxX,minX] = drnormalization(Xtrain);
        Xtest = (Xtest - minX)/(maxX-minX);
    end
    %confusion matrix
    nC = unique(ltest);
    for ii = 1 : length(nC)
        nnC(ii,1) = sum(ltest == nC(ii));
    end
    
    %% learning curve feature selection
    for  z = 1 : Nw %loop feature selection approaches
        parfor j = 1 : length(vv) % loop features
            %% knn-based classification
            %training knn
            fprintf('Fold %d/%d-M%d/%d(%s)-setfeat%d/%d',i,nr,z,Nw,featselname{z},j,numel(vv))
            [k_opt,sig_opt] = findparknn_nested(Xtrain(:,Mir(1:vv(j),z)),ltrain,kv);
            %testing over test set
            Ktest = exp(-pdist2(Xtrain(:,Mir(1:vv(j),z)),Xtest(:,Mir(1:vv(j),z))).^2/(2*sig_opt^2));
            lteste = A_kernel_knn(Ktest,ltrain,k_opt,'mode');
            aknn = 100*sum(lteste==ltest)/numel(ltest);
            acc_knn(z,j,i) = aknn;
            par_opt(z,j,i,:) = [k_opt,sig_opt];
            try
                Cmknn(:,:,z,j,i) = confusionmat(ltest,lteste);
                Cmknn(:,:,z,j,i) = 100*squeeze(Cmknn(:,:,z,j,i))./repmat(nnC,1,size(squeeze(Cmknn(:,:,z,j,i)),2));
            end
            fprintf('-accKNN=%.2f',aknn);
        end
    end
    tc(i) =  toc;
    fprintf('e.time=%.4f[s]\n',tc(i));
    save('[Western][FeatSel][Set1][c4]','Mir','Wr','acc_knn','par_opt','indices','Nw','Cmknn','vv');
end


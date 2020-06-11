% -------------------------------------------------------------------------
% Algoritmo para realizar la selección de características con HMM
% Segunda parte: Generación de los vectores de ranking
% Base de datos: Western Case University
% Implementado por: Ing. José Alberto Hernández Muriel
% -------------------------------------------------------------------------

%% Preinicialización
clc; clear all; close all; format compact;

%% Carga de la super matriz
Data = ['[Western][48k][Xdat2][feat6][Super][c4].mat'];

%% Agregar la carpeta que contiene las funciones necesarias para proceso 
addpath(genpath('/home/SiegE89/MatToolsAM'));

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

% unsupervised ............................................................
% PCA - based
tic
fprintf('PCA...')
[i_pca,w_pca] = pcafrank(Xn);
fprintf('done\n')
toc

% Self-weigth
tic
fprintf('self-weigth...')
[i_sw,w_sw] = sweigthfrank(Xn);
fprintf('done\n')
toc

% Laplacian-Score
tic
fprintf('laplacian-weigth...')
[i_ls,w_ls,Kg] = laplacianscorefrank(Xn);
fprintf('done\n')
toc

% Supervised ..............................................................

% Distance-based algorithm
tic
fprintf('distance-based weigth...')
[i_dr,w_dr] = distsupfrank(Xn,labels);
fprintf('done\n')
toc

% Relieff-based using 1-KNN prediction
tic
fprintf('Relieff...')
[i_rf,w_rf] = reliefnor(Xn,labels);
fprintf('done\n')
toc

% cka metric learning
tic
fprintf('CKA-ML...')
[i_ml,w_ml,A] = ckamlfrank(Xn,labels);
fprintf('done\n')
toc

% Save weights and relevant indices
Mir = [i_pca,i_sw,i_ls,i_dr,i_rf,i_ml];
Wr = [w_pca,w_sw,w_ls,w_dr,w_rf,w_ml];
Nw = size(Mir,2);   %number of tested approaches
save('[Western][48k][Xdat2][feat6][Rank][c4]')


% -------------------------------------------------------------------------
% Algoritmo para realizar la selección de características con HMM
% Segunda parte: Generación de los vectores de ranking (4 clases)
% Base de datos: Western Case University
% Implementado por: Ing. José Alberto Hernández Muriel
% -------------------------------------------------------------------------

%% Preinicialización
clc; clear all; close all; format compact;

%% Carga de la super matriz
% Ubicación del drive
Path   = ['C:\Users\Usuario UTP\'];
% Path = ['C:\Users\SiegE89\'];

% Carpeta de ubicación del proyecto
Proy   = ['Google Drive\Proyecto_Vibraciones\'];
% Carpeta donde están las bases de datos a utilizar
D_base = ['Database\Bearings\Western\48 kHz\'];

% Nombre del archivo que contiene la base de datos
Data = ['[Western][48k][Xdat2][feat6][Super][c4].mat'];
% Carga de la base de datos
load([Path Proy D_base Data]);

%% Agregar la carpeta que contiene las funciones necesarias para proceso 
% Ubicación de la carpeta de códigos
Code     = ['Codigos\'];

% Carpetas a utilizar
AMtools  = ['matToolsAm'];

% Agregar las carpetas que contienen las funciones necesarias para el
% procesamiento
addpath(genpath([Path Proy Code AMtools]))

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

% % cka metric learning
% tic
% fprintf('CKA-ML...')
% [i_ml,w_ml,A] = ckamlfrank(Xn,labels);
% fprintf('done\n')
% toc

% Save weights and relevant indices
% Mir = [i_pca,i_sw,i_ls,i_dr,i_rf,i_ml];
Mir = [i_pca,i_sw,i_ls,i_dr,i_rf];
% Wr = [w_pca,w_sw,w_ls,w_dr,w_rf,w_ml];
Wr = [w_pca,w_sw,w_ls,w_dr,w_rf];
Nw = size(Mir,2);   %number of tested approaches

save('[Western][48k][Xdat2][feat6][Rank][c4]-2')


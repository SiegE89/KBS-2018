% -------------------------------------------------------------------------
% Algoritmo para realizar la selecci�n de segmentos a utilizar mediante
% bolsa de palabras
% Base de datos: Western Case University
% Implementado por: Ing. Jos� Alberto Hern�ndez Muriel
% -------------------------------------------------------------------------
%% Preinicializaci�n
clc; clear all; close all; format compact;

%% Agregar la funci�n que contiene las funciones necesarias para utilizar 
 % bolsa de palabras
addpath(genpath('C:\Users\jorge\Google Drive\Proyecto_Vibraciones\Codigos\MatToolsAM'));

%% Cargar la base de datos a utilizar
load('C:\Users\jorge\Google Drive\Paper\Paper vibraciones\Fractals\GetFeatures\MFD_MFCC_Western_FULL_3d')

% %% Primera parte: Elecci�n de las muestras m�s signifitivas (Set 1 - 4 Clases)
%     X = Xdat1;
%     labels = Labels4;
%       
%     % Bolsa de palabras
%     nsc  = 1000;
%     inds = A_bagwordkmeansI(X,labels,nsc);
%     X = X(inds,:);
%     labels = labels(inds,1);
%     
%     % Almacenamiento de la matriz obtenida
%     save('[Western][Xdat1][BoW][c4]','X','labels','inds')
% 
% %% Segunda parte: Elecci�n de las muestras m�s signifitivas (Set 1 - 10 Clases)
%     X = Xdat1;
%     labels = Labels10;
%     
%     % Bolsa de palabras
%     nsc  = 500;
%     inds = A_bagwordkmeansI(X,labels,nsc);
%     X = X(inds,:);
%     labels = labels(inds,1);
%     
%     % Almacenamiento de la matriz obtenida
%     save('[Western][Xdat1][BoW][c10]','X','labels','inds')
% 
% %% Tercera parte: Elecci�n de las muestras m�s signifitivas (Set 1 - 16 Clases)
%     X = Xdat1;
%     labels = Labels16;
%     
%     % Bolsa de palabras
%     nsc  = 400;
%     inds = A_bagwordkmeansI(X,labels,nsc);
%     X = X(inds,:);
%     labels = labels(inds,1);
%     
%     % Almacenamiento de la matriz obtenida
%     save('[Western][Xdat1][BoW][c16]','X','labels','inds')

%% Cuarta parte: Elecci�n de las muestras m�s signifitivas (Set 2 - 4 Clases)
       
    XMFCC = reshape(MFCC,size(MFCC,3),);
    

    
    labels = Labels4;
    
    % Bolsa de palabras
    nsc  = 1000;
    inds = A_bagwordkmeansI(X,labels,nsc);
    X = X(inds,:);
    labels = labels(inds,1);
    
    % Almacenamiento de la matriz obtenida
    save('[Western][Xdat2][BoW][c4]','X','labels','inds')

%% Quinta parte: Elecci�n de las muestras m�s signifitivas (Set 2 - 10 Clases)
    Xo = Xdat2;
    [M,N,O] = size(Xo);    
        
    X = zeros(O,M*N);
    
    for i=1:O
        X(i,:) = reshape(Xo(:,:,i),1,M*N);
    end
    
    labels = Labels10;
    
    % Bolsa de palabras
    nsc  = 500;
    inds = A_bagwordkmeansI(X,labels,nsc);
    X = X(inds,:);
    labels = labels(inds,1);
    
    % Almacenamiento de la matriz obtenida
    save('[Western][Xdat2][BoW][c10]','X','labels','inds')

%% Sexta parte: Elecci�n de las muestras m�s signifitivas (Set 2 - 16 Clases)
    Xo = Xdat2;
    [M,N,O] = size(Xo);    
        
    X = zeros(O,M*N);
    
    for i=1:O
        X(i,:) = reshape(Xo(:,:,i),1,M*N);
    end
    
    labels = Labels16;
    
    % Bolsa de palabras
    nsc  = 400;
    inds = A_bagwordkmeansI(X,labels,nsc);
    X = X(inds,:);
    labels = labels(inds,1);
    
    % Almacenamiento de la matriz obtenida
%     save('[Western][Xdat2][BoW][c16]','X','labels','inds')
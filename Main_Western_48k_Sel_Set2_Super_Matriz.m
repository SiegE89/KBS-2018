% -------------------------------------------------------------------------
% Algoritmo para realizar la selecci�n de caracter�sticas con HMM
% Primera parte: Generaci�n de la Super-Matriz
% Base de datos: Western Case University
% Implementado por: Ing. Jos� Alberto Hern�ndez Muriel
% -------------------------------------------------------------------------

%% Preinicializaci�n
clc; clear all; close all; format compact;

%% Carga de la base de datos
% Carga del path de localizaci�n de las carpetas
% Path  = ['C:\Users\Usuario UTP\'];   % Desde UTP
% Path  = ['C:\Users\Siege89\'];       % Desde SiegE89

% Nombre del archivo que contiene la base de datos
Data  = ['[Western][48k][feat6].mat'];

% Localizaci�n del archivo de la base de datos
% Local = ['Google Drive\Proyecto_Vibraciones\Database\Bearings\Western\48 kHz\']

% Carga de la base de datos
load(Data);

%% Carga de los �ndices obtenidos mediante bag of words

BoW = ['[Western][Xdat2][feat6][BoW][c4].mat'];

% Carga de la base de datos
load(BoW);

%% Selecci�n de las muestras a utilizar (Teniendo en cuenta BoW)
Xi = Xdat2(:,:,inds);

%% Generaci�n de la Super-Matriz
O     = length(Xi);
[M N] = size(Xi(:,:,1));

X     = zeros(O,M*N);

fprintf('\nSuper Matrix generation: \n')

for i=1:O        
    Aux = Xi(:,:,i); 
    Init  = 1;
    Final = M;
    for j = 1:N     
        Aux2 = Aux(:,j);
        X(i,Init:Final) = Aux2;
        Init  = Init  + M;
        Final = Final + M;
    end
    fprintf('     Fold %d/%d\n',i,O)
end

save('[Western][48k][Xdat2][feat6][Super][c4]','X','labels')

%% Preinicializaci�n
clc; clear all; close all; format compact;

%% Carga de la base de datos
% Carga del path de localizaci�n de las carpetas
% Path  = ['C:\Users\Usuario UTP\'];   % Desde UTP
% Path  = ['C:\Users\Siege89\'];       % Desde SiegE89

% Nombre del archivo que contiene la base de datos
Data  = ['[Western][48k][feat6].mat'];

% Localizaci�n del archivo de la base de datos
% Local = ['Google Drive\Proyecto_Vibraciones\Database\Bearings\Western\48 kHz\']

% Carga de la base de datos
load(Data);

%% Carga de los �ndices obtenidos mediante bag of words

BoW = ['[Western][Xdat2][feat6][BoW][c10].mat'];

% Carga de la base de datos
load(BoW);

%% Selecci�n de las muestras a utilizar (Teniendo en cuenta BoW)
Xi = Xdat2(:,:,inds);

%% Generaci�n de la Super-Matriz
O     = length(Xi);
[M N] = size(Xi(:,:,1));

X     = zeros(O,M*N);

fprintf('\nSuper Matrix generation: \n')

for i=1:O        
    Aux = Xi(:,:,i); 
    Init  = 1;
    Final = M;
    for j = 1:N     
        Aux2 = Aux(:,j);
        X(i,Init:Final) = Aux2;
        Init  = Init  + M;
        Final = Final + M;
    end
    fprintf('     Fold %d/%d\n',i,O)
end  

save('[Western][48k][Xdat2][feat6][Super][c10]','X','labels')

%% Preinicializaci�n
clc; clear all; close all; format compact;

%% Carga de la base de datos
% Carga del path de localizaci�n de las carpetas
% Path  = ['C:\Users\Usuario UTP\'];   % Desde UTP
% Path  = ['C:\Users\Siege89\'];       % Desde SiegE89

% Nombre del archivo que contiene la base de datos
Data  = ['[Western][48k][feat6].mat'];

% Localizaci�n del archivo de la base de datos
% Local = ['Google Drive\Proyecto_Vibraciones\Database\Bearings\Western\48 kHz\']

% Carga de la base de datos
load(Data);

%% Carga de los �ndices obtenidos mediante bag of words

BoW = ['[Western][Xdat2][feat6][BoW][c16].mat'];

% Carga de la base de datos
load(BoW);

%% Selecci�n de las muestras a utilizar (Teniendo en cuenta BoW)
Xi = Xdat2(:,:,inds);

%% Generaci�n de la Super-Matriz
O     = length(Xi);
[M N] = size(Xi(:,:,1));

X     = zeros(O,M*N);

fprintf('\nSuper Matrix generation: \n')

for i=1:O        
    Aux = Xi(:,:,i); 
    Init  = 1;
    Final = M;
    for j = 1:N     
        Aux2 = Aux(:,j);
        X(i,Init:Final) = Aux2;
        Init  = Init  + M;
        Final = Final + M;
    end
    fprintf('     Fold %d/%d\n',i,O)
end   

save('[Western][48k][Xdat2][feat6][Super][c16]','X','labels')
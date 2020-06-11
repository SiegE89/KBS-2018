% -------------------------------------------------------------------------
% Algoritmo para realizar la caracterizaci�n de la base de datos (M�todo 6)
% Base de datos: Western Case University
% Implementado por: Ing. Jos� Alberto Hern�ndez Muriel
% -------------------------------------------------------------------------

%% Preinicializaci�n
clc; clear all; close all; format compact

%% Carga de la base de datos
% Carga del path de localizaci�n de las carpetas
Path  = ['C:\Users\Usuario UTP\'];   % Desde UTP
% Path  = ['C:\Users\Siege89\'];       % Desde SiegE89

% Nombre del archivo que contiene la base de datos
Data  = ['[Western][48k][Preprocessing][4096].mat'];

% Localizaci�n del archivo de la base de datos
Local = ['Google Drive\Proyecto_Vibraciones\Database\Bearings\Western\48 kHz\'];

% Carga de la base de datos
load([Path Local Data]);

%% Carga de las carpetas que contienen las funciones necesarias para el 
 % procesamiento

% Carpeta con las funciones de preprocesamiento
Prep_Folder = ['Google Drive\Proyecto_Vibraciones\Codigos\Vibr_Toolbox\Vibr_Preprocessing'];
% Carpeta con las funciones de Caracterizaci�n
Feat_Folder = ['Google Drive\Proyecto_Vibraciones\Codigos\Vibr_Toolbox\Vibr_Caracterization'];
% Carpeta con las funciones de Caracterizaci�n (Mel-Cepstrum)
MFCC_Folder = ['Google Drive\Proyecto_Vibraciones\Codigos\Vibr_Toolbox\Vibr_Caracterization\voicebox'];

% Agregar las carpetas al espacio de trabajo de Matlab
addpath(genpath([Path Prep_Folder]))
addpath(genpath([Path Feat_Folder]))
addpath(genpath([Path MFCC_Folder]))

%% Cargar la base de datos desde el servidor
% load('/home/SiegE89/[Western][48k][Preprocessing][4096].mat');
% 
% %% Agregar las carpetas contenedoras de las funciones (Desde el servidor)
% addpath(genpath('/home/SiegE89/Vibr_Toolbox/Vibr_Caracterization/voicebox'))
% addpath(genpath('/home/SiegE89/Vibr_Toolbox/Vibr_Caracterization'))
% addpath(genpath('/home/SiegE89/Vibr_Toolbox/Vibr_Preprocessing'))
%% Inicializaci�n de variables
Fs    = 48000;              % Frecuencia de muestreo
[N M] = size(Vibr_DE)       % Dimensiones de la matriz de datos
Ns1   = 2^12;               % NFFT para la primer FFT

Xdat1 = zeros(N,125);       % Inicializaci�n de Xdat1 (Para KNN)
Xdat2 = zeros(53,7,N);      % Inicializaci�n de Xdat2 (Para HMM)

%% Caracterizaci�n de toda la base de datos
for i=1:N
    signal  = Vibr_DE(i,:);  % Carga del segmento
    % ---------------------------------------------------------------------
    % C�lculo del primer set de caracter�sticas (Para KNN)
    % ---------------------------------------------------------------------
        % Caracterizaci�n del segmento en el tiempo
        T1      = stat1(signal);            % Par�metros de la Tabla 1
        % Caracterizaci�n del segmento en la frecuencia
        [Spect Freq] = Vibr_FFT(signal,Fs,Ns1,0); 
        F11     = stat2(Spect,Freq);        % Par�metros de la Tabla 2
        F21     = stat1(Spect);             % Par�metros de la Tabla 1
        % Caracterizaci�n del segmento mediante Mel-Cepstrum
        [MC Ds] = melcepst(signal,Fs);
        [V Pmc] = size(MC);
        MC1     = reshape(MC,1,V*Pmc);
        % Almacenamiento de los datos calculados
        Xdat1(i,:) = [T1 F11 F21 MC1];
    % ---------------------------------------------------------------------
    % C�lculo del primer set de caracter�sticas (Para HMM)
    % ---------------------------------------------------------------------
        T2  = zeros(18,V);
        F12 = zeros(5,V);
        F22 = zeros(18,V);
        parfor j=1:V
            % Elecci�n del subsegmento
            Sgn      = Ds(V,:);
            % Caracterizaci�n del subsegmento en el dominio del tiempo
            T2(:,j)  = stat1(Sgn);          % Par�metros de la Tabla 1
            % Caracterizaci�n del subsegmento en el dominio de la
            % frecuencia
            [Spect Freq] = Vibr_FFT(Sgn,Fs,Ns1,0); 
            F12(:,j) = stat2(Spect,Freq);   % Par�metros de la Tabla 2
            F22(:,j) = stat1(Spect);        % Par�metros de la Tabla 1
        end
        MC2 = MC';
        Xdat2(:,:,i) = [T2; F12; F22; MC2];
            
     fprintf('Fold: %d/%d \n',i,N);   
end

% OutFile = [Folder '[Western][48k][feat6]'];
% save(OutFile,'Xdat1','Xdat2','Labels4','Labels10','Labels16')
save('[Western][48k][feat6]','Xdat1','Xdat2','Labels4','Labels10','Labels16')
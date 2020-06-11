% -------------------------------------------------------------------------
% METODOLOGÍA DE CARACTERIZACIÓN # 6
% Etapa de procesamiento: Caracterización
% Entradas: Señal a caracterizar, Frecuencia de muestreo y el número de
% ventanas a emplear durante el análisis en frecuencia (FFT+Windowing)
% Implementado por: Ing. José Alberto Hernández Muriel
% -------------------------------------------------------------------------

% Preinicialización
clc; clear all; close all; format compact;

% Cargar carpetas desdel el PC de la UTP
addpath(genpath('C:\Users\Usuario UTP\Google Drive\Proyecto_Vibraciones\Codigos\Vibr_Toolbox\Vibr_Caracterization'))
addpath(genpath('C:\Users\Usuario UTP\Google Drive\Proyecto_Vibraciones\Codigos\Vibr_Toolbox\Vibr_Preprocess'))
Folder = ['C:\Users\Usuario UTP\Google Drive\Proyecto_Vibraciones\Database\Bearings\Western\48 kHz\'];

% % Ubicación de la base de dato (Desde el PC de mi casa)
% addpath(genpath('C:\Users\SiegE89\Google Drive\Proyecto_Vibraciones\Codigos\Vibr_Toolbox\Vibr_Caracterization'))
% addpath(genpath('C:\Users\SiegE89\Google Drive\Proyecto_Vibraciones\Codigos\Vibr_Toolbox\Vibr_Preprocess'))
% Folder = ['C:\Users\SiegE89\Google Drive\Proyecto_Vibraciones\Database\Bearings\Western\48 kHz\'];

Data = ['[Western][48k][Preprocessing][4096].mat'];
load([Folder Data])

% Inicialización de variables
Fs    = 48000;              % Frecuencia de muestreo
[N M] = size(Vibr_DE);      % Dimensiones de la matriz de datos
Ns1   = 2^12;               % NFFT para la primer FFT
Ns2   = 2^13;               % NFFT para la FFT enventanada
Nb    = 40;                 % Número de bandas
Nstd  = 0.2;                % Desviación estandar del ruido blanco
NE    = 100;                % Number of emsembles
NLev  = 3;                  % Número de niveles de descomposición
Wname = 'db10';             % Wavelet madre

Xdata = zeros(N,12631);     % Inicialización de Xdata

parfor i=1:N
    signal  = Vibr_DE(i,:);  % Carga del segmento
    [RAW,W] = Vibr_Features(signal,Fs,Ns1,Ns2,Nb); 
    IMFs_DE = Pro_EEMD(signal,Nstd,NE);
    IMF     = zeros(1,5944);
    for k = 2:9
        Init  = 1+(743*(k-2));
        Final = 743+(743*(k-2));
        IMF(Init:Final) = Vibr_Features(signal,Fs,Ns1,Ns2,Nb); 
    end
    Wdec    = wpdec(signal,NLev,Wname);
    WPT     = zeros(1,5944);
    for k =1:8
        Coef  = wpcoef(Wdec,k);
        Init  = 1+(743*(k-1));
        Final = 743+(743*(k-1));
        WPT(Init:Final) = Vibr_Features(Coef,Fs,Ns1,Ns2,Nb); 
    end
    
    Xdata(i,:) = [RAW IMF WPT];
    fprintf('Fold: %d/%d \n',i,N);
end

OutFile = [Folder '[Western][48k][feat5]'];
save(OutFile,'Xdata','Labels4','Labels10','Labels16')
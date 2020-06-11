% -------------------------------------------------------------------------
% METODOLOGÍA DE SEGMENTACIÓN
% Etapa de procesamiento: Segmentación
% Entradas: Dirección de los archivos de la base de datos a procesar
% Implementado por: Ing. José Alberto Hernández Muriel
% -------------------------------------------------------------------------

% Preinicialización
clc; clear all; close all;

% % Ubicación de la base de datos (Desde el PC de mi casa)
% Folder = ['C:\Users\SiegE89\Google Drive\Proyecto_Vibraciones\Database\Bearings\Western\48 kHz\'];

% Ubicación d ela base de datos (Desde el PC de la UTP)
Folder = ['C:\Users\Usuario UTP\Google Drive\Proyecto_Vibraciones\Database\Bearings\Western\48 kHz\'];

% Extracción de los archivos al interior de la carpeta que contiene la base
% de datos a analizar
Files = dir(fullfile(Folder)); 
Aux   = {Files.name};
N     = length(Aux);

% Inicialización de la posición de las celdas donde se almacenan los datos
Pseg  = 0;
% Tamaño del segmento
Lseg = 4096;

for i = 3:58
    fprintf('Segmentando archivo: %s \n',Aux{i})
    File     = [Folder Aux{i}];
    Data     = load(File);
    Name_Var = fieldnames(Data);

    Signal1  = getfield(Data,Name_Var{1});
    Signal2  = getfield(Data,Name_Var{2});
    
    Nseg = floor(length(Signal1)/Lseg);
    
    Aux2 = Aux{i};
    Name = Aux2(1:4);
    
    % ---------------------------------------------------------------------
    % Generación de la etiqueta del conjunto de segmentos
    % ---------------------------------------------------------------------
    State = Name(1);        % 4 Estados
    switch State
        case '0'            % Normal
            L4 = 1;          
        case '1'            % Inner
            L4 = 2; 
        case '2'            % Ball
            L4 = 3; 
        case '3'            % Outer
            L4 = 4; 
    end
    
    State = [Name(1) Name(3)];
    switch State            % 10 Estados
        case '00'
            L10 = 1;        % Normal
        case '11'           
            L10 = 2;        % Inner + Baja
        case '12'           
            L10 = 3;        % Inner + Media
        case '13'
            L10 = 4;        % Inner + Alta
        case '21'           
            L10 = 5;        % Ball + Baja
        case '22'           
            L10 = 6;        % Ball + Media
        case '23'
            L10 = 7;        % Ball + Alta
        case '31'           
            L10 = 8;        % Outer + Baja
        case '32'           
            L10 = 9;        % Outer + Media
        case '33'
            L10 = 10;       % Outer + Alta
    end    
    
    State = [Name(1) Name(2)];
    switch State            % 16 Estados
        case '00'
            L16 = 1;        % Normal + Sin carga
        case '01'
            L16 = 2;        % Normal + Carga baja
        case '02'
            L16 = 3;        % Normal + Carga media
        case '03'
            L16 = 4;        % Normal + Carga alta
        case '10'
            L16 = 5;        % Inner + Sin carga
        case '11'
            L16 = 6;        % Inner + Carga baja
        case '12'
            L16 = 7;        % Inner + Carga media
        case '13'
            L16 = 8;        % Inner + Carga alta
        case '20'
            L16 = 9;        % Ball + Sin carga
        case '21'
            L16 = 10;       % Ball + Carga baja
        case '22'
            L16 = 11;       % Ball + Carga media
        case '23'
            L16 = 12;       % Ball + Carga alta
        case '30'
            L16 = 13;       % Outer + Sin carga
        case '31'
            L16 = 14;       % Outer + Carga baja
        case '32'
            L16 = 15;       % Outer + Carga media
        case '33'
            L16 = 16;       % Outer + Carga alta
    end
    
    % ---------------------------------------------------------------------
    % Segmentación de la base de datos
    % ---------------------------------------------------------------------
    
    % Inicialización de las variables de segmentación
    Init  = 1;
    Final = Lseg;
    
    % Segmentación de la base de datos    
    for j = 1:Nseg
        M_DE{Pseg+j,1}   = Signal1(Init:Final)';
        M_FE{Pseg+j,1}   = Signal2(Init:Final)';
        Init             = Init  + Lseg;
        Final            = Final + Lseg;

        % Vector columna de las etiquetas
        Labels4(Pseg+j,1)  = L4;
        Labels10(Pseg+j,1) = L10;
        Labels16(Pseg+j,1) = L16;
    end
    
    % Aumento de la posición de almancenamiento de las celdas
    Pseg = Pseg + Nseg;
    
end

Vibr_DE = cell2mat(M_DE);
Vibr_FE = cell2mat(M_FE);

OutFile = [Folder '[Western][48k][Preprocessing][4096]'];

save(OutFile,'Vibr_DE','Vibr_FE','Labels4','Labels10','Labels16')



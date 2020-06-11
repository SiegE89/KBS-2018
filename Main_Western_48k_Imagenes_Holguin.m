% =========================================================================
% Algoritmo para generar las gráficas utilizados en el artículo de Mauricio
% Holguín 2017
% -------------------------------------------------------------------------
% Implementado por: Ing. José Alberto Hernández Muriel
% =========================================================================

%% Preinicialización
clc; clear all; close all; format compact;

%% Carga de los resultados obtenidos
% Carga del Path de ubicación
% Path = ['C:\Users\SiegE89\'];       % Desde SiegE89
Path = ('C:\Users\Usuario UTP\');   % Desde UTP

% Carpeta donde se encuentran los resultados
Folder = ('Google Drive\Proyecto_Vibraciones\Database\Bearings\Western\48 kHz\');

% Nombre del archivo de la base de datos
Data = ('[Western][48k][Preprocessing][4096].mat');

% Carga del archivo
load([Path Folder Data]);

%% Carga de las carpetas que contienen las funciones necesarias

% Carpeta con las funciones de preprocesamiento
Prep_Folder = ['Google Drive\Proyecto_Vibraciones\Codigos\Vibr_Toolbox\Vibr_Preprocessing'];
% Carpeta con las funciones de Caracterización
Feat_Folder = ['Google Drive\Proyecto_Vibraciones\Codigos\Vibr_Toolbox\Vibr_Caracterization'];
% Carpeta con las funciones de Caracterización (Mel-Cepstrum)
MFCC_Folder = ['Google Drive\Proyecto_Vibraciones\Codigos\Vibr_Toolbox\Vibr_Caracterization\voicebox'];

% Agregar las carpetas al espacio de trabajo de Matlab
addpath(genpath([Path Prep_Folder]))
addpath(genpath([Path Feat_Folder]))
addpath(genpath([Path MFCC_Folder]))

%% Gráfica No. 1 - Señales en el tiempo y la frecuencia
ind_c1 = find(Labels4==1);
ind_c2 = find(Labels4==2);
ind_c3 = find(Labels4==3);
ind_c4 = find(Labels4==4);

Fs     = 48e3;
t = linspace(0,0.853,4096);

figure(1)
subplot(4,2,1)
plot(t,Vibr_DE(ind_c1(1),:),'b')
axis([0 0.85 -0.3 0.3])
ylabel('Amplitude [V]')
title('Signal in time domain')

subplot(4,2,3)
plot(t,Vibr_DE(ind_c2(1),:),'b')
axis([0 0.85 -3 3])
ylabel('Amplitude [V]')

subplot(4,2,5)
plot(t,Vibr_DE(ind_c3(1),:),'b')
axis([0 0.85 -0.55 0.55])
xlabel('Outer')
ylabel('Amplitude [V]')

subplot(4,2,7)
plot(t,Vibr_DE(ind_c4(1),:),'b')
axis([0 0.85 -5 5])
ylabel('Amplitude [V]')
xlabel('Time [s]')

% Espectros en frecuencia
Ns = 4096;
[Spect1 Freq] = Vibr_FFT(Vibr_DE(ind_c1(1),:),Fs,Ns,0);
[Spect2 Freq] = Vibr_FFT(Vibr_DE(ind_c2(1),:),Fs,Ns,0);
[Spect3 Freq] = Vibr_FFT(Vibr_DE(ind_c3(1),:),Fs,Ns,0);
[Spect4 Freq] = Vibr_FFT(Vibr_DE(ind_c4(1),:),Fs,Ns,0);

subplot(4,2,2)
plot(Freq,Spect1,'r')
axis([0 20000 0 0.07])
title('Signal in frequency domain')
xlabel('Normal')

subplot(4,2,4)
plot(Freq,Spect2,'r')
axis([0 20000 0 0.25])
xlabel('Ball')

subplot(4,2,6)
plot(Freq,Spect3,'r')
axis([0 20000 0 0.08])
xlabel('Inner')

subplot(4,2,8)
plot(Freq,Spect4,'r')
axis([0 20000 0 0.7])

xlabel('Frequency [Hz]')

%% Gráfica No. 2 - Curvas de rendimiento
% Nombre del archivo de la base de datos
Data = ['[Western][48k][feat6][Xdat2][FeatSel][HMM][Q2][M2][Full][C4].mat'];
% Data = ['[Western][48k][feat6][Xdat2][FeatSel][HMM][Q2][M2][Full][C10].mat'];
% 
% % Carga del archivo
load([Path Folder Data]);

% load('[Western][FeatSel][Set2][c10]-p3.mat')

% Plot (si Plot = 1, error bar)
Plot = 1;

[~,Nw]  = size(Mir);

for i=1:10
    v1 = acc_hmm(5,16,i);
    v2 = acc_hmm(5,18,i);
    acc_hmm(5,17,i) = (v1+v2)/2;
end

h = figure; %knn learning curve
hold on
P = numel(vv);
cm = [0 0 1; 1 0 0; 0 0.5 0.3; 0.5 0 0.5; 1 0.5 0];
acc_m_hmm = zeros(P,Nw);
acc_s_hmm = zeros(P,Nw);

for z = 1: 5
    acc_m_hmm(:,z) = mean(squeeze(acc_hmm(z,:,:)),2);
    acc_s_hmm(:,z) = std(squeeze(acc_hmm(z,:,:)),[],2);
    j = Nw-z;
    if Plot == 1
        errorbar(vv,acc_m_hmm(:,z),acc_s_hmm(:,z),'Color',cm(j,:),'LineWidth',2);
    else
        plot(vv,acc_m_hmm(:,z),'LineWidth',2,'Color',cm(j,:))
    end
    grid on
end

featselname = {'VRA';'Self-weigth';'LaplacianScore';'Dist.-weigth';'Relieff'};
legend(featselname,'Location','southeast')
xlabel('Number of relevant features')
ylabel('Classification accuracy [%]')
set(gca,'FontSize',16)
axis([0 53 50 100])

%% Caso: [Q2][M2][Full][C4]
ind = [26 21 17 7 5];

for k = 1:Nw-1

    j = Nw - k;
    jj = ind(k);
    
    Line  = vv(jj)*ones(1,100);
    Yaxis = linspace(20,100,100);
    plot(Line,Yaxis,'LineStyle','--','LineWidth',2,'Color',cm(j,:))
    Mean = acc_m_hmm(jj,k)
    Std  = acc_s_hmm(jj,k)

end

%% Caso: [Q2][M2][Full][C10]
ind = [24 19 17 4 4];

for k = 1:Nw-1

    j = Nw-k;
    jj = ind(k);
    
    Line  = vv(jj)*ones(1,100);
    Yaxis = linspace(20,100,100);
    plot(Line,Yaxis,'LineStyle','--','LineWidth',2,'Color',cm(j,:))
    Mean = acc_m_hmm(jj,k)
    Std  = acc_s_hmm(jj,k)

end

%% Gráfica No. 4 - Matriz de confusión

%% Matriz de confusión: Mean of Accuracy with standard deviation (C=4)
figure(2)
subplot(1,2,1)
imagesc(mean(Cmhmm(:,:,5,ind(5),:),5))
title('Best accuracy for SFS (Mean)')
set(gca,'FontSize',10)
caxis([0,100])
colorbar
names = {'N';'B';'IR';'OR'};
set(gca,'xtick',[1:4],'xticklabel',names)
set(gca,'ytick',[1:4],'yticklabel',names)
set(gca,'XTickLabelRotation',90)
set(gca,'FontSize',30)
subplot(1,2,2)
imagesc(std(Cmhmm(:,:,5,ind(5),:),1,5))
title('Best accuracy for SFS (Std)')
caxis([0,2])
colorbar
names = {'N';'B';'IR';'OR'};
set(gca,'xtick',[1:4],'xticklabel',names)
set(gca,'ytick',[1:4],'yticklabel',names)
set(gca,'XTickLabelRotation',90)
set(gca,'FontSize',30)

%% Matriz de confusión: Mean of Accuracy with standard deviation (C=10)
figure(2)
subplot(1,2,1)
imagesc(mean(Cmhmm(:,:,5,ind(5),:),5))
title('Best accuracy for SFS (Mean)')
set(gca,'FontSize',10)
caxis([0,100])
colorbar
names = {'N';'B1';'B2';'B3';'IR1';'IR2';'IR3';'OR1';'OR2';'OR3'};
set(gca,'xtick',[1:10],'xticklabel',names)
set(gca,'ytick',[1:10],'yticklabel',names)
set(gca,'XTickLabelRotation',90)
set(gca,'FontSize',30)
subplot(1,2,2)
imagesc(std(Cmhmm(:,:,5,ind(5),:),1,5))
title('Best accuracy for SFS (Std)')
caxis([0,2])
colorbar
names = {'N';'B1';'B2';'B3';'IR1';'IR2';'IR3';'OR1';'OR2';'OR3'};
set(gca,'xtick',[1:10],'xticklabel',names)
set(gca,'ytick',[1:10],'yticklabel',names)
set(gca,'XTickLabelRotation',90)
set(gca,'FontSize',30)

%% Gráfica No. 3 - Diagramas de barra

% Extracción de los indices de las características
Aux = Mir(1:vv(ind(5)),5);
T   = 0;
F1  = 0;
F2  = 0;
TF  = 0;

for i = 1:length(Aux);
    Val = Aux(i);
    if Val>=1  && Val<=18
        T = T+1;
    end
    if Val>=19 && Val<=23
        F1 = F1 + 1;
    end
    if Val>=24 && Val<=41
        F2 = F2 + 1;
    end
    if Val>=42
        TF = TF + 1;
    end
end

Feat = [T F1 F2 TF];
Feat = 100.*Feat./(T+F1+F2+TF);

figure(3)
bar(Feat)
ylabel('Number of relevance features [%]')
ax = gca;
xticks = get(ax,'XTickLabel');
xticks(1) = {'Time'};
xticks(2) = {'Freq1'};
xticks(3) = {'Freq2'};
xticks(4) = {'Time-Freq'};
set(gca,'FontSize',16)

set(gca,'xticklabel',xticks)

%% Imagesc de las características

% Nombre del resultado a analizar
Result = ['[Western][48k][Xdat2][feat6][Rank][C4].mat'];
% Result = ['[Western][48k][Xdat2][feat6][Rank][C10].mat'];

% Carga del archivo
load([Path Folder Result]);

%% Gráfica No. 4 Imagesc con todas las características
M = 53;     % Número de filas
N = 7;      % Número de columnas
% Inicialización de la matriz de pesos
W_RF  = zeros(M,N);

Init  = 1;
Final = M;

for i = 1:N
    W_RF(:,i) = w_rf(Init:Final);
    Init  = Init  + M;
    Final = Final + M;
end
   
figure(4)
imagesc(W_RF)
grid on
xlabel('Number of analized window in time domain')
ylabel('Number of feature')
set(gca,'FontSize',16)
colorbar

%% 

load('G:\Mi unidad\Proyecto_Vibraciones\Database\Bearings\Western\48 kHz\[Western][48k][feat6][Xdat2][FeatSel][HMM][Q2][M2][Full][C4].mat')

%% Proyección de los datos
% Cargar la matriz de caracterización
load('G:\Mi unidad\Proyecto_Vibraciones\Database\Bearings\Western\48 kHz\[Western][48k][feat6].mat')

% Carga de los vectores de Ranking
load('G:\Mi unidad\Proyecto_Vibraciones\Database\Bearings\Western\48 kHz\[Western][48k][Xdat2][feat6][Rank][c4].mat')
% load('G:\Mi unidad\Proyecto_Vibraciones\Database\Bearings\Western\48 kHz\[Western][48k][Xdat2][feat6][Rank][c10].mat')

% Bolsa de palabras
load('G:\Mi unidad\Proyecto_Vibraciones\Database\Bearings\Western\48 kHz\[Western][Xdat2][feat6][BoW][c4].mat')

% Carpeta con las funciones de preprocesamiento
Folder = ['G:\Mi unidad\Proyecto_Vibraciones\Codigos\MatToolsAM\'];

% Agregar las carpetas al espacio de trabajo de Matlab
addpath(genpath([Folder]))

% Para C = 4
ind = [26 21 17 7 5];

% % Para C = 10
% ind = [24 19 17 4 4];

X = zeros(5610,53*7);

for i = 1:5610
    ii = 0;
    for j = 1:53
        for k = 1:7
            ii = ii + 1;
            X(i,ii) = Xdat2(j,k,i);
        end
    end
end

% Matriz de distancias ----------------------------------------------------
Dx  = pdist2(Xn,Xn);
% Sigma
sig   = median(squareform(Dx));
% Kernel gaussiano sobre las distancias
Ksig  = exp(-Dx.^2/(2*sig^2));  
% Valor de perpexidad con ecualización de la entropía
Perpl = [10,-1];
% Selección del vector de etiquetas
Labels = Labels4;

% -------------------------------------------------------------------------
best = 53*7;
Xr = X(inds,i_rf(1:best));

% Normalización de los datos ----------------------------------------------
[Xn,uz,sz] = zscore(Xr);
% [Xn,maxX,minX] = drnormalization(Xn);


% PCA ---------------------------------------------------------------------
Xpr_pca1 = A_pca(Xn,2);

figure(5)
scatter(Xpr_pca1(:,1),Xpr_pca1(:,2),40,Labels(inds),'filled')
colorbar
title('PCA')

Xpr_pca2 = A_pca(Xn,3);

figure(6)
scatter3(Xpr_pca2(:,1),Xpr_pca2(:,2),Xpr_pca2(:,3),40,Labels(inds),'filled')
colorbar
title('PCA')

fprintf('PCA - done\n')

% -------------------------------------------------------------------------

% KPCA --------------------------------------------------------------------
Xpr_kpca1 = A_kpca(Ksig,2);

figure(7)
scatter(Xpr_kpca1(:,1),Xpr_kpca1(:,2),40,Labels(inds),'filled')
colorbar
title('K-PCA')

Xpr_kpca2 = A_kpca(Ksig,3);

figure(8)
scatter3(Xpr_kpca2(:,1),Xpr_kpca2(:,2),Xpr_kpca2(:,3),40,Labels(inds),'filled')
colorbar
title('K-PCA')

fprintf('K-PCA - done\n')

% CCA ---------------------------------------------------------------------
Xpr_cca1 = cca(Ksig,2);

figure(9)
scatter(Xpr_cca1(:,1),Xpr_cca1(:,2),40,Labels(inds),'filled')
colorbar
title('CCA')

Xpr_cca2 = cca(Ksig,3);

figure(10)
scatter3(Xpr_cca2(:,1),Xpr_cca2(:,2),Xpr_cca2(:,3),40,Labels(inds),'filled')
colorbar
title('CCA')

fprintf('CCA - done\n')

% NeRV --------------------------------------------------------------------
Xpr_nerv1 = sbdr_abd({Ksig,Labels(inds)},2,Perpl,'n');

figure(11)
scatter(Xpr_nerv1(:,1),Xpr_nerv1(:,2),40,Labels(inds),'filled')
colorbar
title('NeRV')

Xpr_nerv2 = sbdr_abd({Ksig,Labels(inds)},3,Perpl,'n');

figure(12)
scatter3(Xpr_nerv2(:,1),Xpr_nerv2(:,2),Xpr_nerv2(:,3),40,Labels(inds),'filled')
colorbar
title('NeRV')

fprintf('NeRV - done\n')

% JSE --------------------------------------------------------------------
Xpr_jse1 = sbdr_abd({Ksig,Labels(inds)},2,Perpl,'j');

figure(13)
scatter(Xpr_jse1(:,1),Xpr_jse1(:,2),40,Labels(inds),'filled')
colorbar
title('JSE')

Xpr_jse2 = sbdr_abd({Ksig,Labels(inds)},3,Perpl,'j');

figure(14)
scatter3(Xpr_jse2(:,1),Xpr_jse2(:,2),Xpr_jse2(:,3),40,Labels(inds),'filled')
colorbar
title('JSE')

fprintf('JSE - done\n')

% SNE ---------------------------------------------------------------------
Xpr_sne1 = sbdr_abd({Ksig,Labels(inds)},2,Perpl,'s');

figure(15)
scatter(Xpr_sne1(:,1),Xpr_sne1(:,2),40,Labels(inds),'filled')
colorbar
title('SNE')

Xpr_sne2 = sbdr_abd({Ksig,Labels(inds)},3,Perpl,'s');

figure(16)
scatter3(Xpr_sne2(:,1),Xpr_sne2(:,2),Xpr_sne2(:,3),40,Labels(inds),'filled')
colorbar
title('SNE')

fprintf('SNE - done\n')

% t-SNE -------------------------------------------------------------------
Xpr_tsne1 = sbdr_abd({Ksig,Labels(inds)},2,Perpl,'t');

figure(17)
scatter(Xpr_tsne1(:,1),Xpr_tsne1(:,2),40,Labels(inds),'filled')
colorbar
title('t-SNE')

Xpr_tsne2 = sbdr_abd({Ksig,Labels(inds)},3,Perpl,'t');

figure(18)
scatter3(Xpr_tsne2(:,1),Xpr_tsne2(:,2),Xpr_tsne2(:,3),40,Labels(inds),'filled')
colorbar
title('t-SNE')

fprintf('t-SNE - done\n')

save('[Western][feat6][projection]','Xpr_cca1','Xpr_cca2','Xpr_jse1','Xpr_jse2','Xpr_kpca1','Xpr_kpca2','Xpr_nerv1','Xpr_nerv2','Xpr_pca1','Xpr_pca2','Xpr_sne1','Xpr_sne2','Xpr_tsne1','Xpr_tsne2')

%% Gráfica de nube de puntos
load('[Western][feat6][projection].mat')
load('G:\Mi unidad\Proyecto_Vibraciones\Database\Bearings\Western\48 kHz\[Western][48k][feat6].mat')
load('G:\Mi unidad\Proyecto_Vibraciones\Database\Bearings\Western\48 kHz\[Western][Xdat2][feat6][BoW][c4].mat')

Labels = Labels4;

% PCA ---------------------------------------------------------------------
figure(5)
scatter(Xpr_pca1(:,1),Xpr_pca1(:,2),40,Labels(inds),'filled')
colorbar
title('PCA')

figure(6)
scatter3(Xpr_pca2(:,1),Xpr_pca2(:,2),Xpr_pca2(:,3),40,Labels(inds),'filled')
colorbar
title('PCA')

% KPCA --------------------------------------------------------------------
figure(7)
scatter(Xpr_kpca1(:,1),Xpr_kpca1(:,2),40,Labels(inds),'filled')
colorbar
title('K-PCA')

figure(8)
scatter3(Xpr_kpca2(:,1),Xpr_kpca2(:,2),Xpr_kpca2(:,3),40,Labels(inds),'filled')
colorbar
title('K-PCA')

% CCA ---------------------------------------------------------------------
figure(9)
scatter(Xpr_cca1(:,1),Xpr_cca1(:,2),40,Labels(inds),'filled')
colorbar
title('CCA')

figure(10)
scatter3(Xpr_cca2(:,1),Xpr_cca2(:,2),Xpr_cca2(:,3),40,Labels(inds),'filled')
colorbar
title('CCA')

% NeRV --------------------------------------------------------------------
figure(11)
scatter(Xpr_nerv1(:,1),Xpr_nerv1(:,2),40,Labels(inds),'filled')
colorbar
title('NeRV')

figure(12)
scatter3(Xpr_nerv2(:,1),Xpr_nerv2(:,2),Xpr_nerv2(:,3),40,Labels(inds),'filled')
colorbar
title('NeRV')

% JSE --------------------------------------------------------------------
figure(13)
scatter(Xpr_jse1(:,1),Xpr_jse1(:,2),40,Labels(inds),'filled')
colorbar
title('JSE')

figure(14)
scatter3(Xpr_jse2(:,1),Xpr_jse2(:,2),Xpr_jse2(:,3),40,Labels(inds),'filled')
colorbar
title('JSE')

% SNE ---------------------------------------------------------------------
figure(15)
scatter(Xpr_sne1(:,1),Xpr_sne1(:,2),40,Labels(inds),'filled')
colorbar
title('SNE')

figure(16)
scatter3(Xpr_sne2(:,1),Xpr_sne2(:,2),Xpr_sne2(:,3),40,Labels(inds),'filled')
colorbar
title('SNE')

% t-SNE -------------------------------------------------------------------
figure(17)
scatter(Xpr_tsne1(:,1),Xpr_tsne1(:,2),40,Labels(inds),'filled')
colorbar
title('t-SNE')

figure(18)
scatter3(Xpr_tsne2(:,1),Xpr_tsne2(:,2),Xpr_tsne2(:,3),40,Labels(inds),'filled')
colorbar
title('t-SNE')

%%

% Cargar la matriz de caracterización
load('G:\Mi unidad\Proyecto_Vibraciones\Database\Bearings\Western\48 kHz\[Western][48k][feat6].mat')

% Carga de los vectores de Ranking
load('G:\Mi unidad\Proyecto_Vibraciones\Database\Bearings\Western\48 kHz\[Western][48k][Xdat2][feat6][Rank][c4].mat')

% Bolsa de palabras
load('G:\Mi unidad\Proyecto_Vibraciones\Database\Bearings\Western\48 kHz\[Western][Xdat2][feat6][BoW][c4].mat')

% Carpeta con las funciones de preprocesamiento
Folder = ['G:\Mi unidad\Proyecto_Vibraciones\Codigos\MatToolsAM\'];

% Agregar las carpetas al espacio de trabajo de Matlab
addpath(genpath([Folder]))

% Para C = 4
ind = [26 21 17 7 5];

X = zeros(5610,53*7);

for i = 1:5610
    ii = 0;
    for j = 1:53
        for k = 1:7
            ii = ii + 1;
            X(i,ii) = Xdat2(j,k,i);
        end
    end
end

% Matriz de distancias ----------------------------------------------------
Dx  = pdist2(Xn,Xn);
% Sigma
sig   = median(squareform(Dx));
% Kernel gaussiano sobre las distancias
Ksig  = exp(-Dx.^2/(2*sig^2));  
% Valor de perpexidad con ecualización de la entropía
Perpl = [100,-1];
% Selección del vector de etiquetas
Labels = Labels4;

% -------------------------------------------------------------------------
best = 53*7;
Xr = X(inds,i_rf(1:best));

% Normalización de los datos ----------------------------------------------
[Xn,uz,sz] = zscore(Xr);
% [Xn,maxX,minX] = drnormalization(Xn);


% KPCA --------------------------------------------------------------------
Xpr_kpca1 = A_kpca(Ksig,2);

figure(9)
scatter(Xpr_kpca1(:,1),Xpr_kpca1(:,2),40,Labels(inds),'filled')
colorbar
title('K-PCA')

Xpr_kpca2 = A_kpca(Ksig,3);

figure(10)
scatter3(Xpr_kpca2(:,1),Xpr_kpca2(:,2),Xpr_kpca2(:,3),40,Labels(inds),'filled')
title('K-PCA')

fprintf('K-PCA - done\n')


%%

% Cargar la matriz de caracterización
load('G:\Mi unidad\Proyecto_Vibraciones\Database\Bearings\Western\48 kHz\[Western][48k][feat6].mat')

% Carga de los vectores de Ranking
load('G:\Mi unidad\Proyecto_Vibraciones\Database\Bearings\Western\48 kHz\[Western][48k][Xdat2][feat6][Rank][c10].mat')
% load('G:\Mi unidad\Proyecto_Vibraciones\Database\Bearings\Western\48 kHz\[Western][48k][Xdat2][feat6][Rank][c10].mat')

% Bolsa de palabras
load('G:\Mi unidad\Proyecto_Vibraciones\Database\Bearings\Western\48 kHz\[Western][Xdat2][feat6][BoW][c10].mat')

% Carpeta con las funciones de preprocesamiento
Folder = ['G:\Mi unidad\Proyecto_Vibraciones\Codigos\MatToolsAM\'];

% Agregar las carpetas al espacio de trabajo de Matlab
addpath(genpath([Folder]))

% Para C = 4
% ind = [26 21 17 7 5];

% % Para C = 10
ind = [24 19 17 4 4];

X = zeros(5610,53*7);

for i = 1:5610
    ii = 0;
    for j = 1:53
        for k = 1:7
            ii = ii + 1;
            X(i,ii) = Xdat2(j,k,i);
        end
    end
end

% Matriz de distancias ----------------------------------------------------
Dx  = pdist2(Xn,Xn);
% Sigma
sig   = median(squareform(Dx));
% Kernel gaussiano sobre las distancias
Ksig  = exp(-Dx.^2/(2*sig^2));  
% Valor de perpexidad con ecualización de la entropía
Perpl = [10,-1];
% Selección del vector de etiquetas
Labels = Labels10;

% -------------------------------------------------------------------------
best = 53*7;
Xr = X(inds,i_rf(1:best));

% Normalización de los datos ----------------------------------------------
[Xn,uz,sz] = zscore(Xr);
% [Xn,maxX,minX] = drnormalization(Xn);


% KPCA --------------------------------------------------------------------
Xpr_kpca1 = A_kpca(Ksig,2);

figure(7)
scatter(Xpr_kpca1(:,1),Xpr_kpca1(:,2),40,Labels(inds),'filled')
colorbar
title('K-PCA')

Xpr_kpca2 = A_kpca(Ksig,3);

figure(8)
scatter3(Xpr_kpca2(:,1),Xpr_kpca2(:,2),Xpr_kpca2(:,3),40,Labels(inds),'filled')
title('K-PCA')

fprintf('K-PCA - done\n')

%% Curva de relevancia
load('[Western][feat6][projection].mat')
load('G:\Mi unidad\Proyecto_Vibraciones\Database\Bearings\Western\48 kHz\[Western][48k][feat6].mat')
load('G:\Mi unidad\Proyecto_Vibraciones\Database\Bearings\Western\48 kHz\[Western][Xdat2][feat6][BoW][c4].mat')
load('G:\Mi unidad\Proyecto_Vibraciones\Database\Bearings\Western\48 kHz\[Western][48k][feat6][Xdat2][FeatSel][HMM][Q2][M2][Full][C4].mat')

Plot = 1;
cm = [0 0 1; 1 0 0; 0 0.5 0.3; 0.5 0 0.5; 1 0.5 0];


acc_m_hmm = mean(squeeze(acc_hmm(5,:,:)),2);
acc_s_hmm = std(squeeze(acc_hmm(5,:,:)),[],2);
j = Nw-z;
figure(11)
if Plot == 1
    errorbar(vv,acc_m_hmm,acc_s_hmm,'Color',cm(j,:),'LineWidth',2);
else
    plot(vv,acc_m_hmm,'LineWidth',2,'Color',cm(j,:))
end
grid on

% Para C = 4
% ind = [26 21 17 7 5];
ind = [26 21 17 7 6];

figure(12)
subplot(1,2,1)
imagesc(mean(Cmhmm(:,:,5,ind(3),:),5))
title('Accuracy: best result for Laplacian-Score (Mean)')
set(gca,'FontSize',14)
caxis([0,100])
colorbar
subplot(1,2,2)
imagesc(std(Cmhmm(:,:,5,ind(3),:),1,5))
title('Accuracy: best result for Laplacian-Score (Std)')
set(gca,'FontSize',16)
caxis([0,100])
colorbar

%% Curva de relevancia
load('[Western][feat6][projection].mat')
load('G:\Mi unidad\Proyecto_Vibraciones\Database\Bearings\Western\48 kHz\[Western][48k][feat6].mat')
load('G:\Mi unidad\Proyecto_Vibraciones\Database\Bearings\Western\48 kHz\[Western][Xdat2][feat6][BoW][c10].mat')
load('G:\Mi unidad\Proyecto_Vibraciones\Database\Bearings\Western\48 kHz\[Western][48k][feat6][Xdat2][FeatSel][HMM][Q2][M2][Full][C10].mat')

Plot = 1;
cm = [0 0 1; 1 0 0; 0 0.5 0.3; 0.5 0 0.5; 1 0.5 0];


acc_m_hmm = mean(squeeze(acc_hmm(5,:,:)),2);
acc_s_hmm = std(squeeze(acc_hmm(5,:,:)),[],2);
j = Nw-z;
figure(13)
if Plot == 1
    errorbar(vv,acc_m_hmm,acc_s_hmm,'Color',cm(j,:),'LineWidth',2);
else
    plot(vv,acc_m_hmm,'LineWidth',2,'Color',cm(j,:))
end
grid on

% Para C = 4
% ind = [26 21 17 7 5];
ind = [26 21 17 7 6];

figure(14)
subplot(1,2,1)
imagesc(mean(Cmhmm(:,:,5,ind(3),:),5))
title('Accuracy: best result for Laplacian-Score (Mean)')
set(gca,'FontSize',14)
caxis([0,100])
colorbar
subplot(1,2,2)
imagesc(std(Cmhmm(:,:,5,ind(3),:),1,5))
title('Accuracy: best result for Laplacian-Score (Std)')
set(gca,'FontSize',16)
caxis([0,100])
colorbar

%% CKA
% Cargar la matriz de caracterización
load('G:\Mi unidad\Proyecto_Vibraciones\Database\Bearings\Western\48 kHz\[Western][48k][feat6].mat')

% Carga de los vectores de Ranking
load('G:\Mi unidad\Proyecto_Vibraciones\Database\Bearings\Western\48 kHz\[Western][48k][Xdat2][feat6][Rank][c4].mat')
% load('G:\Mi unidad\Proyecto_Vibraciones\Database\Bearings\Western\48 kHz\[Western][48k][Xdat2][feat6][Rank][c10].mat')

% Bolsa de palabras
load('G:\Mi unidad\Proyecto_Vibraciones\Database\Bearings\Western\48 kHz\[Western][Xdat2][feat6][BoW][c4].mat')

% Carpeta con las funciones de preprocesamiento
Folder = ['G:\Mi unidad\Proyecto_Vibraciones\Codigos\MatToolsAM\'];

% Agregar las carpetas al espacio de trabajo de Matlab
addpath(genpath([Folder]))

% Para C = 4
ind = [26 21 17 7 5];

% % Para C = 10
% ind = [24 19 17 4 4];

X = zeros(5610,53*7);

for i = 1:5610
    ii = 0;
    for j = 1:53
        for k = 1:7
            ii = ii + 1;
            X(i,ii) = Xdat2(j,k,i);
        end
    end
end

% Matriz de distancias ----------------------------------------------------
Dx  = pdist2(Xn,Xn);
% Sigma
sig   = median(squareform(Dx));
% Kernel gaussiano sobre las distancias
Ksig  = exp(-Dx.^2/(2*sig^2));  
% Valor de perpexidad con ecualización de la entropía
Perpl = [100,-1];
% Selección del vector de etiquetas
Labels = Labels4;

% -------------------------------------------------------------------------
best = 53*7;
Xr = X(inds,i_rf(1:best));

% Normalización de los datos ----------------------------------------------
[Xn,uz,sz] = zscore(Xr);
% [Xn,maxX,minX] = drnormalization(Xn);


% CKA ---------------------------------------------------------------------
[rind,w,A] = ckamlfrank(Xn,Labels(inds));

Xpr_cka = Xn*A;

figure(8)
scatter3(Xpr_cka(:,1),Xpr_cka(:,2),Xpr_cka(:,3),40,Labels(inds),'filled')
title('CKA-based projection')

set(gca,'FontSize',30)

%% CKA
% Cargar la matriz de caracterización
load('G:\Mi unidad\Proyecto_Vibraciones\Database\Bearings\Western\48 kHz\[Western][48k][feat6].mat')

% Carga de los vectores de Ranking
% load('G:\Mi unidad\Proyecto_Vibraciones\Database\Bearings\Western\48 kHz\[Western][48k][Xdat2][feat6][Rank][c4].mat')
load('G:\Mi unidad\Proyecto_Vibraciones\Database\Bearings\Western\48 kHz\[Western][48k][Xdat2][feat6][Rank][c10].mat')

% Bolsa de palabras
load('G:\Mi unidad\Proyecto_Vibraciones\Database\Bearings\Western\48 kHz\[Western][Xdat2][feat6][BoW][c10].mat')

% Carpeta con las funciones de preprocesamiento
Folder = ['G:\Mi unidad\Proyecto_Vibraciones\Codigos\MatToolsAM\'];

% Agregar las carpetas al espacio de trabajo de Matlab
addpath(genpath([Folder]))

% Para C = 4
% ind = [26 21 17 7 5];

% % Para C = 10
ind = [24 19 17 4 4];

X = zeros(5610,53*7);

for i = 1:5610
    ii = 0;
    for j = 1:53
        for k = 1:7
            ii = ii + 1;
            X(i,ii) = Xdat2(j,k,i);
        end
    end
end

% Matriz de distancias ----------------------------------------------------
Dx  = pdist2(Xn,Xn);
% Sigma
sig   = median(squareform(Dx));
% Kernel gaussiano sobre las distancias
Ksig  = exp(-Dx.^2/(2*sig^2));  
% Valor de perpexidad con ecualización de la entropía
Perpl = [100,-1];
% Selección del vector de etiquetas
Labels = Labels10;

% -------------------------------------------------------------------------
best = 53*7;
Xr = X(inds,i_rf(1:best));

% Normalización de los datos ----------------------------------------------
[Xn,uz,sz] = zscore(Xr);
% [Xn,maxX,minX] = drnormalization(Xn);


% CKA ---------------------------------------------------------------------
[rind,w,A] = ckamlfrank(Xn,Labels(inds));

Xpr_cka = Xn*A;

figure(9)
scatter3(Xpr_cka(:,1),Xpr_cka(:,2),Xpr_cka(:,3),40,Labels(inds),'filled')
title('CKA')

figure(9)
scatter(Xpr_cka(:,1),Xpr_cka(:,2),40,Labels(inds),'filled')
title('CKA-based projection')
grid on

set(gca,'FontSize',30)
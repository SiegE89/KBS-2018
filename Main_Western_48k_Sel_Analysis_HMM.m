% =========================================================================
% Algoritmo para visualizar los datos obtenidos mediante selección de
% características
% -------------------------------------------------------------------------
% Entrada: Archivo .mat que contiene los resultados obtenidos durante la
% selección de características
% Salida: Gráfica de los resultados obtenidos
% -------------------------------------------------------------------------
% Implementado por: Ing. José Alberto Hernández Muriel
% =========================================================================

%% Preinicialización
clc; clear all; close all; format compact;

%% Carga de los resultados obtenidos
% Carga del Path de ubicación
% Path = ['C:\Users\SiegE89\'];       % Desde SiegE89
Path = ['C:\Users\Usuario UTP\'];   % Desde UTP

% Carpeta donde se encuentran los resultados
Folder = ['Google Drive\Proyecto_Vibraciones\Database\Bearings\Western\48 kHz\'];

% Nombre del resultado a analizar
% Result = ['[Western][48k][feat6][Xdat2][FeatSel][HMM][Q2][M2][Full][C4].mat'];
Result = ['[Western][48k][feat6][Xdat2][FeatSel][HMM][Q2][M2][Full][C10].mat'];
% Result = ['[Western][48k][feat6][Xdat2][FeatSel][HMM][Q2][M2][Full][C16].mat'];

% Carga del archivo
load([Path Folder Result]);

%% Construcción de la curva de rendimiento
[~,Nw]  = size(Mir);

h = figure; %knn learning curve
hold on
P = numel(vv);
cm = [0 0 1; 1 0 0; 0 0.5 0.3; 0.5 0 0.5; 1 0.5 0];
% cm = jet(2*Nw-2);
acc_m_hmm = zeros(P,Nw);
acc_s_hmm = zeros(P,Nw);
for z = 1 : Nw-1
    acc_m_hmm(:,z) = mean(squeeze(acc_hmm(z,:,:)),2);
    acc_s_hmm(:,z) = std(squeeze(acc_hmm(z,:,:)),[],2);
    j = Nw-z;
    %     j = z*2-1;
%     errorbar(vv,acc_m_hmm(:,z),acc_s_hmm(:,z),'Color',cm(j,:),'LineWidth',2);
    plot(vv,acc_m_hmm(:,z),'LineWidth',2,'Color',cm(j,:))
    grid on
end

% featselname = {'PCA';'Self-weigth';'LaplacianScore';'Dist.-weigth';'Relieff';'CKAML'};
featselname = {'PCA';'Self-weigth';'LaplacianScore';'Dist.-weigth';'Relieff'};
legend(featselname,'Location','southeast')
xlabel('Number of relevant features')
ylabel('Classification accuracy [%]')
set(gca,'FontSize',16)
axis([0 53 50 100])

%% Puntos máximos de las diferentes curvas
[ii jj] = max(acc_m_hmm)
vv(jj)
acc_s_hmm(jj,:)

%% Caso: [Q2][M2][Full][C4]
ind = [26 21 4 7 5 22];

for k = 1:Nw-1
%     j = k*2-1;
    j = Nw - k;
    jj = ind(k);
    
    Line  = vv(jj)*ones(1,100);
    Yaxis = linspace(20,100,100);
    plot(Line,Yaxis,'LineStyle','--','LineWidth',2,'Color',cm(j,:))
    Mean = acc_m_hmm(jj,k)
    Std  = acc_s_hmm(jj,k)

end

% title('Feature selection for 4 classes problem with HMM ([Q2][M2][Full])')

%% Caso: [Q2][M2][Full][C10]
ind = [24 19 4 4 4 22];

for k = 1:Nw-1
%     j = k*2-1;
    j = Nw-k;
    jj = ind(k);
    
    Line  = vv(jj)*ones(1,100);
    Yaxis = linspace(20,100,100);
    plot(Line,Yaxis,'LineStyle','--','LineWidth',2,'Color',cm(j,:))
    Mean = acc_m_hmm(jj,k)
    Std  = acc_s_hmm(jj,k)

end

% title('Feature selection for 10 classes problem with HMM ([Q2][M2][Full])')

%% Matriz de confusión: Mean of Accuracy with standard deviation
figure(2)
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

%% Análisis del porcentaje del tipo de características utilizadas
% Extracción de los indices de las características
Aux = Mir(1:vv(ind(3)),3);
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
% title('Relevance features analysis (Laplacian-Score with HMM)')
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
% Result = ['[Western][48k][Xdat2][feat6][Rank][C4].mat'];
Result = ['[Western][48k][Xdat2][feat6][Rank][C4].mat'];

% Carga del archivo
load([Path Folder Result]);

Ranking = w_ls;
% % Ranking = zeros(size(w_ls));
% Ranking(i_ls(1:vv(ind(3)))) = w_ls(i_ls(1:vv(ind(3))));

M = 53;     % Número de filas
N = 7;      % Número de columnas
% Inicialización de la matriz de pesos
W_LS = zeros(M,N);

Init  = 1;
Final = M;


Ranking = zeros(size(w_ls));

for i=1:N
    for j=1:ind(3)*N
        if i == i_ls(j)
            Ranking(i) = w_ls(i);
        else
            Ranking(i) = 0;
        end
    end
end       
        

for i = 1:N
    W_LS(:,i) = w_ls(Init:Final);
    Init  = Init  + M;
    Final = Final + M;
end

figure(4)
imagesc(W_LS)
grid on
% axis([0 8 0 54])
xlabel('Number of analized window in time domain')
ylabel('Number of feature')
set(gca,'FontSize',16)
colorbar

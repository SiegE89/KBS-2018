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

% Nombre del resultado a analizar (Set1)
% Result = ['[Western][FeatSel][feat6][Set1][c4].mat'];      %  4 clases
% Result = ['[Western][FeatSel][feat6][Set1][c10].mat'];     % 10 clases
Result = ['[Western][FeatSel][feat6][Set1][c16].mat'];     % 16 clases

% Nombre del resultado a analizar (Set2)
% Result = ['[Western][FeatSel][Set2][c4].mat'];              %  4 clases
% Result = ['[DHMM][FeatSel].mat']

% Carga del archivo
load([Path Folder Result]);

acc_knn = accDHMM;
[~,Nw]  = size(Mir); 

%% Para el set de características 1
% close all
h = figure; %knn learning curve
hold on
P = numel(vv);
cm = jet(2*Nw);
acc_m_knn = zeros(P,Nw);
acc_s_knn = zeros(P,Nw);
for z = 1 : Nw
    acc_m_knn(:,z) = mean(squeeze(acc_knn(z,:,:)),2);
    acc_s_knn(:,z) = std(squeeze(acc_knn(z,:,:)),[],2);
    j = z*2-1;
%     errorbar(vv,acc_m_knn(:,z),acc_s_knn(:,z),'Color',cm(j,:));
    plot(vv,acc_m_knn(:,z),'LineWidth',2,'Color',cm(j,:))
    grid on
end

featselname = {'PCA';'Self-weigth';'LaplacianScore';'Dist.-weigth';'Relieff';'CKAML'};
legend(featselname,'Location','southeast')
title('Feature selection for 4 classes problem with DHMM')
xlabel('Relevant feature')
ylabel('DHMM Classificaton accuracy [%]')
set(gca,'FontSize',14)
axis([0 53 40 90])

%%
[ii jj] = max(acc_m_knn)
vv(jj)
acc_s_knn(jj,:)


%% Caso: Western 4 clases con Fs = 48kHz

% ind = [1 6 2 3 2 2];
ind = [1 4 2 1 2 1];

for k = 1:Nw
    j = k*2-1;
    
    jj = ind(k);
    
    Line  = vv(jj)*ones(1,100);
    Yaxis = linspace(20,100,100);
    plot(Line,Yaxis,'LineStyle','--','LineWidth',2,'Color',cm(j,:))
    Mean = acc_m_knn(jj,k)
    Std  = acc_s_knn(jj,k)

end

%% Caso: Western 10 clases con Fs = 48kHz

ind = [31 28 21 23 22 26];

for k = 1:Nw
    j = k*2-1;
    
    jj = ind(k);
    
    Line  = vv(jj)*ones(1,100);
    Yaxis = linspace(50,100,100);
    plot(Line,Yaxis,'LineStyle','--','LineWidth',2,'Color',cm(j,:))
    Mean = acc_m_knn(jj,k)
    Std  = acc_s_knn(jj,k)

end

%% Caso: Western 16 clases con Fs = 48kHz

ind = [3 2 5 4 3 2];

for k = 1:Nw
    
    j = k*2-1;
    
    jj = ind(k);
    
    Line  = vv(jj)*ones(1,100);
    Yaxis = linspace(50,100,100);
    plot(Line,Yaxis,'LineStyle','--','LineWidth',2,'Color',cm(j,:))
    Mean = acc_m_knn(jj,k)
    Std  = acc_s_knn(jj,k)

end

%% Matriz de confusión: Mean of Accuracy with standard deviation
Cmknn = CmDHMM;
figure(2)
subplot(1,2,1)
imagesc(mean(Cmknn(:,:,5,ind(3),:),5))
title('Accuracy: best result for Laplacian-Score (Mean)')
set(gca,'FontSize',14)
caxis([0,100])
colorbar
subplot(1,2,2)
imagesc(std(Cmknn(:,:,5,ind(3),:),1,5))
title('Accuracy: best result for Laplacian-Score (Std)')
set(gca,'FontSize',14)
caxis([0,100])
colorbar

%% Análisis del porcentaje del tipo de características utilizadas
% Extracción de los indices de las características
Aux = Mir(vv(1:ind(3)),5);
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
title('Relevance features analysis (Laplacian-Score with DHMM)')
ax = gca;
xticks = get(ax,'XTickLabel');
xticks(1) = {'Time'};
xticks(2) = {'Freq1'};
xticks(3) = {'Freq2'};
xticks(4) = {'Time-Freq'};
set(gca,'FontSize',14)



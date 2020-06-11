% =========================================================================
% Pruebas finales Western + HMM
% -------------------------------------------------------------------------
% Implementado por: Ing. José Alberto Hernández-Muriel
% =========================================================================

%% Preinicialización
clc; clear all; close all;

%% Path de ubicación
Path = ('G:\Mi unidad\Proyecto_Vibraciones\');

%% Cargar funciones necesarias para el procesamiento de los datos
% Agregar las carpetas al espacio de trabajo de Matlab
addpath(genpath([Path 'Codigos']))


%% Proyección de los datos - Problema de 4 clases
% Cargar la matriz de caracterización
load([Path 'Database\Bearings\Western\48 kHz\[Western][48k][feat6].mat'])

% Carga de los vectores de Ranking
load([Path 'Database\Bearings\Western\48 kHz\[Western][48k][Xdat2][feat6][Rank][c4].mat'])

% Bolsa de palabras
load([Path 'Database\Bearings\Western\48 kHz\[Western][Xdat2][feat6][BoW][c4].mat'])

% Inicialización de variables
ind = [26 21 17 7 5];       % Guía de índices

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

% Sub-conjunto de características relevantes ------------------------------
best = 53*7;
Xr = X(inds,i_rf(1:best));

% Normalización de los datos ----------------------------------------------
[Xn,uz,sz] = zscore(Xr);

% PCA ---------------------------------------------------------------------
Xpr_pca1 = A_pca(Xn,2);
Xpr_pca2 = A_pca(Xn,3);
fprintf('PCA - done\n')

% K-PCA -------------------------------------------------------------------
Xpr_kpca1 = A_kpca(Ksig,2);
Xpr_kpca2 = A_kpca(Ksig,3);
fprintf('K-PCA - done\n')

% CCA ---------------------------------------------------------------------
Xpr_cca1 = cca(Ksig,2);
Xpr_cca2 = cca(Ksig,3);
fprintf('CCA - done\n')

% NeRV --------------------------------------------------------------------
Xpr_nerv1 = sbdr_abd({Ksig,Labels(inds)},2,Perpl,'n');
Xpr_nerv2 = sbdr_abd({Ksig,Labels(inds)},3,Perpl,'n');
fprintf('NeRV - done\n')

% JSE --------------------------------------------------------------------
Xpr_jse1 = sbdr_abd({Ksig,Labels(inds)},2,Perpl,'j');
Xpr_jse2 = sbdr_abd({Ksig,Labels(inds)},3,Perpl,'j');
fprintf('JSE - done\n')

% SNE ---------------------------------------------------------------------
Xpr_sne1 = sbdr_abd({Ksig,Labels(inds)},2,Perpl,'s');
Xpr_sne2 = sbdr_abd({Ksig,Labels(inds)},3,Perpl,'s');
fprintf('SNE - done\n')

% t-SNE -------------------------------------------------------------------
Xpr_tsne1 = sbdr_abd({Ksig,Labels(inds)},2,Perpl,'t');
Xpr_tsne2 = sbdr_abd({Ksig,Labels(inds)},3,Perpl,'t');
fprintf('t-SNE - done\n')

% CKA ---------------------------------------------------------------------
[rind,w,A] = ckamlfrank(Xn,Labels(inds));

Xpr_cka = Xn*A;

%% Generación de las gráficas ----------------------------------------------
figure(1)
scatter(Xpr_pca1(:,1),Xpr_pca1(:,2),40,Labels(inds),'filled')
% colorbar
title('PCA')

figure(2)
scatter3(Xpr_pca2(:,1),Xpr_pca2(:,2),Xpr_pca2(:,3),40,Labels(inds),'filled')
% colorbar
title('PCA')

figure(3)
scatter(Xpr_kpca1(:,1),Xpr_kpca1(:,2),40,Labels(inds),'filled')
% colorbar
title('K-PCA')

figure(4)
scatter3(Xpr_kpca2(:,1),Xpr_kpca2(:,2),Xpr_kpca2(:,3),40,Labels(inds),'filled')
% colorbar
title('K-PCA')

figure(5)
scatter(Xpr_cca1(:,1),Xpr_cca1(:,2),40,Labels(inds),'filled')
% colorbar
title('CCA')

figure(6)
scatter3(Xpr_cca2(:,1),Xpr_cca2(:,2),Xpr_cca2(:,3),40,Labels(inds),'filled')
% colorbar
title('CCA')

figure(7)
scatter(Xpr_nerv1(:,1),Xpr_nerv1(:,2),40,Labels(inds),'filled')
% colorbar
title('NeRV')

figure(8)
scatter3(Xpr_nerv2(:,1),Xpr_nerv2(:,2),Xpr_nerv2(:,3),40,Labels(inds),'filled')
% colorbar
title('NeRV')

figure(9)
scatter(Xpr_jse1(:,1),Xpr_jse1(:,2),40,Labels(inds),'filled')
% colorbar
title('JSE')

figure(10)
scatter3(Xpr_jse2(:,1),Xpr_jse2(:,2),Xpr_jse2(:,3),40,Labels(inds),'filled')
% colorbar
title('JSE')

figure(11)
scatter(Xpr_sne1(:,1),Xpr_sne1(:,2),40,Labels(inds),'filled')
% colorbar
title('SNE')

figure(12)
scatter3(Xpr_sne2(:,1),Xpr_sne2(:,2),Xpr_sne2(:,3),40,Labels(inds),'filled')
% colorbar
title('SNE')

figure(13)
scatter(Xpr_tsne1(:,1),Xpr_tsne1(:,2),40,Labels(inds),'filled')
% colorbar
title('t-SNE')

figure(14)
scatter3(Xpr_tsne2(:,1),Xpr_tsne2(:,2),Xpr_tsne2(:,3),40,Labels(inds),'filled')
% colorbar
title('t-SNE')

figure(15)
scatter(Xpr_cka(:,1),Xpr_cka(:,2),40,Labels(inds),'filled')
% colorbar
title('CKA')

figure(16)
scatter3(Xpr_cka(:,1),Xpr_cka(:,2),Xpr_cka(:,3),40,Labels(inds),'filled')
% colorbar
title('CKA')

showfigs_c(4)

save('[Western][feat6][projection]','Xpr_cca1','Xpr_cca2','Xpr_jse1','Xpr_jse2','Xpr_kpca1','Xpr_kpca2','Xpr_nerv1','Xpr_nerv2','Xpr_pca1','Xpr_pca2','Xpr_sne1','Xpr_sne2','Xpr_tsne1','Xpr_tsne2','Xpr_cka')

%% Proyección de los datos - Problema de 4 clases
% Cargar la matriz de caracterización
load([Path 'Database\Bearings\Western\48 kHz\[Western][48k][feat6].mat'])

% Carga de los vectores de Ranking
load([Path 'Database\Bearings\Western\48 kHz\[Western][48k][Xdat2][feat6][Rank][c10].mat'])

% Bolsa de palabras
load([Path 'Database\Bearings\Western\48 kHz\[Western][Xdat2][feat6][BoW][c10].mat'])

% Inicialización de variables
ind = [24 19 17 4 4];       % Guía de índices

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

% Sub-conjunto de características relevantes ------------------------------
best = 53*7;
Xr = X(inds,i_rf(1:best));

% Normalización de los datos ----------------------------------------------
[Xn,uz,sz] = zscore(Xr);

% PCA ---------------------------------------------------------------------
Xpr_pca1 = A_pca(Xn,2);
Xpr_pca2 = A_pca(Xn,3);
fprintf('PCA - done\n')

% K-PCA -------------------------------------------------------------------
Xpr_kpca1 = A_kpca(Ksig,2);
Xpr_kpca2 = A_kpca(Ksig,3);
fprintf('K-PCA - done\n')

% CCA ---------------------------------------------------------------------
Xpr_cca1 = cca(Ksig,2);
Xpr_cca2 = cca(Ksig,3);
fprintf('CCA - done\n')

% NeRV --------------------------------------------------------------------
Xpr_nerv1 = sbdr_abd({Ksig,Labels(inds)},2,Perpl,'n');
Xpr_nerv2 = sbdr_abd({Ksig,Labels(inds)},3,Perpl,'n');
fprintf('NeRV - done\n')

% JSE --------------------------------------------------------------------
Xpr_jse1 = sbdr_abd({Ksig,Labels(inds)},2,Perpl,'j');
Xpr_jse2 = sbdr_abd({Ksig,Labels(inds)},3,Perpl,'j');
fprintf('JSE - done\n')

% SNE ---------------------------------------------------------------------
Xpr_sne1 = sbdr_abd({Ksig,Labels(inds)},2,Perpl,'s');
Xpr_sne2 = sbdr_abd({Ksig,Labels(inds)},3,Perpl,'s');
fprintf('SNE - done\n')

% t-SNE -------------------------------------------------------------------
Xpr_tsne1 = sbdr_abd({Ksig,Labels(inds)},2,Perpl,'t');
Xpr_tsne2 = sbdr_abd({Ksig,Labels(inds)},3,Perpl,'t');
fprintf('t-SNE - done\n')

% CKA ---------------------------------------------------------------------
[rind,w,A] = ckamlfrank(Xn,Labels(inds));

Xpr_cka = Xn*A;
fprintf('CKA - done\n')

%% Generación de las gráficas ----------------------------------------------
figure(1)
scatter(Xpr_pca1(:,1),Xpr_pca1(:,2),40,Labels(inds),'filled')
% colorbar
title('PCA')

figure(2)
scatter3(Xpr_pca2(:,1),Xpr_pca2(:,2),Xpr_pca2(:,3),40,Labels(inds),'filled')
% colorbar
title('PCA')

figure(3)
scatter(Xpr_kpca1(:,1),Xpr_kpca1(:,2),40,Labels(inds),'filled')
% colorbar
title('K-PCA')

figure(4)
scatter3(Xpr_kpca2(:,1),Xpr_kpca2(:,2),Xpr_kpca2(:,3),40,Labels(inds),'filled')
% colorbar
title('K-PCA')

figure(5)
scatter(Xpr_cca1(:,1),Xpr_cca1(:,2),40,Labels(inds),'filled')
% colorbar
title('CCA')

figure(6)
scatter3(Xpr_cca2(:,1),Xpr_cca2(:,2),Xpr_cca2(:,3),40,Labels(inds),'filled')
% colorbar
title('CCA')

figure(7)
scatter(Xpr_nerv1(:,1),Xpr_nerv1(:,2),40,Labels(inds),'filled')
% colorbar
title('NeRV')

figure(8)
scatter3(Xpr_nerv2(:,1),Xpr_nerv2(:,2),Xpr_nerv2(:,3),40,Labels(inds),'filled')
% colorbar
title('NeRV')

figure(9)
scatter(Xpr_jse1(:,1),Xpr_jse1(:,2),40,Labels(inds),'filled')
% colorbar
title('JSE')

figure(10)
scatter3(Xpr_jse2(:,1),Xpr_jse2(:,2),Xpr_jse2(:,3),40,Labels(inds),'filled')
% colorbar
title('JSE')

figure(11)
scatter(Xpr_sne1(:,1),Xpr_sne1(:,2),40,Labels(inds),'filled')
% colorbar
title('SNE')

figure(12)
scatter3(Xpr_sne2(:,1),Xpr_sne2(:,2),Xpr_sne2(:,3),40,Labels(inds),'filled')
% colorbar
title('SNE')

figure(13)
scatter(Xpr_tsne1(:,1),Xpr_tsne1(:,2),40,Labels(inds),'filled')
% colorbar
title('t-SNE')

figure(14)
scatter3(Xpr_tsne2(:,1),Xpr_tsne2(:,2),Xpr_tsne2(:,3),40,Labels(inds),'filled')
% colorbar
title('t-SNE')

figure(15)
scatter(Xpr_cka(:,1),Xpr_cka(:,2),40,Labels(inds),'filled')
% colorbar
title('CKA')

figure(16)
colormap jet
scatter3(Xpr_cka(:,1),Xpr_cka(:,2),Xpr_cka(:,3),40,Labels(inds),'filled')
title('CKA')
axis([-0.4 0.3 -0.5 0.8 -0.5 1])
% showfigs_c(4)

% save('[Western][feat6][projection][c10]','Xpr_cca1','Xpr_cca2','Xpr_jse1','Xpr_jse2','Xpr_kpca1','Xpr_kpca2','Xpr_nerv1','Xpr_nerv2','Xpr_pca1','Xpr_pca2','Xpr_sne1','Xpr_sne2','Xpr_tsne1','Xpr_tsne2','Xpr_cka')


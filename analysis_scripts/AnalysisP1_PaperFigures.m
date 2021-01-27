% Paper Figures: Produce all the plots for the paper Project 1

get(0,'Factory');
set(0,'defaultfigurecolor',[1 1 1])


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Figure 1 = task image
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% FIGURE 2 - Group Psychometric Plots & perceptual weights
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear
close all
cd('W:\Project 1\log\EEG Data');
addpath('W:\Project 1\log\EEG Data\functions');
addpath('Z:\Lab\ckmatlab\ckstatistics')

% load files
load('groupPsyCurves.mat');                                                     % psychometric curves
load('groupPsyWeights.mat');                                                    % psychometric perceptual weights
load('groupData.mat','BWeights','TAX','R');                                     % behavioural regression weights & correlation  
load('W:/Project 1/log/EEG Data/Analysis_AzBoot/azBoot_beh.mat','as');          % bootstrap Az distribution percentiles for behaviour
load('Cluster_BehW_HighLow_Comparison.mat','NC_Aud','NC_Vis');                  % Cluster perm results, Beh Regression Weights : high vs low comparisons                        
load('Cluster_BehPsyCurve_Correlation.mat','PC_corrAV_beh');                    % Cluster perm results, Beh Regression Weights vs. Psychometric Curve Weights           

% colours/names of conditions set up 
cc{1} = [0.5600 0 1.0000; 0.7200 0.5300 0.1500; 0.1900 0.7300 0.5600];
cc{2} = [0 1 0; 0 0 1;1 0 0 ];
cc{3} = [0 1 0; 0 0 1;1 0 0 ];
names{1} = {'VH','VL','AUD'};
names{2} = {'AVH0','AVH+2','AVH-2'};
names{3} = {'AVL0','AVL+2','AVL-2'};

figure('units','normalized','outerposition',[0 0 0.6 1])
cd('W:\Project 1\log\EEG Data');

%--------------------------------------------------------------------------
% Plot psychometric curves 
for k = 1:3;                                                                    % 1 = unisensory, 2 = ms high, 3 = ms low 
    subplot(3,3,k);
    h = zeros(1,3); 
    for m = 1:3                                                                 % m = conditions within unisensory/ms high/ms low
        data = groupM{k}(:,:,m);
        plotpd(data,'color', cc{k}(m,:),'LineWidth',3);
        hold on
        
        shape = 'cumulative Gaussian';
        prefs = batch('shape', shape, 'n_intervals', 1, 'runs', 2000);
        outputPrefs = batch('write_pa', 'pa', 'write_th', 'th','write_st','st');
        h2 = psignifit(data, [prefs outputPrefs]);
        
        % Plot the fit to the original data
        h(:,m) = plotpf(shape, pa.est,'color', cc{k}(m,:),'LineWidth',3);
        drawHeights = psi(shape, pa.est, th.est);
        
    end
    
    vline(11,':k'); hline(0.5,':k'); ylim([0 1]);
    if k==1; xlabel('Event Rate (Hz)'); ylabel('Proportion 1st Stream Choices'); title('Unisensory Group')
    else xlabel('Average Event Rate (Hz)'); end
    if k==2; title('AH/VH Group'); elseif k==3 ; title('AH/VL Group'); end
    set(gca,'YTick',(0:0.1:1))
    legend(h(1,:),names{k},'Location','NorthWest','FontSize',8);
end
%--------------------------------------------------------------------------


%--------------------------------------------------------------------------
% Perceptual Weights from psychometric curves
% observed weights difference
awAll = awObs; awAll(:,3:4) = 1-awObs;                                          % awObs = auditory weights. Visual weights are 1 - auditory weights. 
subplot 334
hold on; 
plot([1 2],awObs(:,1:2)',':.k'); 
plot([3 4],awAll(:,3:4)',':.k')
boxplot(awAll); 
set(gca,'XTick',1:4,'XTickLabel',{'AVH Aud','AVL Aud','AVH Vis','AVL Vis'}); 
ylabel('Observed Weight');
%--------------------------------------------------------------------------


%--------------------------------------------------------------------------
% difference between Predicted and Observed Auditory Weights 
subplot 335; hold on
scatterplot(awObs(:,1),awP(:,1))                                                % observed weights against predicted Auditory
scatterplot(awObs(:,2),awP(:,2),'Color','m')                                    % observed weights against predicted Visual 
axis([0.3 1 0.3 1])
line([0.3,1],[0.3,1],'Color',[0.7 0.7 0.7])
legend({'High Rel','Low Rel'},'Location','NorthWest','FontSize',6)
ylabel('Predicted Auditory Weights')
xlabel('Observed Auditory Weights')
%--------------------------------------------------------------------------


%--------------------------------------------------------------------------
% Individual subjects differences 
subplot 336
bar(awObs(:,2)-awObs(:,1));                                                     % low reliability minus high reliability
set(gca,'XTick',1:20); xlim([0 21]); 
ylabel('AVL Aud - AVH Aud'); 
%--------------------------------------------------------------------------


%--------------------------------------------------------------------------
% AZ: Regression Analysis 
I = 3:51;                                                                       % index for time window cutoff
subplot 337; hold on
plot(TAX(I),sq(mean(BWeights(:,I,6),1)))                                        % plot the Az value (stored in BWeights,6); 
hline(as(2))                                                                    % add percentile
xlabel('Time(ms)'); ylabel('Mean Group ROC (Az)')
xlim([TAX(I(1)) TAX(I(end))])
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
% Behavioural regression weights 
subplot 338; hold on
% auditory:
errorbar(TAX(I),mean(BWeights(:,I,1)),sem(BWeights(:,I,1)));
errorbar(TAX(I),mean(BWeights(:,I,3)),sem(BWeights(:,I,3)),'Color',[0.5 0.8 0.8]);
% visual:
errorbar(TAX(I),mean(BWeights(:,I,2)),sem(BWeights(:,I,2)),'Color',[1 0.5 0]);
errorbar(TAX(I),mean(BWeights(:,I,4)),sem(BWeights(:,I,4)),'Color',[1 0.9 0.5]);
legend({'AVH Aud','AVL Aud','AVH Vis','AVL Vis'},'Location','NorthWest')
ylabel('Behavioural Regresssion Weight'); xlabel('Time (ms)')

% add in significant time points where there is a difference between high
% and low for each modality individually
ind1 = find(NC_Aud.maskSig~=0);     % aud sig
ind2 = find(NC_Vis.maskSig~=0);     % vis sig
plot(TAX(ind1),mean(BWeights(:,ind1,3)),'.k'); 
plot(TAX(ind2),mean(BWeights(:,ind2,4)),'.k'); 
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
% Behavioural - Behavioural correlation
subplot 339; hold on; 
plot(TAX(I),R{3}(I)); 
xlim([TAX(I(1)) TAX(I(end))]); 
ind =  find(PC_corrAV_beh.maskSig~=0);
plot(TAX(ind),R{3}(ind),'.k');
xlabel('Time (ms)'); ylabel('R (PMC Weights vs Beh Reg Weights)')
%--------------------------------------------------------------------------

% print('Figure2','-depsc');



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Figure 3 - Neural Weights, Decoder, and Neuro-Behavioural Correlation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% close all 
load('groupData'); 
load('W:/Project 1/log/EEG Data/Analysis_AzBoot/azBoot.mat','temp2')            % Distribtion prcntiles Neural Decoder
load('Cluster_NeuW_BehW_Correlation','PC_corr')                                 % Cluster perm results, Neuro:Behaviorual Reg weight correlation
load('Cluster_NeuW_HighLow_Comparison.mat','NC_NW_AUD','PC_NW_VIS');            % Cluster perm results, Neural Weights: high/low reliability comparison
load('Cluster_NeuW_AV_Comparison.mat','PC_AV_HIGH',...                          
    'PC_AV_LOW','NC_AV_HIGH','NC_AV_LOW');                                      % Cluster perm results, Neural Weights: aud/vis comparisons                

figure('units','normalized','outerposition',[0 0 0.6 0.9])

%--------------------------------------------------------------------------
% Decoding performance
subplot 421; hold on
plot(TAX(I),sq(mean(AzGroup(:,I))));
[I2,J2] = max(mean(AzGroup(:,I)));
hline(temp2(2));                                                                % add in percentiles derived from bootstrap
ylim([0.5 0.65 ])
ylabel('AVH AVL Rate AZ')
xlim([TAX(I(1)) TAX(I(end))])
xlabel('Time (ms)'); 
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
% Neural Weights: auditory vs visual comparison for different reliabilities
indAH = find(PC_AV_HIGH.mask~=0);                                               % ah vs. vh difference
indVH = find(NC_AV_HIGH.mask~=0);                                               % ah vs. vh difference
indAL = find(PC_AV_LOW.mask~=0);                                                % al vs. vl differenec
indVL = find(NC_AV_LOW.mask~=0);                                                % al vs. vl difference 

% high reliability weights 
subplot 423; hold on; 
errorbar(TAX(I),mean(NeuralWeights(:,I,1)),sem(NeuralWeights(:,I,1)))
errorbar(TAX(I),mean(NeuralWeights(:,I,2)),sem(NeuralWeights(:,I,2)),'Color',[1 0.5 0]);
legend('AH','VH')
plot(TAX(indAH),mean(NeuralWeights(:,indAH,1)),'.k')                            % add significant points
plot(TAX(indVH),mean(NeuralWeights(:,indVH,2)),'.k')                            % add significant points 
title('high rel'); 
hline(0,':k');
ylim([-1 5]); xlim([TAX(I(1)) TAX(I(end))])

% low reliability weights 
subplot 424; hold on; 
errorbar(TAX(I),mean(NeuralWeights(:,I,3)),sem(NeuralWeights(:,I,3)),'Color',[0.5 0.8 0.8])
errorbar(TAX2,mean(NeuralWeights(:,I,4)),sem(NeuralWeights(:,I,4)),'Color',[1 0.9 0.5]);
legend('AL','VL')
hold on
plot(TAX(indAL),mean(NeuralWeights(:,indAL,3)),'*k')
plot(TAX(indVL),mean(NeuralWeights(:,indVL,4)),'*k')
title('low rel'); 
hline(0,':k');
ylim([-1 5]); xlim([TAX(I(1)) TAX(I(end))])
%--------------------------------------------------------------------------


%--------------------------------------------------------------------------
% Neural Weights: high vs low reliability comparison for different
% modalities

% auditory modality weights 
ind1 = find(NC_NW_AUD.maskSig~=0);
subplot 425; hold on;
errorbar(TAX(I),mean(NeuralWeights(:,I,1)),sem(NeuralWeights(:,I,1))); hold on
errorbar(TAX(I),mean(NeuralWeights(:,I,3)),sem(NeuralWeights(:,I,3)),'Color',[0.5 0.8 0.8]);
legend({'AH','AL'},'Location','NorthEast','FontSize',8);
plot(TAX(ind1),mean(NeuralWeights(:,ind1,3)),'.k')
hline(0,':k');
ylabel('Regression Weight');
title('Auditory Neural Weights')
ylim([-1 5]); xlim([TAX(I(1)) TAX(I(end))])

% visual modality weights 
subplot 426; hold on;
ind2 = find(PC_NW_VIS.maskSig~=0);
errorbar(TAX(I),mean(NeuralWeights(:,I,2)),sem(NeuralWeights(:,I,2)),'Color',[1 0.5 0]);
errorbar(TAX2,mean(NeuralWeights(:,I,4)),sem(NeuralWeights(:,I,4)),'Color',[1 0.9 0.5]);
legend({'VH','VL'},'Location','NorthEast','FontSize',8);
hline(0,':k');
plot(TAX(ind2),mean(NeuralWeights(:,ind2,2)),'.k')
ylabel('Regression Weight');
title('Visual Neural Weights ')
ylim([-1 5]); xlim([TAX(I(1)) TAX(I(end))])
%--------------------------------------------------------------------------


%--------------------------------------------------------------------------
% Correlation: Neural Weights/Regression Weights 
subplot 427; hold on
plot(TAX(I),(R{1}(I)));
ind =  find(PC_corr.maskSig~=0);
hold on
plot(TAX(ind),R{1}(ind),'.k');
xlim([TAX(I(1)) TAX(I(end))])
xlabel('Time (ms)'); ylabel('R value')
%--------------------------------------------------------------------------

% print('Figure3','-depsc');



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Figure 4 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all 
figure('units','normalized','outerposition',[0 0 0.5 0.9])
% significant clusters 
ind1 = [8:12];  % 0.12 : 0.132
ind2 = 14:20;    % 156:228
ind3 = 22:24;   % 252:276

cluster1 = sq(mean(NeuralWeights(:,ind1,1:4),2)); 
cluster2 = sq(mean(NeuralWeights(:,ind2,1:4),2)); 
cluster3 = sq(mean(NeuralWeights(:,ind3,1:4),2)); 

% rearrange to be AH,AL, VH, VL 
cluster1_NEW = [cluster1(:,1),cluster1(:,3),cluster1(:,2),cluster1(:,4)];
cluster2_NEW = [cluster2(:,1),cluster2(:,3),cluster2(:,2),cluster2(:,4)];
cluster3_NEW = [cluster3(:,1),cluster3(:,3),cluster3(:,2),cluster3(:,4)];
load('ftDummy','EvpDummy'); 


subplot 131
boxplot(cluster1_NEW)
hold on
for s = 1:20; 
    plot(1:2,cluster1_NEW(s,1:2),'-o','MarkerSize',2,'Color',[0.7 0.7 0.7]); 
    plot(3:4,cluster1_NEW(s,3:4),'-o','MarkerSize',2,'Color',[0.7 0.7 0.7]);
end
title('Cluster 1 - 84:132 avg')
set(gca,'XTick',1:4,'XTickLabel',{'AH','AL','VH','VL'});
ylim([-1 5])

subplot 132
boxplot(cluster2_NEW)
hold on
for s = 1:20; 
    plot(1:2,cluster2_NEW(s,1:2),'-o','MarkerSize',2,'Color',[0.7 0.7 0.7]); 
    plot(3:4,cluster2_NEW(s,3:4),'-o','MarkerSize',2,'Color',[0.7 0.7 0.7]);
end
title('Cluster 2 - 156:228 avg')
set(gca,'XTick',1:4,'XTickLabel',{'AH','AL','VH','VL'});
ylim([-1 7])

subplot 133
boxplot(cluster3_NEW)
hold on
for s = 1:20; 
    plot(1:2,cluster3_NEW(s,1:2),'-o','MarkerSize',2,'Color',[0.7 0.7 0.7]); 
    plot(3:4,cluster3_NEW(s,3:4),'-o','MarkerSize',2,'Color',[0.7 0.7 0.7]);
end
title('Cluster 3 - 252:276 avg')
set(gca,'XTick',1:4,'XTickLabel',{'AH','AL','VH','VL'});
ylim([-1 10])

%%
subplot 222
topo1st = sq(mean(mean(Topo1(:,:,ind1),3),1)); 
cfg = []; 
cfg.layout = 'biosemi64.lay'; 
EvpDummy.avg = topo1st'; 
ft_topoplotER(cfg,EvpDummy); 
colorbar

subplot 223
boxplot(cluster2_NEW)
hold on
plot(1,cluster2_NEW(:,1),'.','MarkerSize',10,'Color',[0.7 0.7 0.7])
plot(2,cluster2_NEW(:,2),'.','MarkerSize',10,'Color',[0.7 0.7 0.7])
plot(3,cluster2_NEW(:,3),'.','MarkerSize',10,'Color',[0.7 0.7 0.7])
plot(4,cluster2_NEW(:,4),'.','MarkerSize',10,'Color',[0.7 0.7 0.7])
title('Cluster 2 - 204:228 avg')
set(gca,'XTick',1:4,'XTickLabel',{'AH','AL','VH','VL'});
ylim([-1 7])


subplot 224
topo2nd = sq(mean(mean(Topo1(:,:,ind2),3),1)); 
cfg = []; 
cfg.layout = 'biosemi64.lay'; 
EvpDummy.avg = topo2nd'; 
ft_topoplotER(cfg,EvpDummy); 
colorbar

print('Figure5','-depsc');

%% PLOT THE THREE SIGNIFICANT CONCATENATED CLUSTERS 


clear
% load('AVH_AVL_DIAG_RATE_NO11HZ_ALLTRIALS.mat','TAX','NeuralWeights','Topo1');
load('groupData.mat','TAX','NeuralWeights','Topo1');

% three clusters
ind{1} = 8:12;
ind{2} = 14:20;
ind{3} = 22:24;

times = cell(1,length(ind));
for k = 1:length(ind);
    times{k} = sprintf('%s:%s',num2str(TAX(ind{k}(1))),num2str(TAX(ind{k}(end))));
end

Clusters = zeros(20,4,length(ind));
TopoC = zeros(20,64,length(ind));
for k = 1:length(ind);
    Clusters(:,:,k) = sq(mean(NeuralWeights(:,ind{k},1:4),2));      % (AH-VH)-(AL-VL)
    TopoC(:,:,k) = sq(mean(Topo1(:,:,ind{k}),3));
end

% reshape it to be AH,AL,VH,VL
Clusters2 = [Clusters(:,1,:),Clusters(:,3,:),Clusters(:,2,:),Clusters(:,4,:)];
load('ftDummy','EvpDummy');


% load('AVH_AVL_DIAG_RATE_NO11HZ_ALLTRIALS.mat','BWeights')
load('groupData.mat','BWeights')



%%
clf
np = 1;
for k = 1:length(ind);

    % behavioural weights 
    subplot(length(ind),3,np)
    BWeights_Cluster(:,:,k) = sq(mean(BWeights(:,ind{k},1:4),2));
    BWeights_Cluster(:,:,k) = [BWeights_Cluster(:,1,k) BWeights_Cluster(:,3,k) BWeights_Cluster(:,2,k) BWeights_Cluster(:,4,k)];
    plot(1:2,BWeights_Cluster(:,1:2,k),':.','Color',[0.7 0.7 0.7],'MarkerSize',4.5); hold on; 
    plot(3:4,BWeights_Cluster(:,3:4,k),':.','Color',[0.7 0.7 0.7],'MarkerSize',4.5)
    boxplot(BWeights_Cluster(:,:,k));
    set(gca,'XTick',1:4,'XTickLabel',{'AH','AL','VH','VL'})
    title(times{k});
    ylim([-1 32])
    ylabel('Perceptual Weight')
    
    % neural weights 
    subplot(length(ind),3,np+3)
    
    plot(1:2,Clusters2(:,1:2,k),':.','Color',[0.7 0.7 0.7],'MarkerSize',4.5); hold on;
    plot(3:4,Clusters2(:,3:4,k),':.','Color',[0.7 0.7 0.7],'MarkerSize',4.5); hold on;
    boxplot(Clusters2(:,:,k)); title(times{k});
    ylim([-1.5 10])
    set(gca,'XTick',1:4,'XTickLabel',{'AH','AL','VH','VL'})
    ylabel('Neural Weight')
    
    % topographies
    subplot(length(ind),3,np+6)
    EvpDummy.avg = sq(mean(TopoC(:,:,k),1))';
    cfg = [];
    cfg.layout = 'biosemi64.lay';
    cfg.comment = ' ';
%     cfg.colorbar = 'yes';
colorbar
    
    ft_topoplotER(cfg,EvpDummy);
    np = np+1;
    
    title(times{k});
    
end

print('Figure6','-depsc');


%%

Diff = zeros(20,length(ind));
ADiff = zeros(20,length(ind));
VDiff = zeros(20,length(ind));
for k = 1:length(ind);
    % (AH-VH)-(AL-VL)
    Diff(:,k) = (Clusters2(:,1,k)-Clusters2(:,3,k)) - (Clusters2(:,2,k)-Clusters2(:,4,k));
    ADiff(:,k) = Clusters2(:,1,k)-Clusters2(:,2,k); % AH-AL
    VDiff(:,k) = Clusters2(:,3,k)-Clusters2(:,4,k); % VH-VL
    
end

clf
subplot 131
boxplot(Diff);
set(gca,'XTick',1:3,'XTickLabel',times)
hline(0,':k')
title('(AH-VH) - (AL-VL)');
ylim([-10 6])
set(gca,'YTick',-10:2:6)
ylabel('Neural Weights')

subplot 132
boxplot(ADiff);
set(gca,'XTick',1:3,'XTickLabel',times)
hline(0,':k')
title('(AL-AH)')
ylim([-6 4])
ylabel('Neural Weights')

subplot 133
boxplot(VDiff);
set(gca,'XTick',1:3,'XTickLabel',times)
hline(0,':k')
title('(VL-VH)')
ylim([-4 6])
ylabel('Neural Weights')

print('Figure7','-depsc');


%%

% load your data from above.
cd('W:\Project 1\log\EEG Data\EEG64Source');

load('C:\Program Files\MATLAB\R2012a\toolbox\fieldtrip-20150413\template\headmodel\standard_mri.mat')
load('W:\Project 1\log\EEG Data\EEG64Source\BioSemi_EEG_64_Headmodel_standard.mat')
load('W:\Project 1\log\EEG Data\EEG64Source\source_GROUP.mat');
load('W:\Project 1\log\EEG Data\EEG64Source\source_S1.mat','source');

% z-score correlations and compute average for each grid point and time
% pointof interest
for S=1:length(CORR_GROUP)%:18
    XX(S,:,:) = corr2z(CORR_GROUP{S});
end

% three clusters
ind{1} = 8:12;
ind{2} = 14:20;
ind{3} = 22:24;

% SIGNIFICANT CLUSTER TIME POINTS
% load('AVH_AVL_DIAG_RATE_NO11HZ_ALLTRIALS.mat','TAX');
load('groupData.mat','TAX');
for k = 1:length(ind);
    timings{k} = sprintf('%s to %s',num2str(TAX(ind{k}(1))),num2str(TAX(ind{k}(end))));
end

% CHANGE THIS
k = 3;

source2 = Cluster_Source{k}; 
t = ind{k};

tmp = sq(mean(mean(XX(:,:,t),1),3)); % avg across subjects
source.avg.pow = zeros(size(source.avg.pow));
source.avg.pow(find(grid.inside),:) = tmp;

% interpolate
maxl = max(tmp(:));
cfg            = [];
cfg.parameter  = 'pow';
source2  = ft_sourceinterpolate(cfg, source , mri);

cfg = [];
cfg.method        = 'ortho';
cfg.interactive   = 'yes';
cfg.funparameter  = 'pow';
cfg.atlas        = 'C:\Program Files\MATLAB\R2012a\toolbox\fieldtrip-20150413\template\atlas\aal\ROI_MNI_V4.nii';
cfg.maskparameter = cfg.funparameter;
cfg.funcolorlim   = [0 maxl];
cfg.opacitylim    = cfg.funcolorlim;
cfg.opacitymap    = 'rampup';
ft_sourceplot(cfg, source2);

cfg = [];
cfg.method         = 'surface';
cfg.interactive   = 'yes';
cfg.funparameter   = 'pow';
cfg.maskparameter  = cfg.funparameter;
cfg.funcolorlim   = [0 0.3];
% cfg.funcolormap    = 'jet';
cfg.opacitylim    = cfg.funcolorlim;
cfg.opacitymap     = 'rampup';  
cfg.projmethod     = 'nearest'; 
cfg.surffile       = 'surface_white_both.mat'; %Standard MNI brain
cfg.surfdownsample = 10;  % downsample to speed up processing
ft_sourceplot(cfg, source2);
view ([100 0]) 



%% SOURCE SIGNAL WEIGHTS
figure

load('source_GROUP.mat'); 

np = 1; 
for c = 1:3; 

    %------------------------------------------
    % neural weights:source
    
    subplot(3,2,np); hold on
    NWuse1 = sq(mean(REL{c}.NW(:,:,1)));
    NWuse2 = sq(mean(REL{c}.NW(:,:,3)));
    
    % AUDITORY HIGH 
    errorbar(TAX2,sq(mean(REL{c}.NW(:,:,1))),sq(sem(REL{c}.NW(:,:,1)))); hold on
    if isempty(REL{c}.AUD_POS)==0
        ind =  find(REL{c}.AUD_POS.maskSig~=0);
        plot(TAX2(ind),NWuse1(ind),'*k');
        indsave{c}.AUDPOS = TAX2(ind);
    end

    % AUDITORY LOW
    errorbar(TAX2,sq(mean(REL{c}.NW(:,:,3))),sq(sem(REL{c}.NW(:,:,3))),':g'); hold on
    ind = [];
    if isempty(REL{c}.AUD_NEG)==0
        ind =  find(REL{c}.AUD_NEG.maskSig~=0);
        indsave{c}.AUDNEG = TAX2(ind);
        plot(TAX2(ind),NWuse2(ind),'*k');
    end
    
    hline(50000,':k'); hline(-50000,':k')
    ylim([-100000 100000])
    xlim([TAX2(1) TAX2(end)])
    title(sprintf('%s \n NW SOURCE WEIGHTS: AUD',num2str(times{c})))
    ylabel('NEURAL WEIGHT value');
    hline(0,':k');
    legend({'HIGH','LOW'},'FontSize',8,'Location','SouthEast')
    % ylim([-0.8 0.8])

    
    %------------------------------------------
    % VISUAL WEIGHTS 
    subplot(3,2,np+1); hold on;
    NWuse3 = sq(mean(REL{c}.NW(:,:,2)));
    NWuse4 = sq(mean(REL{c}.NW(:,:,4)));
    
    % VISUAL HIGH
    errorbar(TAX2,sq(mean(REL{c}.NW(:,:,2))),sq(sem(REL{c}.NW(:,:,2))),'r'); hold on
    % plot(TAX2,NWuse3,'r');
    if isempty(REL{c}.VIS_POS)==0
        ind =  find(REL{c}.VIS_POS.maskSig~=0);
        plot(TAX2(ind),NWuse3(ind),'*k');
        indsave{c}.VISPOS = TAX2(ind);
    end
    
    % VISUAL LOW 
    errorbar(TAX2,sq(mean(REL{c}.NW(:,:,4))),sq(sem(REL{c}.NW(:,:,4))),':m'); hold on
    ind = [];
    if isempty(REL{c}.VIS_NEG)==0
        ind =  find(REL{c}.VIS_NEG.maskSig~=0);
        plot(TAX2(ind),NWuse4(ind),'*k');
        indsave{c}.VISNEG = TAX2(ind);
    end
    
    hline(50000,':k'); hline(-50000,':k')
    ylim([-100000 100000])
    xlim([TAX2(1) TAX2(end)])
    title(sprintf('%s \n NW SOURCE WEIGHTS: VIS',num2str(times{c})))
    ylabel('NEURAL WEIGHT value');
    hline(0,':k');
    legend({'HIGH','LOW'},'FontSize',8,'Location','SouthEast')
    np = np+2;

end

% print('Figure9','-depsc');



% %% NW_SOURCE_WEIGHTS CORRELATIONS 
% 
% 
% figure
% np = 1; 
% 
% for c = 1:3; 
% 
% %------------------------------------------
% % neural weights:source
% 
% % subplot(3,2,np)
% % plot(TAX2,T{c}.R_NDtrue)
% % hold on
% % 
% % if isempty(T{c}.ND_POS)==0
% % ind =  find(T{c}.ND_POS.maskSig~=0); 
% % plot(TAX2(ind),T{c}.R_NDtrue(ind),'*r');
% % end
% % 
% % ind = []; 
% % if isempty(T{c}.ND_NEG)==0
% %     ind =  find(T{c}.ND_NEG.maskSig~=0); 
% % plot(TAX2(ind),T{c}.R_NDtrue(ind),'*k');
% % end
% % 
% % title(sprintf('%s \n NW SOURCE WEIGHTS: ND WEIGHTS',num2str(times{c})))
% % ylabel('r value'); 
% % hline(0,':k'); 
% % % ylim([-0.8 0.8])
% 
% 
% 
% %------------------------------------------
% % behavioural:source
% subplot(3,1,c)
% plot(TAX2,T{c}.R_BDtrue)
% hold on
% 
% ind = []; 
% if isempty(T{c}.BD_POS)==0
% ind =  find(T{c}.BD_POS.maskSig~=0); 
% plot(TAX2(ind),T{c}.R_BDtrue(ind),'*r');
% end
% indS{c}.POS = TAX2(ind);
% 
% ind = []; 
% if isempty(T{c}.BD_NEG)==0
%     ind =  find(T{c}.BD_NEG.maskSig~=0); 
% plot(TAX2(ind),T{c}.R_BDtrue(ind),'*k');
% end
% indS{c}.NEG = TAX2(ind);
% 
% title(sprintf('%s \n NW SOURCE WEIGHTS: BD WEIGHTS',num2str(times{c})))
% ylabel('r value'); 
% hline(0,':k'); 
% ylim([-0.8 0.8])
%     xlim([TAX2(1) TAX2(end)])
% 
% np = np+2; 
% 
% end
% % print('Figure10','-depsc');
% 
% 

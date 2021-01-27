%% SOURCE LOCALISATION SCRIPT : RELIABILITY EXPERIMENT STEPH

clear; close all;
addpath('C:\Program Files\MATLAB\R2012a\toolbox\fieldtrip-20150413\forward');
savedir = 'W:/Project 1/log/EEG Data/EEG64Source';

%--------------------------------------------------------------------------
% Set up headmodel and leadfield
load('C:\Program Files\MATLAB\R2012a\toolbox\fieldtrip-20150413\template\headmodel\standard_mri.mat')                   % MRI template
load('C:\Program Files\MATLAB\R2012a\toolbox\fieldtrip-20150413\template\sourcemodel\standard_sourcemodel3d6mm.mat')    % default sourcemodel
load('C:\Program Files\MATLAB\R2012a\toolbox\fieldtrip-20150413\template\headmodel\standard_bem.mat')                   % standard bem model
sourcemodel = ft_convert_units(sourcemodel,'mm');
elec=ft_read_sens('standard_1020.elc');                              % electrode info
[labels,LabelOrig,cfg_elecpos] = eegck_BiosemiLabels();              % biosemi positions

% load aligned data
load('W:\Project 1\log\EEG Data\EEG64Source\BioSemi_EEG_64_Headmodel_elecaligned_test.mat','elec_aligned')

%----------------------------------------------------------
% load an atlas and interpolate with source model
atlasdir = 'C:\Program Files\MATLAB\R2012a\toolbox\fieldtrip-20150413\template\atlas\aal\ROI_MNI_V4.nii';
atlas = ft_read_atlas(atlasdir);
atlas = ft_convert_units(atlas, 'mm');
cfg = [];
cfg.interpmethod = 'nearest';
cfg.parameter = 'tissue';
sourceatlas = ft_sourceinterpolate(cfg, atlas, sourcemodel);
Excl = find(sourceatlas.tissue(:)>90);                              % exclude grid points with tissue > 90

%-----------------------------------------------------
% prepare leadfield
in = zeros(1,size(sourcemodel.pos,1));
in(sourcemodel.inside) = 1;
in(Excl) = 0;
cfgS        = [];
cfgS.grid   = sourcemodel;
cfgS.vol    = vol;
cfgS.elec      = elec_aligned;
cfgS.grid.inside = logical(in');
grid = ft_prepare_leadfield(cfgS);
save('W:\Project 1\log\EEG Data\EEG64Source\BioSemi_EEG_64_Headmodel_standard.mat','vol','sourcemodel','cfg_elecpos','elec_aligned','grid')

figure(1);clf;hold on;
ft_plot_sens(elec_aligned,'style', 'r.','label','off','coil','false');                          % show electrodes
ft_plot_mesh(vol.bnd(1),'facealpha', 0.85, 'edgecolor', 'none', 'facecolor', [0.65 0.65 0.65]); % scalp
ft_plot_mesh(sourcemodel, 'edgecolor', 'none'); camlight



%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% RUN CORRELATION OF SOURCE ACTIVITY AND DECODER
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subs = dir('W:\Project 1\log\EEG Data\Subj_*');

% LOAD DECODER RESULTS:
load('AVH_AVL_DIAG_RATE_NO11HZ_ALLTRIALS.mat','Y','trange','stimTimes');
trange = trange(stimTimes);             % timings

% Preprocessing settings
cfg_prepro = [];
cfg_prepro.lpfilter   = 'yes';          % apply lowpass filter
cfg_prepro.lpfreq     = 30;             % lowpass at 30 Hz.
cfg_prepro.hpfilter   = 'yes';          % apply highpass filter
cfg_prepro.hpfreq     = 1;
cfg_prepro.demean ='no';
cfg_prepro.reref = 'no';
cfg_prepro.refchannel = labels;

%--------------------------------------------------------------------------
% Source project the relevant trials
%--------------------------------------------------------------------------
clc
I = 3:51;                   % trange (24ms : 600ms - avoid colinearities and nulls)
trange2 = trange(I);

for subj = 1:length(subs);
    
    % load EEG and behavioural data from PMC (psychometric curve):
    cd(sprintf('W:/Project 1/log/EEG Data/%s',subs(subj).name))
    load(sprintf('Prepro_all_S0%d_EOG_out.mat',subj))
    
    % Preprocess EEG data again:
    data = ft_preprocessing(cfg_prepro,dataX);
    
    % compute covariance matrix
    cfg=[];
    cfg.covariance='yes';
    cfg.covariancewindow = [-0.2 1];        % this should cover the trial period of interest
    avg=ft_timelockanalysis(cfg,data);
    
    % the following needs data loaded from BioSemi_EEG_64_Headmodel_standard
    cfg=[];
    cfg.vol             = vol;
    cfg.grid            = grid;
    cfg.elec = elec_aligned;
    cfg.method          = 'lcmv';
    cfg.lcmv.fixedori   = 'yes';
    cfg.lcmv.feedback   = 'no';
    cfg.lcmv.normalize  = 'yes';
    cfg.lcmv.keepfilter = 'yes';
    cfg.lcmv.lambda     = '7%';
    source = ft_sourceanalysis(cfg, avg);
    cfg=[];
    avg=[];
    
    % this will return source activity and filters for source projection for all grid points and single trials
    % within the data set.
    Ingrid = find(grid.inside);
    ntrial = length(data.trial);
    
    % DECODING SIGNAL FOR THAT SUBJECT: 
    Yuse = Y{subj};
    
    %----------------------------------------------------------------
    % Correlation in source space
    %----------------------------------------------------------------
    STORE_CORR = zeros(length(Ingrid),length(I));
    clear X
    for G=1:length(Ingrid)                                          % loop all grid points in brain
        beamformer = source.avg.filter{Ingrid(G)};
        
        % compute single trial activity for each time point
        for t=1:ntrial
            X(t,:) = beamformer * data.trial{t}([1:64],trange2);    % X is (trial,time) activtiy for this grid point
            
        end
        
        % Compute the correlation of X and decoder for each time point
        for t = 1:length(trange2);
            STORE_CORR(G,t) = corr(X(:,t),Yuse(:,t));               % Should give one number
        end
        
    end
    
    filename = sprintf('%s/source_S%d.mat',savedir,subj);
    save(filename,'STORE_CORR','source');
    fprintf('S0%d done... \n',subj)
end

%--------------------------------------------------------------------------
% GROUP FILE
files = dir('source_*');
CORR_GROUP = cell(1,length(files));
for k = 1:length(files);
    load(sprintf('source_S%d.mat',k),'STORE_CORR');
    
    CORR_GROUP{k} = STORE_CORR;
    clear STORE_CORR
end

filename = sprintf('%s/source_GROUP.mat',savedir);
save(filename,'CORR_GROUP','source');
%--------------------------------------------------------------------------


%%
%--------------------------------------------------------------------------
% display  source correlations
clear
cd('W:\Project 1\log\EEG Data\EEG64Source');
load('C:\Program Files\MATLAB\R2012a\toolbox\fieldtrip-20150413\template\headmodel\standard_mri.mat')
load('W:\Project 1\log\EEG Data\EEG64Source\BioSemi_EEG_64_Headmodel_standard.mat')
load('W:\Project 1\log\EEG Data\EEG64Source\source_GROUP.mat');

% z-score correlations and compute average for each grid point and time
% pointof interest
XX = zeros(20,11432,49);
for S=1:length(CORR_GROUP)%:18
    XX(S,:,:) = corr2z(CORR_GROUP{S});
end

% SIGNIFICANT CLUSTER TIME POINTS FROM MAIN ANALYSIS 
ind{1} = 8:12;          % 84ms to 132ms
ind{2} = 14:20;         % 156ms to 228ms
ind{3} = 22:24;         % 252ms to 276ms

load('AVH_AVL_DIAG_RATE_NO11HZ_ALLTRIALS.mat','TAX');
timings = cell(1,length(ind));
for k = 1:length(ind);
    timings{k} = sprintf('%s to %s',num2str(TAX(ind{k}(1))),num2str(TAX(ind{k}(end))));
end

% LOOP THROUGH THE SIGNIFICANT CLUSTER TIME POINTS
Cluster_Source = cell(1,length(ind));
for k = 1:length(ind);
    
    t = ind{k};
    tmp = sq(median(mean(XX(:,:,t),1),3));          % avg across subjects
    source.avg.pow = zeros(size(source.avg.pow));
    source.avg.pow(grid.inside,:) = tmp;
    
    % interpolate
    maxl = max(tmp(:));
    cfg            = [];
    cfg.parameter  = 'pow';
    source2  = ft_sourceinterpolate(cfg, source , mri);
    
    % PLOT
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
    suptitle(timings{k})
    
    cfg = [];
    cfg.method        = 'slice';
    cfg.funparameter  = 'avg.pow';
    cfg.maskparameter = cfg.funparameter;
    cfg.funcolorlim   = [0 maxl];
    cfg.opacitylim    = [0 maxl];
    cfg.opacitymap    = 'rampup';
    ft_sourceplot(cfg, source2);
    suptitle(timings{k})
    
    title(timings{k})
    Cluster_Source{k} = source2;        % {1} = 84:132ms; {2} = 156:228ms; {3} = 252:276ms;
    
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% GET COORDINATES OF INTERESTING CLUSTERS 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% TIME POINTS 1 & 2 (84ms:132ms and 156:228ms) 

% EPOCH 1 : 84ms 
% CLUSTER 1, VISUAL   : 44,61,78   (spm: -47,-65,5);
% CLUSTER 1, AUDITORY : 40,80,69   (spm: -51,-46,-4);
coords1 = [44,61,78; 40 80 69];
source2 = Cluster_Source{1};
index = zeros(1,3); 
for k = 1:size(coords1,1)
    pos_source = (source2.transform)*[coords1(k,1) coords1(k,2) coords1(k,3) 1]';
    d = mean(abs(source.pos- repmat(pos_source(1:3)',[size(source.pos,1),1])).^2,2);
    [~,index(k)] = min(d);
end


% EPOCH 2, PARIETAL : 128,71,109 (spm: 37,-55,36)
coords2 = [128 71 109];
source2 = Cluster_Source{2};
pos_source = (source2.transform)*[coords2(1) coords2(2) coords2(3) 1]';
d = mean(abs(source.pos- repmat(pos_source(1:3)',[size(source.pos,1),1])).^2,2);
[~,index(length(coords1+1))] = min(d);

disp(index)
% note index



%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% GET EEG DATA FROM THE INTERESTING SOURCES FROM ABOVE, THEN CALCULATE
% NEURAL WEIGHTS AGAIN USING THESE SIGNALS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

subs = dir('W:\Project 1\log\EEG Data\Subj_*');
Nsub = length(subs);
addpath('C:\Program Files\MATLAB\R2012a\toolbox\fieldtrip-20150413\forward');
load('W:\Project 1\log\EEG Data\EEG64Source\BioSemi_EEG_64_Headmodel_standard.mat');

load('AVH_AVL_DIAG_RATE_NO11HZ_ALLTRIALS.mat','Y','trange','stimTimes','TAX');
trange = trange(stimTimes);                                 % timings
I = 3:51;
trange2 = trange(I);
[labels,LabelOrig,cfg_elecpos] = eegck_BiosemiLabels();     % use BioSemi positions

% Preprocessing settings
cfg_prepro = [];
cfg_prepro.lpfilter   = 'yes';          % apply lowpass filter
cfg_prepro.lpfreq     = 30;             % lowpass at 30 Hz.
cfg_prepro.hpfilter   = 'yes';          % apply highpass filter
cfg_prepro.hpfreq     = 1;
cfg_prepro.demean ='no';
cfg_prepro.reref = 'no';
cfg_prepro.refchannel = labels;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% loop participants
SourceSig = cell(1,20);

for S=1  :Nsub
    
    fprintf('Subj %d... \n',S)

    % load EEG and behavioural data from PMC (psychometric curve):
    cd(sprintf('W:/Project 1/log/EEG Data/%s',subs(S).name))
    load(sprintf('Prepro_all_S0%d_EOG_out.mat',S))
    
    % Preprocess EEG data again:
    data = ft_preprocessing(cfg_prepro,dataX);
    
    % compute covariance
    cfg=[];
    cfg.covariance='yes';
    cfg.covariancewindow = [-0.2 1]; % this should cover the trial period of interest
    avg=ft_timelockanalysis(cfg,data);
    
    % the following needs data loaded from BioSemi_EEG_64_Headmodel_standard
    cfg=[];
    cfg.vol             = vol;
    cfg.grid            = grid;
    cfg.elec = elec_aligned;
    cfg.method          = 'lcmv';
    cfg.lcmv.fixedori   = 'yes';
    cfg.lcmv.feedback   = 'no';
    cfg.lcmv.normalize  = 'yes';
    cfg.lcmv.keepfilter = 'yes';
    cfg.lcmv.lambda     = '7%';
    source = ft_sourceanalysis(cfg, avg);
    cfg=[];
    avg=[];
    
    % this will return source activity and filters for source projection for all grid points and single trials
    % within the data set.
    Ingrid = find(grid.inside);
    ntrial = length(data.trial);
    
    %----------------------------------------------------------------
    % ENTER INDEX OF INTEREST HERE
    %----------------------------------------------------------------
    LocalGrid = index;
    
    for G=1:length(LocalGrid) % loop all grid points in brain
        beamformer = source.avg.filter{LocalGrid(G)};
        
        % compute single trial activity for each time point
        X = [];
        for t=1:ntrial
            X(t,:) = beamformer * data.trial{t}([1:64],trange2);
        end
        SourceSig{S}.data{G} = X; % Source signal for each trial
    end
    SourceSig{S}.trialinfo = data.trialinfo;
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% REGRESSION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subs = dir('W:\Project 1\log\EEG Data\Subj_*');
Nsub = length(subs);
load('W:\Project 1\allTrials7.mat')     % stimuli matrix
Set{1} = [5,6];                         % AVH incongruent
Set{2} = [8,9];                         % AVL incongruent
NW_source = cell(1,3);

for c = 1:3;        % 1/2 = visual and temporal, epoch 1 (84:132ms); 3 = parietal 156:228ms
    
    for s = 1:Nsub
        
        % EEG trial info:
        condition = SourceSig{s}.trialinfo(:,3);                     % condition
        rate = SourceSig{s}.trialinfo(:,4);                          % rate
        expInd = SourceSig{s}.trialinfo(:,9);                        % stimuli index
        rateA =   SourceSig{s}.trialinfo(:,13);                      % auditory rate
        rateV =  SourceSig{s}.trialinfo(:,12);                       % visual rate
        choice = SourceSig{s}.trialinfo(:,14)-1;                     % 1 for first higher (i.e rate > 11), 2 second higher
        choice = 1-choice;                                           % 1 for rate > 11
        
        
        %----------------------------------------------------------------------
        % STIMULI : extract Auditory and Visual stimuli for these trials
        NewMatA = zeros(size(rateA,1),75);
        NewMatV = zeros(size(rateV,1),75);
        for o=1:length(choice)
            NewMatA(o,:) = allTrials(expInd((o)),:,rateA((o)))-1;
            NewMatV(o,:) = allTrials(expInd((o)),:,rateV((o)))-1;
        end
        
        
        % Work out cumulative stimuli rate for each time point:
        RatePast = zeros(75,size(NewMatA,1),2);
        for w=1:75
            Ra = mean(NewMatA(:,(1:w)),2);
            Rv = mean(NewMatV(:,(1:w)),2);
            RatePast(w,:,:)= [Ra,Rv];
        end
        
        % get AV incongruent trials
        cond = 1;
        jh = find( (condition>=Set{cond}(1)).*(condition<=Set{cond}(end)));
        cond = 2;
        jl = find( (condition>=Set{cond}(1)).*(condition<=Set{cond}(end)));
        Jtot = cat(1,jh,jl);
        
        
        for tw = 3 :51;
            
            %  regression model on both reliabilities at the same time
            xh = sq(RatePast(tw,jh,:));
            xl = sq(RatePast(tw,jl,:));
            
            %------------------------------------------------
            % EEG: regress Y on A, V rates in past time window
            acth = SourceSig{s}.data{c}(jh,tw-2);
            actl = SourceSig{s}.data{c}(jl,tw-2);
            
            
            % regress EEG Decoding activity on sensory input rate
            act = cat(1,acth,actl);
            x = zeros(size(act,1),5);
            x((1:length(jh)),(1:2)) = xh;
            x((1:length(jl))+length(jh),(1:2)+2) = xl;
            x(:,5) = 1;
            b = regress(act,x);
            NW_source{c}(s,tw-2,:) = b;                       % AH,VH, AL, VL,-
            
        end
        fprintf('S0%d \n',s)
    end
    
end

save('source_GROUP.mat','CORR_GROUP','Cluster_Source','NW_source','LocalGrid','STORE_CORR','SourceSig','clusters','index','times')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% PLOT SOURCE WEIGHTS 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath('W:\Project 1\log\EEG Data')
clf
load('AVH_AVL_DIAG_RATE_NO11HZ_ALLTRIALS.mat','TAX')
I = 3:51; % time period
TAX2 = TAX(I);

for c = 1:3;    
    
    % source neural weights 
    subplot(3,1,c);hold on
    plot(TAX2,sq(mean(NW_source{c}(:,:,1),1)));               % AH
    plot(TAX2,sq(mean(NW_source{c}(:,:,3),1)),':b');          % AL
    plot(TAX2,sq(mean(NW_source{c}(:,:,2),1)),'r');           % VH
    plot(TAX2,sq(mean(NW_source{c}(:,:,4),1)),':r');          % VL
    
    legend('AH','AL','VH','VL')
    hline(0,'-k')
    ylabel('Neural Weight')
    title(sprintf('NEURAL WEIGHTS: %s',times{c}))
    hline(50000,':k'); hline(-50000,':k')
    ylim([-100000 100000])
    set(gca,'yticklabel',num2str(get(gca,'ytick')'))
    
end

suptitle('regression weights using eeg data from source localisaton index')
legend('AH','AL','VH','VL')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% NEURAL SOURCE WEIGHTS: RELIABILITY EFFECTS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% AH-AL and VH-VL
% TVAL SHUFFLE CLUSTER DIFFERNCE - NEURAL
c = 1;
NeuralA = NW_source{c}(:,:,[1,3]);     % AH, AL
NeuralV = NW_source{c}(:,:,[2,4]);     % VH, VL

Ause = zeros(20,49,2,1000);
Vuse = zeros(20,49,2,1000);
oSave = zeros(20,4,49,100);
REL = cell(1,3); T = cell(1,3); 
clf
for c= 1:3;     % 3 sources (2 epoch 1, 1 epoch 2). 
    
    %---------------------------------------------------------------
    % Create the shuffled data matrics first (quicker)
    % tic
    for b = 1:1000;
        for t=1:49
            for s = 1:20;
                
                order = randi(2,1,4);   % random 1s and 2s
                
                % AUDITORY
                Ause(s,t,1,b) = NeuralA(s,t,order(1));
                Ause(s,t,2,b) = NeuralA(s,t,order(2));
                
                % VISUAL
                Vuse(s,t,1,b) = NeuralV(s,t,order(3));
                Vuse(s,t,2,b) = NeuralV(s,t,order(4));
                
                oSave(s,:,t,b) = order; % just to check later
            end
            
        end
    end

    %---------------------------------------------------------------
    % Do the ttests
    tic
    boot = 1000;
    TshufA = zeros(1,49,boot);
    TshufV = zeros(1,49,boot);
    TtrueA = zeros(1,49); TtrueV = zeros(1,49);
    for k = 1:boot;
        
        for t = 1:49;
            
            if k==1
                % true one
                [HA,PA,CIA,STATSA] = ttest(NW_source{c}(:,t,1),NW_source{c}(:,t,3)); % AUDITORY (HIGH-LOW)
                [HV,PV,CIV,STATSV] = ttest(NW_source{c}(:,t,2),NW_source{c}(:,t,4)); % VISUAL   (HIGH-LOW)
                TtrueA(1,t) = STATSA.tstat;
                TtrueV(1,t) = STATSV.tstat;
            end
            
            
            [~,~,~,STATSA] = ttest(Ause(:,t,1,k),Ause(:,t,2,k)); % AUDITORY (1 - 2)
            [~,~,~,STATSV] = ttest(Vuse(:,t,1,k),Vuse(:,t,2,k)); % VISUAL   (1 - 2)
            TshufA(1,t,k) = STATSA.tstat;
            TshufV(1,t,k) = STATSV.tstat;
            
        end
    end
    toc
    
    % clf
    cfg.critval = 1.8;                    % threshold for signif t-values
    cfg.clusterstatistic = 'maxsize';     % maxsize maxsum
    cfg.critvaltype = 'par';              % parametric threshold
    cfg.minsize = 2;
    cfg.pval = 0.05;                      % threshold to select signifciant clusters
    cfg.df = 19;
    
    
    [AUD_Source_POS,AUD_Source_NEG] = eegck_clusterstats(cfg,TtrueA,TshufA);
    [VIS_Source_POS,VIS_Source_NEG] = eegck_clusterstats(cfg,TtrueV,TshufV);
    
    REL{c}.TtrueA = TtrueA;
    REL{c}.TshufA = TshufA;
    REL{c}.TtrueV = TtrueV;
    REL{c}.TshufV = TshufV;
    REL{c}.AUD_POS = AUD_Source_POS;
    REL{c}.AUD_NEG = AUD_Source_NEG;
    REL{c}.VIS_POS = VIS_Source_POS;
    REL{c}.VIS_NEG = VIS_Source_NEG;
    REL{c}.NW = NW_source{c};
    

end

save('source_GROUP.mat','REL','CORR_GROUP','Cluster_Source','NW_source','LocalGrid','STORE_CORR','SourceSig','clusters','index','times','T','TAX2')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% PLOT THE RELIABILITY EFFECTS 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure
np = 1;
for c = 1:3;
    
    % AUDITORY 
    subplot(3,2,np); hold on
    NWuse1 = sq(mean(REL{c}.NW(:,:,1)));    % AH
    NWuse2 = sq(mean(REL{c}.NW(:,:,3)));    % AL 
    
    % AUDITORY HIGH 
    errorbar(TAX2,sq(mean(REL{c}.NW(:,:,1))),sq(sem(REL{c}.NW(:,:,1)))); hold on
    if isempty(REL{c}.AUD_POS)==0                       % plot the significant time points if there are any 
        ind =  find(REL{c}.AUD_POS.maskSig~=0);
        plot(TAX2(ind),NWuse1(ind),'*k');
    end
    
    % AUDITORY LOW 
    errorbar(TAX2,sq(mean(REL{c}.NW(:,:,3))),sq(sem(REL{c}.NW(:,:,3))),':g'); hold on
    ind = [];
    if isempty(REL{c}.AUD_NEG)==0
        ind =  find(REL{c}.AUD_NEG.maskSig~=0);
        plot(TAX2(ind),NWuse2(ind),'*k');
    end
    
    hline(50000,':k'); hline(-50000,':k')
    ylim([-100000 100000])
    title(sprintf('%s \n NW SOURCE WEIGHTS: AUD',num2str(times{c})))
    ylabel('NEURAL WEIGHT value');
    hline(0,':k');
    legend({'HIGH','LOW'},'FontSize',8,'Location','SouthEast')

    
    %------------------------------------------
    % VISUAL 
    subplot(3,2,np+1); hold on;
    NWuse3 = sq(mean(REL{c}.NW(:,:,2)));    % VH
    NWuse4 = sq(mean(REL{c}.NW(:,:,4)));    % VL 
    
    % VISUAL HIGH 
    errorbar(TAX2,sq(mean(REL{c}.NW(:,:,2))),sq(sem(REL{c}.NW(:,:,2))),'r'); hold on
    if isempty(REL{c}.VIS_POS)==0
        ind =  find(REL{c}.VIS_POS.maskSig~=0);
        plot(TAX2(ind),NWuse3(ind),'*k');
    end
    
    % VISUAL LOW 
    errorbar(TAX2,sq(mean(REL{c}.NW(:,:,4))),sq(sem(REL{c}.NW(:,:,4))),':m'); hold on
    ind = [];
    if isempty(REL{c}.VIS_NEG)==0
        ind =  find(REL{c}.VIS_NEG.maskSig~=0);
        plot(TAX2(ind),NWuse4(ind),'*k');
    end
    
    hline(50000,':k'); hline(-50000,':k')
    ylim([-100000 100000])
    title(sprintf('%s \n NW SOURCE WEIGHTS: VIS',num2str(times{c})))
    ylabel('NEURAL WEIGHT value');
    hline(0,':k');
    legend({'HIGH','LOW'},'FontSize',8,'Location','SouthEast')
    np = np+2;
    
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% PRINT OUT THE SIGNIFICANT TIME POINTS 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc
ind = [];
for c = 1:3;
    
    fprintf('Cluster %d...\n',c)
    
    if isempty(REL{c}.AUD_POS)==0
        ind{c}.POS =  find(REL{c}.AUD_POS.maskSig~=0);
        fprintf('AUD POS: %s \n',num2str(TAX2(ind{c}.POS)))
    end
    
    if isempty(REL{c}.AUD_NEG)==0
        ind{c}.NEG =  find(REL{c}.AUD_NEG.maskSig~=0);
        fprintf('AUD NEG: %s \n',num2str(TAX2(ind{c}.NEG)))
    end
    
    if isempty(REL{c}.VIS_POS)==0
        ind{c}.VPOS =  find(REL{c}.VIS_POS.maskSig~=0);
        
        fprintf('VIS POS: %s \n',num2str(TAX2(ind{c}.VPOS)))
    end
    
    if isempty(REL{c}.VIS_NEG)==0
        ind{c}.VNEG =  find(REL{c}.VIS_NEG.maskSig~=0);
        
        fprintf('VIS NEG: %s \n',num2str(TAX2(ind{c}.VNEG)))
    end
    
    fprintf('\n')
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% NEURAL: SOURCE WEIGHTS CORRELATIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load('AVH_AVL_DIAG_RATE_NO11HZ_ALLTRIALS.mat','ND','BD')

ND2 = ND(:,:,3:51);
BD2 = BD(:,:,3:51);
TAX2 = TAX(3:51);
T = cell(1,3);

for c = 1:3;
    
    %-------------------------------
    % neural weights from regression
    ND_source = [];
    ND_source(:,1,:) = NW_source{c}(:,:,1)-NW_source{c}(:,:,2);      % AH-VH
    ND_source(:,2,:) = NW_source{c}(:,:,3)-NW_source{c}(:,:,4);      % AL-VL
    
    
    %----------------------------------------------------------------------
    % STATS: CORRELATION BETWEEN SOURCE SIGNAL AND NEURAL WEIGHTS 
    n = size(ND_source,1);
    Tshuf =[]; Ttrue=[];
    clear r p tval
    
    for k=1:1000
        
        for t=1:size(ND2,3);
            
            if k==1
                [r(t),p(t),tval(t)]  = spearmanrank( ND_source(:,1,t)-ND_source(:,2,t),ND2(:,1,t)-ND2(:,2,t));
                Ttrue(1,t) = tval(t);
            end
            order = randperm(n);
            if sum(abs(order-[1:n]))==0    % avoid the tru order of subjects
                order = randperm(n);
            end
            [r2,p2,tval2] = spearmanrank(ND_source(:,1,t)-ND_source(:,2,t),ND2(order,1,t)-ND2(order,2,t));
            Tshuf(1,t,k) = tval2;          % has to be a 3D matrix
        end
    end
    
    T{c}.Ntrue = Ttrue;
    T{c}.Nshuf = Tshuf;
    T{c}.R_NDtrue = r;
    T{c}.P_NDtrue = p;

    cfg.critval = 1.5;                    % threshold for signif t-values
    cfg.clusterstatistic = 'maxsize';     % maxsize maxsum
    cfg.critvaltype = 'par';              % parametric threshold
    cfg.minsize = 2;
    cfg.pval = 0.05;                      % threshold to select signifciant clusters
    cfg.df = 19;
    [ND_NS_POS,ND_NS_NEG] = eegck_clusterstats(cfg,Ttrue,Tshuf);
    
    T{c}.ND_POS = ND_NS_POS;
    T{c}.ND_NEG = ND_NS_NEG;
    
    
    
    %----------------------------------------------------------------------
    % STATS: CORRELATION BETWEEN SOURCE SIGNAL AND BEHAVIORUAL WEIGHTS 
    n = size(ND_source,1);
    Tshuf =[]; Ttrue=[];
    I = 3:51;
    clear r p tval
    
    for k=1:1000
        
        for t=1:size(ND_source,3);
            
            if k==1
                [r(t),p(t),tval(t)]  = spearmanrank( ND_source(:,1,t)-ND_source(:,2,t),BD2(:,1,t)-BD2(:,2,t));
                Ttrue(1,t) = tval(t);
            end
            
            order = randperm(n);
            if sum(abs(order-[1:n]))==0  % avoid the tru order of subjects
                order = randperm(n);
            end
            [r2,p2,tval2] = spearmanrank(ND_source(:,1,t)-ND_source(:,2,t),BD2(order,1,t)-BD2(order,2,t));
            Tshuf(1,t,k) = tval2; % has to be a 3D matrix
        end
    end
    
    
    % clf
    cfg.critval = 1.5;                    % threshold for signif t-values
    cfg.clusterstatistic = 'maxsize';     % maxsize maxsum
    cfg.critvaltype = 'par';              % parametric threshold
    cfg.minsize = 2;
    cfg.pval = 0.05;                      % threshold to select signifciant clusters
    cfg.df = 19;
    % clf;
    [BD_NS_POS,BD_NS_NEG] = eegck_clusterstats(cfg,Ttrue,Tshuf)
    
    T{c}.Btrue = Ttrue;
    T{c}.Bshuf = Tshuf;
    T{c}.BD_POS = BD_NS_POS;
    T{c}.BD_NEG = BD_NS_NEG;
    T{c}.R_BDtrue = r;
    T{c}.P_BDtrue = p;
    
    
end

save('source_GROUP.mat','REL','CORR_GROUP','Cluster_Source','NW_source','LocalGrid','STORE_CORR','SourceSig','clusters','index','times','T','TAX2')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% CORRELATION VALUE PLOTS 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure
np = 1;
for c = 1:3;
    
    %------------------------------------------
    % SOURCE WEIGHTS: NEURAL WEIGHTS 
    subplot(3,2,np)
    plot(TAX2,T{c}.R_NDtrue)
    hold on
    if isempty(T{c}.ND_POS)==0
        ind =  find(T{c}.ND_POS.maskSig~=0);
        plot(TAX2(ind),T{c}.R_NDtrue(ind),'*r');
    end
    ind = [];
    if isempty(T{c}.ND_NEG)==0
        ind =  find(T{c}.ND_NEG.maskSig~=0);
        plot(TAX2(ind),T{c}.R_NDtrue(ind),'*k');
    end
    title(sprintf('%s \n NW SOURCE WEIGHTS: ND WEIGHTS \ncrit val:%s, method: %s',num2str(times{c}),num2str(cfg.critval),cfg.clusterstatistic))
    ylabel('r value');
    hline(0,':k');
    
    %------------------------------------------
    % SOURCE WEIGHTS: BEHAVIOURAL WEIGHTS 
    subplot(3,2,np+1)
    plot(TAX2,T{c}.R_BDtrue)
    hold on
    ind = [];
    if isempty(T{c}.BD_POS)==0
        ind =  find(T{c}.BD_POS.maskSig~=0);
        plot(TAX2(ind),T{c}.R_BDtrue(ind),'*r');
    end
    ind = [];
    if isempty(T{c}.BD_NEG)==0
        ind =  find(T{c}.BD_NEG.maskSig~=0);
        plot(TAX2(ind),T{c}.R_BDtrue(ind),'*k');
    end
    
    title(sprintf('%s \n NW SOURCE WEIGHTS: BD WEIGHTS \ncrit val:%s, method: %s',num2str(times{c}),num2str(cfg.critval),cfg.clusterstatistic))
    ylabel('r value');
    hline(0,':k'); 
    
    np = np+2;
    
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% PRINT OUT SIGNIFICANT TIME POINTS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc
ind = [];
for c = 1:3;
    
    fprintf('Cluster %d...\n',c)
    if isempty(T{c}.ND_POS)==0
        ind{c}.POS =  find(T{c}.ND_POS.maskSig~=0);
        fprintf('ND pos: %s \n',num2str(TAX2(ind{c}.POS)))
    end
    
    % ind = [];
    if isempty(T{c}.ND_NEG)==0
        ind{c}.NEG =  find(T{c}.ND_NEG.maskSig~=0);
        fprintf('ND neg: %s \n',num2str(TAX2(ind{c}.NEG)))
    end
    
    if isempty(T{c}.BD_POS)==0
        ind{c}.BPOS =  find(T{c}.BD_POS.maskSig~=0);
        fprintf('BD pos: %s \n',num2str(TAX2(ind{c}.BPOS)))
    end
    
    if isempty(T{c}.BD_NEG)==0
        ind{c}.BNEG =  find(T{c}.BD_NEG.maskSig~=0);
        fprintf('BD neg: %s \n',num2str(TAX2(ind{c}.BNEG)))
    end

    fprintf('\n')
end





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% THIRD EPOCH (ADDED LATER) : 252 TO 276MS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear
cd('W:\Project 1\log\EEG Data\EEG64Source');
load('C:\Program Files\MATLAB\R2012a\toolbox\fieldtrip-20150413\template\headmodel\standard_mri.mat')
load('W:\Project 1\log\EEG Data\EEG64Source\BioSemi_EEG_64_Headmodel_standard.mat')
load('W:\Project 1\log\EEG Data\EEG64Source\source_GROUP.mat');

% z-score correlations and compute average for each grid point and time
% pointof interest
XX = zeros(20,11432,49);
for S=1:length(CORR_GROUP)%:18
    XX(S,:,:) = corr2z(CORR_GROUP{S});
end

% SIGNIFICANT CLUSTER TIME POINTS
load('AVH_AVL_DIAG_RATE_NO11HZ_ALLTRIALS.mat','TAX');
ind{1} = 22:24;         % 252ms to 276ms
timings = sprintf('%s to %s',num2str(TAX(ind{1}(1))),num2str(TAX(ind{1}(end))));

% LOOP THROUGH THE SIGNIFICANT CLUSTER TIME POINTS
Cluster_Source3 = cell(1,length(ind));

for k = 1:length(ind);
    
    t = ind{k};
    tmp = sq(median(mean(XX(:,:,t),1),3));          % avg across subjects
    source.avg.pow = zeros(size(source.avg.pow));
    source.avg.pow(grid.inside,:) = tmp;
    
    % interpolate
    maxl = max(tmp(:));
    cfg            = [];
    cfg.parameter  = 'pow';
    source2  = ft_sourceinterpolate(cfg, source , mri);
    
    % PLOT
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
    suptitle(timings)
    
    cfg = [];
    cfg.method        = 'slice';
    cfg.funparameter  = 'avg.pow';
    cfg.maskparameter = cfg.funparameter;
    cfg.funcolorlim   = [0 maxl];
    cfg.opacitylim    = [0 maxl];
    cfg.opacitymap    = 'rampup';
    ft_sourceplot(cfg, source2);
    suptitle(timings)
    
    title(timings)
    Cluster_Source3{k} = source2;
    % Cluster_Source{1} = 252:276ms;
    
end

%%
% FIND THE NEURAL WEIGHTS/INTERESTING EPOCHS


% find the coordinates for the last epoch (252:276) areas
% visual, : 56 40 86, spm (-35, -86, 13);
% parietal/post central : 58,94,121 (-33 -32 48)

coords1 = [56,40,86; 58 94 121];
source2 = Cluster_Source{1};
for k = 1:size(coords1,1)
    pos_source = (source2.transform)*[coords1(k,1) coords1(k,2) coords1(k,3) 1]';
    d = mean(abs(source.pos- repmat(pos_source(1:3)',[size(source.pos,1),1])).^2,2);
    [~,index2(k)] = min(d);
end

addpath('C:\Program Files\MATLAB\R2012a\toolbox\fieldtrip-20150413\forward');
load('W:\Project 1\log\EEG Data\EEG64Source\BioSemi_EEG_64_Headmodel_standard.mat');

load('AVH_AVL_DIAG_RATE_NO11HZ_ALLTRIALS.mat','Y','trange','stimTimes','TAX');
trange = trange(stimTimes); % timings
I = 3:51;
trange2 = trange(I);

% use BioSemi positions
[labels,LabelOrig,cfg_elecpos] = eegck_BiosemiLabels();

% Preprocessing settings
cfg_prepro = [];
cfg_prepro.lpfilter   = 'yes';          % apply lowpass filter
cfg_prepro.lpfreq     = 30;             % lowpass at 30 Hz.
cfg_prepro.hpfilter   = 'yes';          % apply highpass filter
cfg_prepro.hpfreq     = 1;
cfg_prepro.demean ='no';
cfg_prepro.reref = 'no';
cfg_prepro.refchannel = labels;

subs = dir('W:\Project 1\log\EEG Data\Subj_*');
Nsub = length(subs);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% loop participants
tic
SourceSig3 = cell(1,20);

for S=1  :20
    
    fprintf('Subj %d... \n',S)
    %     load(sprintf('W:/Project 1/log/EEG Data/EEG64Source/source_S%d.mat',S))
    %
    
    % load EEG and behavioural data from PMC (psychometric curve):
    cd(sprintf('W:/Project 1/log/EEG Data/%s',subs(S).name))
    load(sprintf('Prepro_all_S0%d_EOG_out.mat',S))
    
    % Preprocess EEG data again:
    data = ft_preprocessing(cfg_prepro,dataX);
    
    % compute covariance
    cfg=[];
    cfg.covariance='yes';
    cfg.covariancewindow = [-0.2 1]; % this should cover the trial period of interest
    avg=ft_timelockanalysis(cfg,data);
    
    % the following needs data loaded from BioSemi_EEG_64_Headmodel_standard
    cfg=[];
    cfg.vol             = vol;
    cfg.grid            = grid;
    cfg.elec = elec_aligned;
    cfg.method          = 'lcmv';
    cfg.lcmv.fixedori   = 'yes';
    cfg.lcmv.feedback   = 'no';
    cfg.lcmv.normalize  = 'yes';
    cfg.lcmv.keepfilter = 'yes';
    cfg.lcmv.lambda     = '7%';
    source = ft_sourceanalysis(cfg, avg);
    cfg=[];
    avg=[];
    
    % this will return source activity and filters for source projection for all grid points and single trials
    % within the data set.
    Ingrid = find(grid.inside);
    ntrial = length(data.trial);
    
    %----------------------------------------------------------------
    % Extract source data for voxels of interest
    %----------------------------------------------------------------
    
    % ENTER INDEX OF INTEREST HERE
    LocalGrid = [index2];
    %   tax = data.time{1};
    
    
    
    for G=1:length(LocalGrid) % loop all grid points in brain
        beamformer = source.avg.filter{LocalGrid(G)};
        % compute single trial activity for each time point
        % USE trange to reduce teh number of time points computed - incl. onto
        % periods of interest to speed up
        X = [];
        for t=1:ntrial
            X(t,:) = beamformer * data.trial{t}([1:64],trange2);
        end
        SourceSig3{S}.data{G} = X; % Source signal for each trial
    end
    SourceSig3{S}.trialinfo = data.trialinfo;
end

% NEURAL WEIGHTS FOR THE THIRD CLUSTER
subs = dir('W:\Project 1\log\EEG Data\Subj_*');
Nsub = length(subs);
load('W:\Project 1\allTrials7.mat')     % stimuli matrix
Set{1} = [5,6];                         % AVH incongruent
Set{2} = [8,9];                         % AVL incongruent
NW_source3 = cell(1,2);

for c = 1: length(SourceSig3{1}.data);
    
    for s = 1:Nsub
        
        % EEG trial info:
        condition = SourceSig3{s}.trialinfo(:,3);                     % condition
        rate = SourceSig3{s}.trialinfo(:,4);                          % rate
        expInd = SourceSig3{s}.trialinfo(:,9);                        % stimuli index
        rateA =   SourceSig3{s}.trialinfo(:,13);                      % auditory rate
        rateV =  SourceSig3{s}.trialinfo(:,12);                       % visual rate
        choice = SourceSig3{s}.trialinfo(:,14)-1;                     % 1 for first higher (i.e rate > 11), 2 second higher
        choice = 1-choice;                                           % 1 for rate > 11
        
        
        %----------------------------------------------------------------------
        % STIMULI : extract Auditory and Visual stimuli for these trials
        NewMatA = zeros(size(rateA,1),75);
        NewMatV = zeros(size(rateV,1),75);
        for o=1:length(choice)
            NewMatA(o,:) = allTrials(expInd((o)),:,rateA((o)))-1;
            NewMatV(o,:) = allTrials(expInd((o)),:,rateV((o)))-1;
        end
        
        
        % Work out cumulative stimuli rate for each time point:
        RatePast = zeros(75,size(NewMatA,1),2);
        for w=1:75
            Ra = mean(NewMatA(:,(1:w)),2);
            Rv = mean(NewMatV(:,(1:w)),2);
            RatePast(w,:,:)= [Ra,Rv];
        end
        
        % get AV incongruent trials
        cond = 1;
        jh = find( (condition>=Set{cond}(1)).*(condition<=Set{cond}(end)));
        cond = 2;
        jl = find( (condition>=Set{cond}(1)).*(condition<=Set{cond}(end)));
        Jtot = cat(1,jh,jl);
        
        
        for tw = 3 :51;
            
            
            %  regression model on both reliabilities at the same time
            xh = sq(RatePast(tw,jh,:));
            xl = sq(RatePast(tw,jl,:));
            
            %------------------------------------------------
            % EEG: regress Y on A, V rates in past time window
            acth = SourceSig3{s}.data{c}(jh,tw-2);
            actl = SourceSig3{s}.data{c}(jl,tw-2);
            
            
            % regress EEG Decoding activity on sensory input rate
            act = cat(1,acth,actl);
            x = zeros(size(act,1),5);
            x((1:length(jh)),(1:2)) = xh;
            x((1:length(jl))+length(jh),(1:2)+2) = xl;
            x(:,5) = 1;
            b = regress(act,x);
            NW_source3{c}(s,tw-2,:) = b;                       % AH,VH, AL, VL,-
            
        end
        fprintf('S0%d \n',s)
    end
    
end

save('THIRD_CLUSTER_SOURCE.mat','index2','SourceSig3','NW_source3')


%% PLOT
addpath('W:\Project 1\log\EEG Data')
clf
% figure


I = 3:51; % time period
TAX2 = TAX(I);

np = 1;
for c = 1:length(NW_source3)
    
    subplot(2,1,c);hold on
    plot(TAX2,sq(mean(NW_source3{c}(:,:,1),1)));               % AH
    plot(TAX2,sq(mean(NW_source3{c}(:,:,3),1)),':b');          % AL
    plot(TAX2,sq(mean(NW_source3{c}(:,:,2),1)),'r');          % VH
    plot(TAX2,sq(mean(NW_source3{c}(:,:,4),1)),':r');          % VL
    
    legend('AH','AL','VH','VL')
    hline(0,'-k')
    ylabel('Neural Weight')
    title(sprintf('NEURAL WEIGHTS: %s',times{c}))
    
    hline(50000,':k'); hline(-50000,':k')
    ylim([-100000 100000])
    set(gca,'yticklabel',num2str(get(gca,'ytick')'))
    
end

suptitle('regression weights using eeg data from source localisaton index')
legend('AH','AL','VH','VL')


%% AH-AL and VH-VL
% TVAL SHUFFLE CLUSTER DIFFERNCE - NEURAL SOURCE WEIGHTS
NeuralA = NW_source3{c}(:,:,[1,3]);     % AH, AL
NeuralV = NW_source3{c}(:,:,[2,4]);     % VH, VL

Ause = zeros(20,49,2,1000);
Vuse = zeros(20,49,2,1000);
oSave = zeros(20,4,49,100);
REL3 = cell(1,2);


for c= 1:3;
    %---------------------------------------------------------------
    % Create the shuffled data matrics first (quicker)
    % tic
    for b = 1:1000;
        
        for t=1:49
            for s = 1:20;
                
                order = randi(2,1,4);   % random 1s and 2s
                
                % AUDITORY
                Ause(s,t,1,b) = NeuralA(s,t,order(1));
                Ause(s,t,2,b) = NeuralA(s,t,order(2));
                
                % VISUAL
                Vuse(s,t,1,b) = NeuralV(s,t,order(3));
                Vuse(s,t,2,b) = NeuralV(s,t,order(4));
                
                oSave(s,:,t,b) = order; % just to check later
            end
            
        end
    end
    % toc
    
    %---------------------------------------------------------------
    % Do the ttests
    tic
    boot = 1000;
    TshufA = zeros(1,49,boot);
    TshufV = zeros(1,49,boot);
    TtrueA = zeros(1,49); TtrueV = zeros(1,49);
    for k = 1:boot;
        
        for t = 1:49;
            
            if k==1
                % true one
                [HA,PA,CIA,STATSA] = ttest(NW_source3{c}(:,t,1),NW_source3{c}(:,t,3)); % AUDITORY (HIGH-LOW)
                [HV,PV,CIV,STATSV] = ttest(NW_source3{c}(:,t,2),NW_source3{c}(:,t,4)); % VISUAL   (HIGH-LOW)
                TtrueA(1,t) = STATSA.tstat;
                TtrueV(1,t) = STATSV.tstat;
            end
            
            
            [~,~,~,STATSA] = ttest(Ause(:,t,1,k),Ause(:,t,2,k)); % AUDITORY (1 - 2)
            [~,~,~,STATSV] = ttest(Vuse(:,t,1,k),Vuse(:,t,2,k)); % VISUAL   (1 - 2)
            TshufA(1,t,k) = STATSA.tstat;
            TshufV(1,t,k) = STATSV.tstat;
            
        end
    end
    toc
    
    % clf
    cfg.critval = 1.8;                    % threshold for signif t-values
    cfg.clusterstatistic = 'maxsize';     % maxsize maxsum
    cfg.critvaltype = 'par';              % parametric threshold
    cfg.minsize = 2;
    cfg.pval = 0.05;                      % threshold to select signifciant clusters
    cfg.df = 19;
    
    
    [AUD_Source_POS,AUD_Source_NEG] = eegck_clusterstats(cfg,TtrueA,TshufA);
    [VIS_Source_POS,VIS_Source_NEG] = eegck_clusterstats(cfg,TtrueV,TshufV);
    
    REL3{c}.TtrueA = TtrueA;
    REL3{c}.TshufA = TshufA;
    REL3{c}.TtrueV = TtrueV;
    REL3{c}.TshufV = TshufV;
    REL3{c}.AUD_POS = AUD_Source_POS;
    REL3{c}.AUD_NEG = AUD_Source_NEG;
    REL3{c}.VIS_POS = VIS_Source_POS;
    REL3{c}.VIS_NEG = VIS_Source_NEG;
    REL3{c}.NW = NW_source{c};
    
    
end

save('THIRD_CLUSTER_SOURCE.mat','index2','SourceSig3','NW_source3','REL3')

%% PLOT THE RESULTS: RELIABILITY DIFFERENCES
figure
np = 1;
times3 = {'VISUAL COORD, 252:276ms','PARIETAL COORD, 252:276'};

for c = 1:length(NW_source3);
    
    %------------------------------------------
    % AUDITORY
    subplot(2,2,np); hold on
    NWuse1 = sq(mean(REL3{c}.NW(:,:,1)));
    NWuse2 = sq(mean(REL3{c}.NW(:,:,3)));
    
    % AUDITORY HIGH
    errorbar(TAX2,sq(mean(REL3{c}.NW(:,:,1))),sq(sem(REL3{c}.NW(:,:,1)))); hold on
    if isempty(REL3{c}.AUD_POS)==0
        ind =  find(REL3{c}.AUD_POS.maskSig~=0);
        plot(TAX2(ind),NWuse1(ind),'*k');
    end
    
    % AUDITORY LOW
    errorbar(TAX2,sq(mean(REL3{c}.NW(:,:,3))),sq(sem(REL3{c}.NW(:,:,3))),':g'); hold on
    ind = [];
    if isempty(REL3{c}.AUD_NEG)==0
        ind =  find(REL3{c}.AUD_NEG.maskSig~=0);
        plot(TAX2(ind),NWuse2(ind),'*k');
    end
    hline(50000,':k'); hline(-50000,':k')
    ylim([-100000 100000])
    title(sprintf('%s \n NW SOURCE WEIGHTS: AUD',num2str(times3{c})))
    ylabel('NEURAL WEIGHT value');
    hline(0,':k');
    legend({'HIGH','LOW'},'FontSize',8,'Location','SouthEast')
    
    
    
    %------------------------------------------
    % VISUAL
    subplot(2,2,np+1); hold on;
    NWuse3 = sq(mean(REL3{c}.NW(:,:,2)));
    NWuse4 = sq(mean(REL3{c}.NW(:,:,4)));
    
    % VISUAL HIGH
    errorbar(TAX2,sq(mean(REL3{c}.NW(:,:,2))),sq(sem(REL3{c}.NW(:,:,2))),'r'); hold on
    if isempty(REL3{c}.VIS_POS)==0
        ind =  find(REL3{c}.VIS_POS.maskSig~=0);
        plot(TAX2(ind),NWuse3(ind),'*k');
    end
    
    % VISUAL LOW
    errorbar(TAX2,sq(mean(REL3{c}.NW(:,:,4))),sq(sem(REL3{c}.NW(:,:,4))),':m'); hold on
    ind = [];
    if isempty(REL3{c}.VIS_NEG)==0
        ind =  find(REL3{c}.VIS_NEG.maskSig~=0);
        plot(TAX2(ind),NWuse4(ind),'*k');
    end
    hline(50000,':k'); hline(-50000,':k')
    ylim([-100000 100000])
    title(sprintf('%s \n NW SOURCE WEIGHTS: VIS',num2str(times3{c})))
    ylabel('NEURAL WEIGHT value');
    hline(0,':k');
    legend({'HIGH','LOW'},'FontSize',8,'Location','SouthEast')
    np = np+2;
    
end



%% NEURAL SOURCE WEIGHTS CORRELATIONS

load('AVH_AVL_DIAG_RATE_NO11HZ_ALLTRIALS.mat','ND','BD','TAX')

ND2 = ND(:,:,3:51);
BD2 = BD(:,:,3:51);
TAX2 = TAX(3:51);
T3 = cell(1,2);

for c = 1:2;
    
    %-------------------------------
    % neural weights from regression
    ND_source3 = [];
    ND_source3(:,1,:) = NW_source3{c}(:,:,1)-NW_source3{c}(:,:,2);      % AH-VH
    ND_source3(:,2,:) = NW_source3{c}(:,:,3)-NW_source3{c}(:,:,4);      % AL-VL
    
    
    %--------------------------------------------------------------------------
    % stats - correlation between source signal and ND weights
    n = size(ND_source3,1);
    Tshuf =[]; Ttrue=[];
    clear r p tval
    
    for k=1:1000
        
        for t=1:size(ND2,3);
            
            if k==1
                [r(t),p(t),tval(t)]  = spearmanrank( ND_source3(:,1,t)-ND_source3(:,2,t),ND2(:,1,t)-ND2(:,2,t));
                Ttrue(1,t) = tval(t);
            end
            
            order = randperm(n);
            if sum(abs(order-[1:n]))==0  % avoid the tru order of subjects
                order = randperm(n);
            end
            [r2,p2,tval2] = spearmanrank(ND_source3(:,1,t)-ND_source3(:,2,t),ND2(order,1,t)-ND2(order,2,t));
            Tshuf(1,t,k) = tval2; % has to be a 3D matrix
        end
    end
    
    T3{c}.Ntrue = Ttrue;
    T3{c}.Nshuf = Tshuf;
    T3{c}.R_NDtrue = r;
    T3{c}.P_NDtrue = p;
    
    % clf
    cfg.critval = 1.5;                    % threshold for signif t-values
    cfg.clusterstatistic = 'maxsize';     % maxsize maxsum
    cfg.critvaltype = 'par';              % parametric threshold
    cfg.minsize = 2;
    cfg.pval = 0.05;                      % threshold to select signifciant clusters
    cfg.df = 19;
    [ND_NS_POS,ND_NS_NEG] = eegck_clusterstats(cfg,Ttrue,Tshuf);
    
    T3{c}.ND_POS = ND_NS_POS;
    T3{c}.ND_NEG = ND_NS_NEG;
    
    
    
    %--------------------------------------------------------------------------
    % stats - correlation between source signal and BD weights
    n = size(ND_source3,1);
    Tshuf =[]; Ttrue=[];
    I = 3:51;
    clear r p tval
    
    for k=1:1000
        
        for t=1:size(ND_source3,3);
            
            if k==1
                [r(t),p(t),tval(t)]  = spearmanrank( ND_source3(:,1,t)-ND_source3(:,2,t),BD2(:,1,t)-BD2(:,2,t));
                Ttrue(1,t) = tval(t);
            end
            
            order = randperm(n);
            if sum(abs(order-[1:n]))==0  % avoid the tru order of subjects
                order = randperm(n);
            end
            [r2,p2,tval2] = spearmanrank(ND_source3(:,1,t)-ND_source3(:,2,t),BD2(order,1,t)-BD2(order,2,t));
            Tshuf(1,t,k) = tval2; % has to be a 3D matrix
        end
    end
    
    
    % clf
    cfg.critval = 1.5;                    % threshold for signif t-values
    cfg.clusterstatistic = 'maxsize';     % maxsize maxsum
    cfg.critvaltype = 'par';              % parametric threshold
    cfg.minsize = 2;
    cfg.pval = 0.05;                      % threshold to select signifciant clusters
    cfg.df = 19;
    
    [BD_NS_POS,BD_NS_NEG] = eegck_clusterstats(cfg,Ttrue,Tshuf);
    
    T3{c}.Btrue = Ttrue;
    T3{c}.Bshuf = Tshuf;
    T3{c}.BD_POS = BD_NS_POS;
    T3{c}.BD_NEG = BD_NS_NEG;
    T3{c}.R_BDtrue = r;
    T3{c}.P_BDtrue = p;
    
    
end

save('THIRD_CLUSTER_SOURCE.mat','index2','SourceSig3','NW_source3','REL3','T3','TAX')

%%

figure
np = 1;

for c = 1:2;
    
    %------------------------------------------
    % neural weights:source
    
    subplot(2,2,np)
    plot(TAX2,T3{c}.R_NDtrue)
    hold on
    
    if isempty(T3{c}.ND_POS)==0
        ind =  find(T3{c}.ND_POS.maskSig~=0);
        plot(TAX2(ind),T3{c}.R_NDtrue(ind),'*r');
    end
    
    ind = [];
    if isempty(T3{c}.ND_NEG)==0
        ind =  find(T3{c}.ND_NEG.maskSig~=0);
        plot(TAX2(ind),T3{c}.R_NDtrue(ind),'*k');
    end
    
    title(sprintf('%s \n NW SOURCE WEIGHTS: ND WEIGHTS \ncrit val:%s, method: %s',num2str(times3{c}),num2str(cfg.critval),cfg.clusterstatistic))
    ylabel('r value');
    hline(0,':k');
    % ylim([-0.8 0.8])
    
    
    
    %------------------------------------------
    % behavioural:source
    subplot(2,2,np+1)
    plot(TAX2,T3{c}.R_BDtrue)
    hold on
    
    ind = [];
    if isempty(T3{c}.BD_POS)==0
        ind =  find(T3{c}.BD_POS.maskSig~=0);
        plot(TAX2(ind),T3{c}.R_BDtrue(ind),'*r');
    end
    
    ind = [];
    if isempty(T3{c}.BD_NEG)==0
        ind =  find(T3{c}.BD_NEG.maskSig~=0);
        plot(TAX2(ind),T3{c}.R_BDtrue(ind),'*k');
    end
    
    title(sprintf('%s \n NW SOURCE WEIGHTS: BD WEIGHTS \ncrit val:%s, method: %s',num2str(times3{c}),num2str(cfg.critval),cfg.clusterstatistic))
    ylabel('r value');
    hline(0,':k');
    % ylim([-0.8 0.8])
    
    np = np+2;
    
end


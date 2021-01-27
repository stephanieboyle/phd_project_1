%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Decoding script - AV Reliability Project
% Steph PhD Project 1
%
% Compute AVH_AVL decoder for the whole brain, at each time point. Don't
% project over time, only trials. (Diagonal decoder).
% Decode High vs Low stimulation rate (1) or participant choice (2) by
% changing line 31 (ARG.CODE).
%
% After decoding signal calculated for each participant individually,
% calculate neural weights and behavioural weights by regressing the
% decoding signal and behavioural choice signal against accumulated rate
% (from the stimulus).
% Finally, correlations between Neural Weights, Behavioural Regression
% Weights and perceptual weights derived from fitting psychometric curves
% to the data (optimal integration model) are calculated.
%--------------------------------------------------------------------------

clear ; clc;
addpath('\\analyse2\Project0109\Lab\ckmatlab\eegck')
addpath('Z:\Lab\ckmatlab\ckinfo')

% set conditions
k = [4 7];                              % AVH 0, AVL 0 conditions
Set{1} = [5,6];                         % AVH incongruent
Set{2} = [8,9];                         % AVL incongruent

ARG.CODE = 1;                           % 1 for Stim , 2 for Choice
if ARG.CODE ==1;
    COND = 'RATE';
elseif ARG.CODE ==2;
    COND = 'CHOICE';
end

% Decoder settings
[labels,~] = eegck_BiosemiLabels();     % eeg labels
ARG.AmpThr = 100;
ARG.Twin = [-0.2 1.3];                  % time points for decoder
ARG.Tsteps = 5;                         % every nth time point
ARG.winL = 12;                          % decoder length
cfg_decode =[];
cfg_decode.CVsteps = 10;
cfg_de.channels = (1:64);
cfg_decode.reg.gamma = 0.1;
cfg_logit=[];
cfg_logit.regularize = 1;               % 0
cfg_logit.lambda = 0.001;
cfg_logit.lambdasearch =0;              % 0
cfg_logit.evratio = 0.1;                % [0 1]%
cfg_logit.CVsteps=10;

% Preprocessing settings
cfg_prepro = [];
cfg_prepro.lpfilter   = 'yes';          % apply lowpass filter
cfg_prepro.lpfreq     = 30;             % lowpass at 30 Hz.
cfg_prepro.hpfilter   = 'yes';          % apply highpass filter
cfg_prepro.hpfreq     = 1;
cfg_prepro.demean ='no';
cfg_prepro.reref = 'no';
cfg_prepro.refchannel = labels;

load('W:\Project 1\allTrials7.mat')     % stimuli matrix
timingsEVENTS = 0:0.0120:0.89;          % stimuli timing

%--------------------------------------------------------------------------
% DECODING/REGRESSION LOOP
%--------------------------------------------------------------------------
subs = dir('W:\Project 1\log\EEG Data\Subj_*');
timeP = length(timingsEVENTS);              % # of time points
subS = length(subs);                        % # of subjects
elec = size(labels,2);                      % # of electrodes

% initialise stuff
Wsave = zeros(subS,elec,timeP);             % Weights from AVH_AVL decoder
Csave = zeros(subS,timeP);                  % Cs from AVH_AVL decoder
AzGroup = zeros(subS,timeP);                % Az AVH_AVL decoder
AzChoice = zeros(subS,6,timeP);             % Az Choice
AzRate = zeros(subS,timeP);                 % Az Projected decoder
Topo = zeros(subS,elec,timeP);              % AVH_AVL topoplots
Topo1 = zeros(subS,elec,timeP);             % all trials topoplots
BWeights = zeros(subS,timeP,6);             % behavioural regression weights
NeuralWeights = zeros(subS,timeP,5);        % neural regression weights
actS = cell(1,subS);                        % decoding activity used (to check diag & projected are the same)
Y_AV = cell(1,subS);                        % AVH_AVL Y
Y = cell(1,subS);                           % All trials Y
T = cell(1,subS);                           % trialinfo


for subj = 1 : length(subs);
    
    % load EEG and behavioural data from PMC (psychometric curve):
    cd(sprintf('W:/Project 1/log/EEG Data/%s',subs(subj).name))
    load(sprintf('Prepro_all_S0%d_EOG_out.mat',subj))
    WBeh = load(sprintf('all_S0%d',subj),'weightsObs');
    
    % Preprocess EEG data again:
    dataTarget = ft_preprocessing(cfg_prepro,dataX);
    MT = eegck_trials2mat(dataTarget);
    trialinfo = dataTarget.trialinfo;
    time = dataTarget.time{1};
    clear dataTarget dataX Yind
    
    % EEG trial info:
    condition = trialinfo(:,3);                     % condition
    rate = trialinfo(:,4);                          % rate
    expInd = trialinfo(:,9);                        % stimuli index
    rateA =   trialinfo(:,13);                      % auditory rate
    rateV =  trialinfo(:,12);                       % visual rate
    choice = trialinfo(:,14)-1;                     % 1 for first higher (i.e rate > 11), 2 second higher
    choice = 1-choice;                              % 1 for rate > 11
    
    
    %----------------------------------------------------------------------
    % STIMULI : extract Auditory and Visual stimuli for these trials
    NewMatA = zeros(size(rateA,1),75);
    NewMatV = zeros(size(rateV,1),75);
    for o=1:length(choice)
        NewMatA(o,:) = allTrials(expInd((o)),:,rateA((o)))-1;
        NewMatV(o,:) = allTrials(expInd((o)),:,rateV((o)))-1;
    end
    % all trials contains the actual stimuli in 1s/2s... 1 is blank, 2 is
    % stimuli presented.
    
    % Work out cumulative stimuli rate for each time point:
    RatePast = zeros(75,size(NewMatA,1),2);
    for w=1:75
        Ra = mean(NewMatA(:,(1:w)),2);
        Rv = mean(NewMatV(:,(1:w)),2);
        RatePast(w,:,:)= [Ra,Rv];
    end
    TimeForRate = timingsEVENTS;
    
    
    % decoder timings
    timings = time;
    dt = 0.012;
    Decode_times2 = (-dt*20:dt:dt*100);                         % match up decoding with stimuli time line
    trange = zeros(1,length(Decode_times2));
    for t = 1:length(Decode_times2);
        trange2 = find(timings<=Decode_times2(t));
        trange(t) = trange2(end);
    end
    stimTimes = find(Decode_times2>=0 & Decode_times2<=0.89);   % stim durations
    
    
    %--------------------------------------------
    % AVH AVL Decoder Trials
    a = find(condition==k(1) & rate~=4 | condition== k(2) & rate~=4);
    if ARG.CODE == 1;
        YCon = zeros(1,length(a));
        YCon(rate(a)<4)=0;                                       % YCon = vector of 0/1 where low rate trials = 0, high rate trials = 1;
        YCon(rate(a)>4)=1;
    elseif  ARG.CODE == 2;
        YCon = choice(a)' ;
    end
    
    size(a);
    
    %----------------------------------------------------------------------
    % EEG: DO THE DECODER :
    Az = zeros(1,length(trange));
    YoverTime = zeros(size(MT,2),timeP);
    Ys = zeros(timeP,size(YCon,2));
    
    Y_AV = cell(1,length(subs)); 
%     Y_ALL = cell(1,length(subs)); 
%     Y_FLIP = cell(1,length(subs)); 
    
    for tw = 3 : length(stimTimes);
        
        % compute weights at tw using activity from averaged time window
        act1 = mean(MT(1:64,a,trange(stimTimes(tw))+(1:ARG.winL)),3);
        [Aeach,Az,Ys(tw,:),Wmix]= eegck_LDA(cfg_decode,act1',YCon');
        Cmix = Wmix(end);
        Wmix = Wmix(1:end-1);
        

        
        % save the stuff
        Wsave(subj,:,tw) = Wmix;
        Csave(subj,tw) = Cmix;
        Topo(subj,:,tw) = Aeach;
        AzGroup(subj,tw) = Az;
        
        % Project Y across All trials this time using the AVH_AVL weight
        act2 = mean(MT(1:64,:,trange(stimTimes(tw))+(1:ARG.winL)),3);
        YoverTime(:,tw)= (Wmix*act2)-Cmix';
        
%         Y_ALL{subj} = YoverTime;
        
        % compute Az for each time point ALL TRIALS
        Jtot = 1:size(MT,2);
        YCon2 = zeros(1,size(MT,2));
        YCon2(rate(Jtot)<4)=0;                                                  % YCon = vector of 0/1 where low rate trials = 0, high rate trials = 1;
        YCon2(rate(Jtot)>4)=1;
        [~,~,tmp,~] = ck_decoding_roc(YCon2,YoverTime(Jtot,tw));
        AzRate(subj,tw) = tmp;
        
        % compute Az vs. choice for each time point. To not confound rate
        % information we compute this for each rate separately
        for r=2:6
            jj = find(rate(Jtot)==r);
            [~,~,tmp,~] = ck_decoding_roc(choice(Jtot(jj)), YoverTime(Jtot(jj),tw));
            AzChoice(subj,r,tw) = tmp;
        end
        
        % compute Topo for all trials
        A = (act2*YoverTime(Jtot,tw))/(YoverTime(Jtot,tw)'*YoverTime(Jtot,tw));
        Topo1(subj,:,tw) = A;
        
        
        %----------------------------------------------------------------------
        % Regression Loop: regress Behavioural choice and EEG against
        % cumulative stimulation rate
        
        % get AV incongruent trials
        cond = 1;
        jh = find( (condition>=Set{cond}(1)).*(condition<=Set{cond}(end)));
        cond = 2;
        jl = find( (condition>=Set{cond}(1)).*(condition<=Set{cond}(end)));
        Jtot = cat(1,jh,jl);
        
        %  regression model on both reliabilities at the same time
        xh = sq(RatePast(tw,jh,:));
        xl = sq(RatePast(tw,jl,:));
        
        
        
        %         %------------------------------------------------
        %         % regression of behavioural choice:
        %         x = zeros(size(Jtot,1),4);
        %         x((1:length(jh)),(1:2)) = xh;
        %         x((1:length(jl))+length(jh),(1:2)+2) = xl;
        %         chc = cat(1,choice(jh),choice(jl));
        %         [~,Az,yhat,Wout,~] = eegck_Logit(cfg_logit,x,chc);
        %         BWeights(subj,tw,:) = [Wout' Az];                        % AH,VH, AL, VL,-
        
        
        %------------------------------------------------
        % EEG: regress Y on A, V rates in past time window
        % flip Y signal
        YoverTime2 = YoverTime;
        YoverTime2(YCon2==0,:) = YoverTime2(YCon2==0,:)*-1;
        
        
        acth = YoverTime2(jh,tw);
        actl = YoverTime2(jl,tw);
        %         acth = YoverTime(jh,tw);
        %         actl = YoverTime(jl,tw);
        
        % regress EEG Decoding activity on sensory input rate
        act = cat(1,acth,actl);
        x = zeros(size(act,1),5);
        x((1:length(jh)),(1:2)) = xh;
        x((1:length(jl))+length(jh),(1:2)+2) = xl;
        x(:,5) = 1;
        b = regress(act,x);
        NeuralWeights(subj,tw,:) = b;                       % AH,VH, AL, VL,-
        TAX = TimeForRate;
        
        actS{subj}.high(:,tw) = acth;
        actS{subj}.low(:,tw) = actl;
        
    end
    
    fprintf('S0%d Done \n',subj)
    
    Y_AV{subj}.Y = Ys';
    Y{subj} = YoverTime;
    Y_FLIP{subj} = YoverTime2; 
    T{subj} = trialinfo;
end

save('new_flipped_results.mat','Y','Y_AV','Y_FLIP','NeuralWeights','TAX',...
    'T','AzChoice','AzRate','AzGroup','Topo','Topo1')

%%
%--------------------------------------------------------------------------
% Psychometric Curve Weights:
%--------------------------------------------------------------------------
PMCWeights = zeros(length(subs),2,2);
for subj = 1:length(subs);
    cd(sprintf('W:/Project 1/log/EEG Data/%s',subs(subj).name))
    WBeh = load(sprintf('all_S0%d',subj),'weightsObs');     % behavioural weights from PMC
    PMCWeights(subj,:,:) = WBeh.weightsObs;
end
% PMCWeights(subj,:,1) = [AH,VH];
% PMCWeights(subj,:,2) = [AL,VL];

PMCDiff(:,1) = PMCWeights(:,1,1) - PMCWeights(:,2,1);
PMCDiff(:,2) = PMCWeights(:,1,2) - PMCWeights(:,2,2);
% PMCDiff = [AH-VH, ];
cd('W:\Project 1\log\EEG Data')
%--------------------------------------------------------------------------

save('new_flipped_results.mat','Y','Y_AV','Y_FLIP','NeuralWeights','TAX',...
    'T','AzChoice','AzRate','AzGroup','Topo','Topo1','PMCWeights','PMCDiff')

%--------------------------------------------------------------------------
% PLOTS
%--------------------------------------------------------------------------
close all;
figure('units','normalized','outerposition',[0 0 1 1])
clear hline
cd('W:\Project 1\log\EEG Data')
I = find(TAX<=0.6);                 % Cut off everything at 0.6s
I = I(3:end);                       % Ignore the first few - no values (not enough stimuli)
TAX2 = TAX(I);

% AVH AVL DECODER AZ Signal
subplot 341
plot(TAX2,sq(mean(AzGroup(:,I))));
[I2,J2] = max(mean(AzGroup(:,I)));
hold on
plot(TAX2(J2),I2,'*r')
ylim([0.45 0.65 ])
hline(0.5,':k')
ylabel('Az')
title('AVH AVL Rate AZ')

% Decoder Topoplot of Best Az Time Point
subplot 342
load('ftDummy','EvpDummy')
EvpDummy.avg = sq(mean(Topo(:,:,J2)))';
cfg = [];
cfg.layout = 'biosemi64.lay';
cfg.comment = ' ';
% cfg.zlim = [-2 4];
ft_topoplotER(cfg,EvpDummy);
title(sprintf('AVH AVL trials \n Time: %s',num2str(TAX2(J2))));
colorbar

% AZ ALL OTHER TRIALS
subplot 343
plot(TAX2,sq(mean(AzRate(:,I))));
[I2,J2] = max(mean(AzRate(:,I)));
hold on
plot(TAX2(J2),I2,'*r')
ylim([0.45 0.7 ])
hline(0.5,':k')
ylabel('Az')
title('All Trials Rate AZ')

% topopplot for that az
subplot 344
load('ftDummy','EvpDummy')
EvpDummy.avg = sq(mean(Topo1(:,:,J2)))';
cfg = [];
cfg.layout = 'biosemi64.lay';
cfg.comment = ' ';
cfg.zlim = [-2 4];
ft_topoplotER(cfg,EvpDummy);
title(sprintf('All Trials \n Time: %s',num2str(TAX2(J2))));
colorbar


%--------------------------------------------------------------------------
% NEURAL WEIGHT PLOTS  :
%--------------------------------------------------------------------------
% High Reliability Neural Weights
subplot 345; hold on;
errorbar(TAX2,mean(NeuralWeights(:,I,1)),sem(NeuralWeights(:,I,1))); hold on
errorbar(TAX2,mean(NeuralWeights(:,I,3)),sem(NeuralWeights(:,I,3)),'r');
legend({'AH','AL'},'Location','NorthEast','FontSize',8);
% ylim([ -1 4.5 ])
hline(0,':k');
ylabel('Neural Weights');
title('High Rel')

% Low Reliability Neural Weights
subplot 346; hold on;
errorbar(TAX2,mean(NeuralWeights(:,I,2)),sem(NeuralWeights(:,I,2)));
errorbar(TAX2,mean(NeuralWeights(:,I,4)),sem(NeuralWeights(:,I,4)),'r');
legend({'VH','VL'},'Location','NorthEast','FontSize',8);
hline(0,':k');
% ylim([ -1 4.5 ])
ylabel('Neural Weights');
title('Low Rel')

%-------------------------------
% neural weights from regression
ND(:,1,:) = NeuralWeights(:,:,1)-NeuralWeights(:,:,2);      % AH-VH
ND(:,2,:) = NeuralWeights(:,:,3)-NeuralWeights(:,:,4);      % AL-VL
HD(:,1,:) = NeuralWeights(:,:,1) - NeuralWeights(:,:,3);    % AH - AL;
HD(:,2,:) = NeuralWeights(:,:,2) - NeuralWeights(:,:,4);    % VH - VL ;


%--------------------------------------------------------------------------
% TVALUES PlOTS -----%
%--------------------------------------------------------------------------
% high - low
subplot 347
tval = sq(mean(HD)./sem(HD));
plot(TAX2,tval(1,I));
hold on
plot(TAX2,tval(2,I),'r');
legend({'AH-AL','VH-VL'},'Location','NorthEast','FontSize',8)
hline(0,':k')
title('High-Low');
ylabel('Tvals')
ylim([-4 4.5])

% auditory - visual
subplot 348
tval = sq(mean(ND)./sem(ND));
plot(TAX2,tval(1,I));
hold on
plot(TAX2,tval(2,I),'r');
legend({'AH-VH','AL-VL'},'Location','NorthEast','FontSize',8)
hline(0,':k')
title('A-V');
ylabel('Tvals')
ylim([-4 4.5])



%--------------------------------------------------------------------------
% Normalise behavioural weights:
%--------------------------------------------------------------------------
load('AVH_AVL_DIAG_RATE_NO11HZ_ALLTRIALS.mat','BWeights')
BB = cat(4,BWeights(:,:,(1:2)),BWeights(:,:,(1:2)+2));
BB = permute(BB,[1,4,2,3]);
BB =BB./repmat(sum(BB(:,:,:,[1,2]),4),[1,1,1,2]);
BD = BB(:,:,:,1)-BB(:,:,:,2);
BA = BB(:,:,:,1);
BV = BB(:,:,:,2);


save('new_flipped_results.mat','Y','Y_AV','Y_FLIP','NeuralWeights','TAX',...
    'T','AzChoice','AzRate','AzGroup','Topo','Topo1','PMCWeights','PMCDiff',...
    'BWeights','ND','HD','BB','BD','BA','BV')


%--------------------------------------------------------------------------
% Correlations:
%--------------------------------------------------------------------------
for t = 1:75;
    % behavioural regression vs neural regression weights:
    [R{1}(t),P{1}(t)] = spearmanrank(BD(:,1,t)-BD(:,2,t), ND(:,1,t)-ND(:,2,t));
    
    % Psychometric curve weights vs Neural regression weights:
    [R{2}(t),P{2}(t)] = spearmanrank(PMCDiff(:,1)-PMCDiff(:,2), ND(:,1,t)-ND(:,2,t));
    
    % Psychometric curve weights vs behavioural regression
    [R{3}(t),P{3}(t)] = spearmanrank(PMCDiff(:,1)-PMCDiff(:,2), BD(:,1,t)-BD(:,2,t));
end

% behavioural regression vs neural regression weights
subplot 349
v = 1;
plot(TAX2,R{v}(I))
j = find(P{v}(I)<0.05);
hold on
ylim([-1 1])
plot(TAX2(j),R{v}(I(j)),'*r')
title('BD vs ND (uncorrected)')
j2{v} = j;
hline(0,':k');

% Psychometric curve weights vs Neural regression weights
v = 2;
subplot(3,4,10);
plot(TAX2,R{v}(I))
j = find(P{v}(I)<0.05);
hold on
plot(TAX2(j),R{v}(I(j)),'*r')
title('PMC vs ND (uncorrected)')
j2{v} = j;
ylim([-1 1])
hline(0,':k');

% Psychometric curve weights vs behavioural regression
v = 3;
subplot(3,4,11);
plot(TAX2,R{v}(I))
j = find(P{v}(I)<0.05);
hold on
plot(TAX2(j),R{v}(I(j)),'*r')
title('PMC vs BD (uncorrected)')
j2{v} = j;
ylim([-1 1])
hline(0,':k');


%%
save('new_flipped_results.mat','Y','Y_AV','Y_FLIP','NeuralWeights','TAX',...
    'T','AzChoice','AzRate','AzGroup','Topo','Topo1','PMCWeights','PMCDiff',...
    'BWeights','ND','HD','BB','BD','BA','BV','TAX2','R','P')

% %% Save the file
% cd('W:\Project 1\log\EEG Data');
% save(sprintf('AVH_AVL_DIAG_%s_NO11HZ_ALLTRIALS.mat',COND),'NeuralWeights','BWeights','Topo','Topo1','TAX','Decode_times2',...
%     'trange','AzRate','AzChoice','AzGroup','Y','T','actS','Wsave','Csave','R',...
%     'P','j2','PMCDiff','PMCWeights','COND',...
%     'stimTimes','TAX2','Y_AV','AzGroup','ND','BD','HD');



%% CLUSTER STATS ON CORRELATIONS

%--------------------------------------------------------------------------
% STATS FOR CORRELATIONS - NEURAL VS BEHAVIOURAL
n = size(ND,1);
Tshuf =[]; Ttrue=[];

I = 3:51;
for k=1:1000
    
    for t=3:51
        
        if k==1
            [r,p,tval]  = spearmanrank( BD(:,1,t)-BD(:,2,t),ND(:,1,t)-ND(:,2,t));
            Ttrue(1,t) = tval;
        end
        
        order = randperm(n);
        if sum(abs(order-[1:n]))==0  % avoid the tru order of subjects
            order = randperm(n);
        end
        [r,p,tval] = spearmanrank(BD(:,1,t)-BD(:,2,t),ND(order,1,t)-ND(order,2,t));
        Tshuf(1,t,k) = tval; % has to be a 3D matrix
    end
end


% clf
cfg.critval = 1.8;                    % threshold for signif t-values
cfg.clusterstatistic = 'maxsize';     % maxsize maxsum
cfg.critvaltype = 'par';              % parametric threshold
cfg.minsize = 2;
cfg.pval = 0.05;                      % threshold to select signifciant clusters
cfg.df = 19;

[PC_corr,NC_corr] = eegck_clusterstats(cfg,Ttrue,Tshuf)

plot(TAX(I),(R{1}(I)));
ind =  find(PC_corr.maskSig~=0);
hold on
plot(TAX(ind),R{1}(ind),'*r');
title(sprintf('Signif Corr BD,ND, crit val:%s, method: %s \n %s',num2str(cfg.critval),cfg.clusterstatistic,num2str(TAX(ind))))

% save('ND_BD_Corr_Cluster.mat','Tshuf','Ttrue','PC_corr','NC_corr');
save('ND_BD_Corr_Cluster_NEW.mat','Tshuf','Ttrue','PC_corr','NC_corr');



%%
%--------------------------------------------------------------------------
% STATS FOR CORRELATIONS - BEHAVIOURAL VS PMC
n = size(ND,1);
Tshuf =[]; Ttrue=[];
I = 3:51;
for k=1:1000
    
    for t=3:51
        
        if k==1
            [r,p,tval]  = spearmanrank( BD(:,1,t)-BD(:,2,t),PMCDiff(:,1)-PMCDiff(:,2));
            Ttrue(1,t) = tval;
        end
        
        order = randperm(n);
        if sum(abs(order-[1:n]))==0  % avoid the tru order of subjects
            order = randperm(n);
        end
        [r,p,tval] = spearmanrank(BD(:,1,t)-BD(:,2,t),PMCDiff(order,1)-PMCDiff(order,2));
        Tshuf(1,t,k) = tval; % has to be a 3D matrix
    end
end


% clf
cfg.critval = 1.8;                    % threshold for signif t-values
cfg.clusterstatistic = 'maxsize';     % maxsize maxsum
cfg.critvaltype = 'par';              % parametric threshold
cfg.minsize = 2;
cfg.pval = 0.05;                      % threshold to select signifciant clusters
cfg.df = 19;
clf;
[PC_corr_BD_PMC,NC_corrAV_BD_PMC] = eegck_clusterstats(cfg,Ttrue,Tshuf)

plot(TAX(I),(R{3}(I)));
ind =  find(PC_corr_BD_PMC.maskSig~=0);
hold on
plot(TAX(ind),R{3}(ind),'*r');
title(sprintf('PC maskSig, crit val:%s, method: %s \n BWEIGHTS: PMC WEIGHTS',num2str(cfg.critval),cfg.clusterstatistic))


% save('BD_PMC_Corr_Cluster.mat','Tshuf','Ttrue','PC_corr_BD_PMC','NC_corr_BD_PMC');
save('BD_PMC_Corr_Cluster_NEW.mat','Tshuf','Ttrue','PC_corr_BD_PMC','NC_corr_BD_PMC');

%% --------------------------------------------------------------------------
% STATS FOR CORRELATIONS - ND VS PMC
n = size(ND,1);
Tshuf =[]; Ttrue=[];
I = 3:51;
for k=1:1000
    
    for t=3:51
        
        if k==1
            [r,p,tval]  = spearmanrank( ND(:,1,t)-ND(:,2,t),PMCDiff(:,1)-PMCDiff(:,2));
            Ttrue(1,t) = tval;
        end
        
        order = randperm(n);
        if sum(abs(order-[1:n]))==0  % avoid the tru order of subjects
            order = randperm(n);
        end
        [r,p,tval] = spearmanrank(ND(:,1,t)-ND(:,2,t),PMCDiff(order,1)-PMCDiff(order,2));
        Tshuf(1,t,k) = tval; % has to be a 3D matrix
    end
end


% clf
cfg.critval = 1.8;                    % threshold for signif t-values
cfg.clusterstatistic = 'maxsize';     % maxsize maxsum
cfg.critvaltype = 'par';              % parametric threshold
cfg.minsize = 2;
cfg.pval = 0.05;                      % threshold to select signifciant clusters
cfg.df = 19;
clf;
[PC_corr_ND_PMC,NC_corr_ND_PMC] = eegck_clusterstats(cfg,Ttrue,Tshuf)

plot(TAX(I),(R{2}(I)));
ind =  find(NC_corr_ND_PMC.maskSig~=0);
hold on
plot(TAX(ind),R{2}(ind),'*r');
title(sprintf('PC maskSig, crit val:%s, method: %s \n BWEIGHTS: PMC WEIGHTS',num2str(cfg.critval),cfg.clusterstatistic))
% print('beh_neural_weights_corr_2clusters','-depsc');

% save('ND_PMC_Corr_Cluster.mat','Tshuf','Ttrue','PC_corr_ND_PMC','NC_corr_ND_PMC');
save('ND_PMC_Corr_Cluster_NEW.mat','Tshuf','Ttrue','PC_corr_ND_PMC','NC_corr_ND_PMC');





%% Neural Weight differences:
% AH-AL and VH-VL
% TVAL SHUFFLE CLUSTER DIFFERNCE - NEURAL

NeuralA = NeuralWeights(:,:,[1,3]);     % AH, AL
NeuralV = NeuralWeights(:,:,[2,4]);     % VH, VL

Ause = zeros(20,51,2,1000);
Vuse = zeros(20,51,2,1000);
oSave = zeros(20,4,51,100);

%---------------------------------------------------------------
% Create the shuffled data matrics first (quicker)
% tic
for b = 1:1000;
    
    for t=3:51
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
TshufA = zeros(1,51,boot);
TshufV = zeros(1,51,boot);
TtrueA = zeros(1,51); TtrueV = zeros(1,51);
for k = 1:boot;
    
    for t = 3:51;
        
        if k==1
            % true one
            [HA,PA,CIA,STATSA] = ttest(NeuralWeights(:,t,1),NeuralWeights(:,t,3)); % AUDITORY (HIGH,LOW)
            [HV,PV,CIV,STATSV] = ttest(NeuralWeights(:,t,2),NeuralWeights(:,t,4)); % VISUAL   (HIGH,LOW)
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

% clf;
[PC_NW_AUD,NC_NW_AUD] = eegck_clusterstats(cfg,TtrueA,TshufA)
[PC_NW_VIS,NC_NW_VIS] = eegck_clusterstats(cfg,TtrueV,TshufV)

%------------------------------------------------
% AUDITORY PLOT
subplot 211
plot(TAX(I),TtrueA(I))
% ind =  find(PC_NW_AUD.maskSig~=0);
indA = find(PC_NW_AUD.maskSig~=0 | NC_NW_AUD.maskSig~=0); % new Results
hold on
plot(TAX(indA),TtrueA(indA),'*r');
title(sprintf('MASK, crit val:%s, method: %s \n AUDITORY H-L',num2str(cfg.critval),cfg.clusterstatistic))
ylabel('tval')

% VISUAL PLOT
subplot 212
plot(TAX(I),TtrueV(I),'g')
% ind =  find(NC_NW_VIS.maskSig~=0);
indV = find(PC_NW_VIS.maskSig~=0 | NC_NW_VIS.maskSig~=0); % new Results
hold on
plot(TAX(indV),TtrueV(indV),'*m');
title(sprintf('PC maskSig, crit val:%s, method: %s \n VISUAL H-L',num2str(cfg.critval),cfg.clusterstatistic))
ylabel('tval')

% save('Cluster_NW_HL.mat','TtrueA','TtrueV','TshufA','TshufV','NC_NW_AUD','PC_NW_AUD','NC_NW_VIS','PC_NW_VIS','oSave','Ause','Vuse');
save('Cluster_NW_HL_NEW.mat','TtrueA','TtrueV','TshufA','TshufV','NC_NW_AUD','PC_NW_AUD','NC_NW_VIS','PC_NW_VIS','oSave','Ause','Vuse','indA','indV');


%% PLOTS 

% AUDITORY HIGH - LOW NEURAL WEIGHTS SIGNIFICANT TIME POINTS
clf
subplot 121 
plot(TAX2,NW(:,1)-NW(:,3))
hold on
plot(TAX2(indA),NW(indA,1)-NW(indA,3),'*r')
title('AUD NEURAL WEIGHTS HIGH - LOW')
ylim([-1.2 1.2]); hline(0)

% VISUAL WEIGHTS 
subplot 122
plot(TAX2,NW(:,2)-NW(:,4),'r')
hold on
plot(TAX2(indV),NW(indV,2)-NW(indV,4),'*k')
title('VIS NEURAL WEIGHTS HIGH - LOW')
ylim([-1.2 1.2]); hline(0)


% %% OLD "WRONG" NEURAL WEIGHTS
% % Split it up into auditory clusters and visual clusters:
% indAud =  find(NC_NW_AUD.maskSig~=0);
% indVis =  find(PC_NW_VIS.maskSig~=0);
% 
% % Aud cluster 1:
% Aud_CL1 = sq(mean(NeuralWeights(:,indAud(1:5),[1 3]),2));       % AH, AL    (156:192ms)
% Aud_CL2 = sq(mean(NeuralWeights(:,indAud(6:end),[1 3]),2));     % AH, AL    (204:276ms)
% 
% Vis_CL1 = sq(mean(NeuralWeights(:,indVis(1:3),[2 4]),2));       % vh, vl    (84:108ms)
% Vis_CL2 = sq(mean(NeuralWeights(:,indVis(4:end),[2 4]),2));     % vh, vl    (252:288ms)

%%
% Split it up into auditory clusters and visual clusters:
% indAud =  find(NC_NW_AUD.maskSig~=0);
% indVis =  find(PC_NW_VIS.maskSig~=0);

% Aud cluster 1:
Aud_CL1 = sq(mean(NeuralWeights(:,indA(1:2),[1 3]),2));       % AH, AL    96:108ms
Aud_CL2 = sq(mean(NeuralWeights(:,indA(3:end),[1 3]),2));     % AH, AL    ( 0.4440    0.4560    0.4680)

Vis_CL1 = sq(mean(NeuralWeights(:,indV(1:2),[2 4]),2));       % vh, vl    (108:120)
Vis_CL2 = sq(mean(NeuralWeights(:,indV(3:end),[2 4]),2));     % vh, vl    (444:468)





%% SIGNIFICANT TIME POINTS - BOXPLOTS 
subplot 221
boxplot(Aud_CL1);
title(sprintf('Auditory Cluster 1 \n Time: %s to %s',num2str(TAX(indA(1))), num2str(TAX(indA(2)))));
set(gca,'XTick',1:2,'XTickLabel',{'AVH','AVL'});
ylabel('Auditory Weight')
a = median(Aud_CL1);

topoAUDCL1 = sq(mean(mean(Topo1(:,:,indA(1:2)),3),1));
cfg = [];
cfg.layout = 'biosemi64.lay';
EvpDummy.avg = topoAUDCL1';
ft_topoplotER(cfg,EvpDummy);
colorbar
title(sprintf('AUD FIRST CLUSTER, TIME = %s ms',num2str(TAX(indA(1:2)))))
suptitle('RELIABILITY EFFECT: AUD HIGH VS AUD L')

subplot 222
boxplot(Aud_CL2);
title(sprintf('Auditory Cluster 2 \n Time: %s to %s',num2str(TAX(indA(3))), num2str(TAX(indA(end)))));
set(gca,'XTick',1:2,'XTickLabel',{'AVH','AVL'});
ylabel('Auditory Weight')
a = median(Aud_CL2);

clf
topoAUDCL2 = sq(mean(mean(Topo1(:,:,indA(3:end)),3),1));
cfg = [];
cfg.layout = 'biosemi64.lay';
EvpDummy.avg = topoAUDCL2';
ft_topoplotER(cfg,EvpDummy);
colorbar
title(sprintf('AUD SECOND CLUSTER, TIME = %s ms',num2str(TAX(indA(3:end)))))
suptitle('RELIABILITY EFFECT: AUD HIGH VS AUD L')



% b = a(1);
% c = a(2);
% text(0.87,b+0.2,num2str(b));
% text(1.87,c+0.2,num2str(c));

subplot 223
boxplot(Vis_CL1);
title(sprintf('Visual Cluster 1 \n Time: %s to %s',num2str(TAX(indV(1))), num2str(TAX(indV(2)))));
set(gca,'XTick',1:2,'XTickLabel',{'AVH','AVL'});
ylabel('Visual Weight')

clf
topoVISCL1 = sq(mean(mean(Topo1(:,:,indV(1:2)),3),1));
cfg = [];
cfg.layout = 'biosemi64.lay';
EvpDummy.avg = topoVISCL1';
ft_topoplotER(cfg,EvpDummy);
colorbar
title(sprintf('VISUAL FIRST CLUSTER, TIME = %s ms',num2str(TAX(indV(1:2)))))
suptitle('RELIABILITY EFFECT: VIS H VS VIS L')

% a = median(Vis_CL1);
% b = a(1);
% c = a(2);
% text(0.87,b+0.2,num2str(b));
% text(1.87,c+0.2,num2str(c));

subplot 224
boxplot(Vis_CL2);
title(sprintf('Visual Cluster 2 \n Time: %s to %s',num2str(TAX(indV(3))), num2str(TAX(indV(end)))));
set(gca,'XTick',1:2,'XTickLabel',{'AVH','AVL'});
ylabel('Visual Weight')

clf
topoVISCL2 = sq(mean(mean(Topo1(:,:,indV(3:end)),3),1));
cfg = [];
cfg.layout = 'biosemi64.lay';
EvpDummy.avg = topoVISCL2';
ft_topoplotER(cfg,EvpDummy);
colorbar
title(sprintf('VISUAL SECOND CLUSTER, TIME = %s ms',num2str(TAX(indV(3:end)))))
suptitle('RELIABILITY EFFECT: VIS H VS VIS L')


% a = median(Vis_CL2);
% b = a(1);
% c = a(2);
% text(0.87,b+0.2,num2str(b));
% text(1.87,c+0.2,num2str(c));


suptitle(sprintf('Neural Weights \n Avg over the time windows for AH vs AL, VH vs VL'))




%% AUDITORY NEURAL WEIGHTS VS VISUAL NEURAL WEIGHTS OVER TIME,
% FOR EACH LEVEL OF RELIABILITY

NeuralA = NeuralWeights(:,:,[1,3]);     % AH, AL
NeuralV = NeuralWeights(:,:,[2,4]);     % VH, VL

Ause = zeros(20,51,2,1000);
Vuse = zeros(20,51,2,1000);
oSave = zeros(20,4,51,100);

%---------------------------------------------------------------
% Create the shuffled data matrics first (quicker)
% tic
for b = 1:1000;
    
    for t=3:51
        for s = 1:20;
            
            order = randi(2,1,4);   % random 1s and 2s
            
            % AUDITORY
            Ause(s,t,1,b) = NeuralA(s,t,order(1));
            Ause(s,t,2,b) = NeuralA(s,t,order(2));
            
            % VISUAL
            Vuse(s,t,1,b) = NeuralV(s,t,order(3));
            Vuse(s,t,2,b) = NeuralV(s,t,order(4));
            
            %             oSave(s,:,t,k) = order; % just to check later
        end
        
    end
end
% toc

%---------------------------------------------------------------
% Do the ttests
tic
boot = 1000;
TshufA = zeros(1,51,boot);
TshufV = zeros(1,51,boot);
TtrueA = zeros(1,51); TtrueV = zeros(1,51);
for k = 1:boot;
    
    for t = 3:51;
        
        if k==1
            % true one
            [HA,PA,CIA,STATSA] = ttest(NeuralWeights(:,t,1),NeuralWeights(:,t,2)); % AUDITORY (HIGH-LOW)
            [HV,PV,CIV,STATSV] = ttest(NeuralWeights(:,t,3),NeuralWeights(:,t,4)); % VISUAL   (HIGH-LOW)
            TtrueA(1,t) = STATSA.tstat;
            TtrueV(1,t) = STATSV.tstat;
        end
        
        
        [~,~,~,STATSA] = ttest(Ause(:,t,1,k),Vuse(:,t,1,k)); % AUDITORY (1 - 2)
        [~,~,~,STATSV] = ttest(Ause(:,t,2,k),Vuse(:,t,2,k)); % VISUAL   (1 - 2)
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
% clf;
[PC_AV_HIGH,NC_AV_HIGH] = eegck_clusterstats(cfg,TtrueA,TshufA);
[PC_AV_LOW,NC_AV_LOW] = eegck_clusterstats(cfg,TtrueV,TshufV);

% save('Cluster_NW_AV.mat','TtrueA','TtrueV','TshufA','TshufV','NC_AV_HIGH','PC_AV_HIGH','NC_AV_LOW','PC_AV_LOW','oSave','Ause','Vuse');
save('Cluster_NW_AV_NEW.mat','TtrueA','TtrueV','TshufA','TshufV','NC_AV_HIGH','PC_AV_HIGH','NC_AV_LOW','PC_AV_LOW','oSave','Ause','Vuse');

% AH VS VH
indAH = find(PC_AV_HIGH.mask~=0);
indVH = find(NC_AV_HIGH.mask~=0);

% Al VS Vl
indAL = find(PC_AV_LOW.mask~=0);
indVL = find(NC_AV_LOW.mask~=0);

clf
subplot 121; hold on;
plot(TAX(I),mean(NeuralWeights(:,I,1)))
plot(TAX(I),mean(NeuralWeights(:,I,2)),'m')
legend('AH','VH')
% plot significant cluster difference time points
plot(TAX(indAH),mean(NeuralWeights(:,indAH,1)),'*k')
plot(TAX(indVH),mean(NeuralWeights(:,indVH,2)),'*k')
title('high rel');

subplot 122; hold on;
plot(TAX(I),mean(NeuralWeights(:,I,3)))
plot(TAX(I),mean(NeuralWeights(:,I,4)),'m')
legend('AL','VL')
hold on
plot(TAX(indAL),mean(NeuralWeights(:,indAL,3)),'*k')
plot(TAX(indVL),mean(NeuralWeights(:,indVL,4)),'*k')
title('low rel');


% plot(mean(NeuralWeights(:,:,3)),':b')
% plot(mean(NeuralWeights(:,:,4)),':m')
% legend('AH','VH','AL','VL')



%% Neural Weights - Auditory/visual weights over time & significant cluster
% boxplots

load('ND_BD_Corr_Cluster.mat','PC_corr');

ind =  find(PC_corr.maskSig~=0);
ind1 = ind(1:2);                    % first cluster % 0.12 : 0.132
% ind2 = ind(3:end);                  % second cluster % 204 : 228

I = 3:51;           % restrict time window
subplot 121
errorbar(TAX(I),mean(NeuralWeights(:,I,1)),sem(NeuralWeights(:,I,1))); hold on
errorbar(TAX(I),mean(NeuralWeights(:,I,3)),sem(NeuralWeights(:,I,3)),'r');
legend('AH','AL');
title('Aud Neural Weights')

subplot 122
errorbar(TAX(I),mean(NeuralWeights(:,I,2)),sem(NeuralWeights(:,I,2))); hold on
errorbar(TAX(I),mean(NeuralWeights(:,I,4)),sem(NeuralWeights(:,I,4)),'r');
legend('VH','VL');
title('Vis Neural Weights')

% get the weights
cluster1 = sq(mean(NeuralWeights(:,ind1,1:4),2));
% cluster2 = sq(mean(NeuralWeights(:,ind2,1:4),2));

% rearrange to be AH,AL, VH, VL
cluster1_NEW = [cluster1(:,1),cluster1(:,3),cluster1(:,2),cluster1(:,4)];
% cluster2_NEW = [cluster2(:,1),cluster2(:,3),cluster2(:,2),cluster2(:,4)];
load('ftDummy','EvpDummy');

% make the plots
subplot 221
boxplot(cluster1_NEW)
hold on
plot(1,cluster1_NEW(:,1),'.','MarkerSize',10,'Color',[0.7 0.7 0.7])
plot(2,cluster1_NEW(:,2),'.','MarkerSize',10,'Color',[0.7 0.7 0.7])
plot(3,cluster1_NEW(:,3),'.','MarkerSize',10,'Color',[0.7 0.7 0.7])
plot(4,cluster1_NEW(:,4),'.','MarkerSize',10,'Color',[0.7 0.7 0.7])
title(sprintf('Cluster 1 %s avg',num2str(TAX(ind))))
set(gca,'XTick',1:4,'XTickLabel',{'AH','AL','VH','VL'});
ylim([-1 7])


subplot 222
topo1st = sq(mean(mean(Topo1(:,:,ind1),3),1));
cfg = [];
cfg.layout = 'biosemi64.lay';
EvpDummy.avg = topo1st';
ft_topoplotER(cfg,EvpDummy);
colorbar

% subplot 223
% boxplot(cluster2_NEW)
% hold on
% plot(1,cluster2_NEW(:,1),'.','MarkerSize',10,'Color',[0.7 0.7 0.7])
% plot(2,cluster2_NEW(:,2),'.','MarkerSize',10,'Color',[0.7 0.7 0.7])
% plot(3,cluster2_NEW(:,3),'.','MarkerSize',10,'Color',[0.7 0.7 0.7])
% plot(4,cluster2_NEW(:,4),'.','MarkerSize',10,'Color',[0.7 0.7 0.7])
% title('Cluster 2 - 204:228 avg')
% set(gca,'XTick',1:4,'XTickLabel',{'AH','AL','VH','VL'});
% ylim([-1 7])
% 
% 
% subplot 224
% topo2nd = sq(mean(mean(Topo1(:,:,ind2),3),1));
% cfg = [];
% cfg.layout = 'biosemi64.lay';
% EvpDummy.avg = topo2nd';
% ft_topoplotER(cfg,EvpDummy);
% colorbar


% tests
n = length(cluster1_NEW); % number of subs

% CLUSTER 1 (120:132MS)
% AH VS AL
[p,h,stats] = signrank(cluster1_NEW(:,1),cluster1_NEW(:,2))
effect = stats.zval/sqrt(n*2)
% VH vs VL
[p,h,stats] = signrank(cluster1_NEW(:,3),cluster1_NEW(:,4))
effect = stats.zval/sqrt(n*2)
% AH vs VH
[p,h,stats] = signrank(cluster1_NEW(:,1),cluster1_NEW(:,3))
effect = stats.zval/sqrt(n*2)
% AL vs VL
[p,h,stats] = signrank(cluster1_NEW(:,2),cluster1_NEW(:,4))
effect = stats.zval/sqrt(n*2)

% clc
% % CLUSTER 2 (204:228MS)
% % AH VS VH
% [p,h,stats] = signrank(cluster2_NEW(:,1),cluster2_NEW(:,2))
% effect = stats.zval/sqrt(n*2)
% % VH vs VL
% [p,h,stats] = signrank(cluster2_NEW(:,3),cluster2_NEW(:,4))
% effect = stats.zval/sqrt(n*2)
% % AH vs VH
% [p,h,stats] = signrank(cluster2_NEW(:,1),cluster2_NEW(:,3))
% effect = stats.zval/sqrt(n*2)
% % AL vs VL
% [p,h,stats] = signrank(cluster2_NEW(:,2),cluster2_NEW(:,4))
% effect = stats.zval/sqrt(n*2)
% 
% % (AH-AL CLUSTER 1) - (AH-AL CLUSTER 2);
% diffs = [cluster1_NEW(:,1)-cluster1_NEW(:,2), cluster2_NEW(:,1)-cluster2_NEW(:,2)...
%     , cluster1_NEW(:,3)-cluster1_NEW(:,4), cluster2_NEW(:,3)-cluster2_NEW(:,4)];
% % [AUD 1, AUD 2, VIS 1, VIS 2];
% clc
% % (AH-AL) vs (AH-AL);
% [p,h,stats] = signrank(diffs(:,1),diffs(:,2))
% effect = stats.zval/sqrt(n*2)
% % (VH-VL) vs (VH-VL);
% [p,h,stats] = signrank(diffs(:,3),diffs(:,4))
% effect = stats.zval/sqrt(n*2)

%%
% CORRELATE THE RELIABILITY EFFECT ON THE DOUBLE DIFFERENEC BETWEEN
% CLUSTERS
clear DD
DD(:,1) = (cluster1_NEW(:,1)-cluster1_NEW(:,3)) - (cluster1_NEW(:,2) -cluster1_NEW(:,4)); % CLUSTER 1 (AH-VH)-(AL-VL)
DD(:,2) = (cluster2_NEW(:,1)-cluster2_NEW(:,3)) - (cluster2_NEW(:,2) -cluster2_NEW(:,4)); % CLUSTER 2 (AH-VH)-(AL-VL)
[r,p,t] = spearmanrank(DD(:,1),DD(:,2))


%% Correlations between the two signifcant cluster time points topographies.

load('ND_BD_Corr_Cluster.mat','PC_corrAV')
ind =  find(PC_corrAV.maskSig~=0); % find the significant neuro-beh correlation time points between 0 and 600ms (between 1 and 51);
% ind = [11 12 18 19 20]
cluster1 = sq(mean(Topo1(:,:,ind(1:2)),3));     % average the topography over that time window
cluster2 = sq(mean(Topo1(:,:,ind(3:end)),3));

for s = 1:20;
    [r(s),p(s)] = spearmanrank(cluster1(s,:),cluster2(s,:));
end

bar(r)
ind2 = find(p<0.05);
hold on; plot(ind2,r(ind2),'*r')



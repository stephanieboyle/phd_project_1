%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PREPROCESSING SCRIPT FOR INDIVIDUAL SUBJECTS. - steph 21/10/15
%--------------------------------------------------------------------------
% 1. Does individual preprocessing on blocks/days & preprocessing plots. 
% 2. EOG analysis for individual subjects/blocks/days & plots.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% IF YOU CHANGE THE NUMBERS OF TRIALS IN THE BLOCK, YOU NEED TO CHANGE THE
% VARIABLE values (see later in script)!
clc
clear
close all
cd('W:\Project 1\log\EEG Data');
addpath('\\analyse2\Project0109\Lab\ckmatlab\eegck')

%--------------------------------------------------------------------------
% PART 1. PREPROCESSING : Preprocesses, resamples, compute eye movement
% signals, power spectrum density & plots, ICA Based Artifact Rejection
% (automatic), calculates power spectrum components (for eye artifact
% removal), calculates eye movement/component correlations, then removes
% all components (automatic). Finally does trial rejection based on
% excessive amplitude. 


% PREPRO PARAMETERS
ARG.Amp_thr = 120;                                                          % threshold for complete trial rejection
ARG.Resample = 200;                                                         % sampling rate, use 150 or 200
ARG.Highpass = 0.5;                                                         % high pass filter

% Artifact Removal parameters
ARG.Remove = 'all';                                                         % 'fast' for  fast processing and removal of eye topo only 'all' for removal of other noise sources
ARG.ICA_artif_thr = 0.7;                                                    % threshold: correlation with template
ARG.ICA_eyecorr = 0.08;                                                     % threshold : correlation with eye movement signal (only for 'all')
ARG.ICA_spectrum_thr = 6;                                                   % spectrum on component amplitude (only for 'all')

% Get subject info
sub_list = dir('Subj_*');                                                   % make a list of subjects
subs = cell(1,(length(sub_list)));
for k = 1:length(sub_list)
    subs{k} = sub_list(k).name;                                             % take out subject names
end

% Preprocess one subject at a time: 
subj = input('Subj?: ');
cd(subs{subj})
addpath('W:\Project 1\log\EEG Data');
sname_text = sprintf('Subj_0%d_Components.txt',subj);
fileID = fopen(sname_text,'a+');

dayList = dir('Exp_S*_B*_D*');                                              % different days
days = length(dayList)/2;               % 2 blocks per day
dayNums = 1:days;


% PREPROCESSING LOOP 
for kn = 1 : days
    
    dayInp = dayNums(kn);
    fnameBehav = dir(sprintf('Exp_S*_B*_D%d*',dayInp));             % list of behavioural files
    fnameEEG = dir(sprintf('Rel_*_D%d*.bdf',dayInp));               % list of EEG files
    
    [labels,LabelOrig] = eegck_BiosemiLabels();                     % Get labels 
    
    % TRIAL PARAMETERS 
    trialdef.prestim = -1;                                          % pre trigger secs
    trialdef.poststim = 2;                                          % after trigger secs
    trigRange = 1;
    trigs = length(trigRange);                                      % for later (to work out if there are 2 sets of triggers or 1)
    
    % REREFERENCING
    reref = 0;                                                      % none
    
    %----------------------------------------------------------------------
    % LOOP BLOCKS
    for s = 1 : length(fnameEEG)
        
        fprintf('\n \n \nSubject %d \nDay %d \nBlock %d \n \n \n \n',subj,dayInp,s);
        
        %------------------------------------------------------------------
        % READ EVENTS AND HEADER FILES 
        fullname = fnameEEG(s).name;                                % EEG FILE
        matname = fnameBehav(s).name;                               % BEHAVIOURAL MATLAB FILE
        cfg = [];
        cfg.dataset = fullname;                                     % EEG data.
        event = ft_read_event(cfg.dataset);
        hdr   = ft_read_header(cfg.dataset);
        eventS = ft_filter_event(event, 'type','STATUS');           % find all triggers
        
        %------------------------------------------------------------------
        % FIND TRIGGERS
        clear val t_onset
        c=1; % counter

        if subj == 7;
            eventS(1351:1364) = [];                                 % triggers recorded wrong for S07
        end
       
        for k=1:length(eventS)
            if sum(eventS(k).value==trigRange) && (eventS(k).sample>1)
                val(c) = eventS(k).value;                          % trigger value
                t_onset(c) = eventS(k).sample;                     % sample number
                c=c+1;
            end
        end
        fprintf('Found %d trials \n',c-1);
        
        %------------------------------------------------------------------
        % LOAD IN MATLAB DATA
        load(matname)
        A = load(matname);
        if subj == 7;
            infoP(451:455,:) = [];                              % triggers recorded wrong - delete in matfile as well 
        end
      
        %------------------------------------------------------------------
        % PUT DATA INTO FIELDTRIP TRIAL STRUCTURE
        trl =[];
        for t=1:length(t_onset)
            ts =t_onset(t);
            begsample     = ts + trialdef.prestim*hdr.Fs; % sample at trigger
            endsample     = ts + trialdef.poststim*hdr.Fs; % sample at end
            offset        = trialdef.prestim*hdr.Fs; % offset
            trl(t,(1:3)) = round([begsample endsample offset]);
            trl(t,4) = val(t); % add in the trigger values
        end
        trl = cat(2,trl,infoP);                                % ADD BEHAVIOURAL DATA INTO FIELDTRIP DATA STRUCTURE
        % [beg,end, offset,trigger value, infoP table]

        %------------------------------------------------------------------
        % PREPROCESSING
        cfg = [];
        cfg.dataset     = fullname; % EEG data name
        cfg.trl = trl;
        cfg.demean     = 'no';
        cfg.detrend = 'no';
        cfg.polyremoval   = 'no';
        cfg.lpfilter   = 'yes';                              % apply lowpass filter
        cfg.lpfreq     = 90;                                 % lowpass at 80 Hz.
        cfg.hpfilter   = 'yes';                              % apply highpass filter
        cfg.hpfreq     = ARG.Highpass;
        cfg.hpfiltord = 4;
        cfg.reref         = 'no';                           % referencing
        cfg.continuous = 'yes';
        data = ft_preprocessing(cfg);
        [dataX] = eegck_BiosemiEselect(data);               % just choose the 64 electrodes
        
        %------------------------------------------------------------------
        % subj 2 day1 B1 = plugged electrodes in  wrong way round - need to
        % swap data around so it's matched with the right label:
        if subj ==2 && strcmp(fnameEEG(s).name,'Rel_S02_B1_D1_0825_1000.bdf');
            for m = 1:length(dataX.trialinfo);
                new{m}(1:32,:) = dataX.trial{1,m}(33:64,:);
                new{m}(33:64,:) = dataX.trial{1,m}(1:32,:);
                new{m}(65:68,:) = dataX.trial{1,m}(65:68,:);
            end
            dataX.trial = [];
            dataX.trial = new;
        end
        
        %------------------------------------------------------------------
        % RESAMPLE THE DATA
        cfg            = [];
        cfg.resamplefs = ARG.Resample; % resampling freq
        cfg.detrend    = 'no';
        dataX           = ft_resampledata(cfg, dataX);
        
        %------------------------------------------------------------------
        % COMPUTE EYE MOVEMENT SIGNALS
        % (THIS ASSUMES THAT ALL FOUR EOG ELECTRODES WERE USED)
        selchan = ft_channelselection({'EXG1' 'EXG2' 'EXG3' 'EXG4'}, dataX.label);
        data_eog = ft_selectdata(dataX, 'channel', selchan);
        % VEOG as difference between (EX3 - EX1) and (EX4-EX2)
        % HEOG as difference EX3-EX4
        % REOG as difference mean(EX1-4) - mean (P1,Pz,P2)
        
        selchan = eegck_returnEIndex({'Pz','P1','P2'},dataX.label);
        for t=1:length(data_eog.trial)
            veog = (data_eog.trial{t}(3,:)- data_eog.trial{t}(1,:))+ ...
                (data_eog.trial{t}(4,:)- data_eog.trial{t}(2,:));
            veog = veog/2;
            heog = (data_eog.trial{t}(3,:)- data_eog.trial{t}(4,:));
            reog = mean(data_eog.trial{t},1)-mean(dataX.trial{t}(selchan,:));
            % z-score each
            data_eog.trial{t} = [cksig2z(veog);cksig2z(heog);cksig2z(reog)];
        end
        data_eog.label ={'VEOG','HEOG','REOG'};
        
        %------------------------------------------------------------------
        % PLOT RAW DATA BEFORE ICA
        XXtemp = zeros(length(dataX.trial),length(dataX.time{1}));
        for ii = 1:length(dataX.trial)
            XXtemp(ii,:) = mean(dataX.trial{ii}(:,:),1);        % average over trials 
        end
        figure(),plot(dataX.time{1},XXtemp');
        xlabel('Time'); ylabel('Amp');
        box off
        title(sprintf('S0%d Before ICA, Day %d, Block %d',subj,dayInp,s));
        saveppt(sprintf('Prepro S0%d',subj));
        
        %------------------------------------------------------------------
        % POWER SPECTRUM DENSITY PLOTS
        figure()
        pwelch(dataX.trial{1},300,150,512,150);
        title(sprintf('S0%d PSD, Day %d, Block %d',subj,dayInp,s));
        saveppt(sprintf('Prepro S0%d',subj));
        close all
        
        %------------------------------------------------------------------
        % ICA ARTIFACT REJECTION
        cfg            = [];
        cfg.method = 'runica';
        switch ARG.Remove
            case {'fast'}                                   % do PCA to speed up process if searching only for frontal topos          
                cfg.runica.pca = 30;
        end
        cfg.runica.maxsteps = 130;
        cfg.channel = labels;                               % this makes sure only eeg channels are used
        comp           = ft_componentanalysis(cfg, dataX);
        close all
        
        %------------------------------------------------------------------
        % DISPLAY COMPONENTS
        figure('units','normalized','outerposition',[0 0 1 1])
        cfg = [];
        cfg.component = [1: length(comp.label)];            % specify the component(s) that should be plotted
        cfg.layout    = 'biosemi64.lay';                    % specify the layout file that should be used for plotting
        cfg.comment   = 'no';
        ft_topoplotIC(cfg, comp)
        
        %------------------------------------------------------------------
        % COMPUTE SUPPORT OF EACH COMPONENT
        s3 = comp.topo;
        s3 = s3-repmat(min(s3,[],1),[64,1]);
        s3 = s3./repmat(max(s3,[],1),[64,1]);
        s3= sum(s3);
        ARG.ICA.support = s3;
        
        %------------------------------------------------------------------
        % AUTOMATIC COMPONENT SELECTION BASED ON TOPOGRAPHY
        Artif_temp = eegck_artiftemp_biosemi();
        art_corr = [];
        for l=1:size(comp.topo,2)
            for k=1:size(Artif_temp,2)
                art_corr(l,k) = corr(comp.topo(:,l),Artif_temp(:,k));
            end
        end
        remove_eye = find( max(art_corr(:,(1:3)),[],2)>ARG.ICA_artif_thr);
        remove_all = find(max(art_corr(:,(4:end)),[],2)>ARG.ICA_artif_thr);
        ARG.ICA.art_corr = art_corr;
        
        %------------------------------------------------------------------
        % POWER SPECTRUM COMPONENTS
        % compute power spectra of components and removme components with a low
        % ratio between low-frequency power and high frequency power
        switch ARG.Remove
            case {'all'}
                [Power,fax,ratio] = eegck_componentPSD(comp);
                remove_power = find(ratio<ARG.ICA_spectrum_thr);
                good = find(ratio>ARG.ICA_spectrum_thr);
                % check those with strange power and remove if
                % correlations with templates
                remove2 =  find(max(abs(art_corr(remove_power,:)),[],2)>0.5);
                remove_all = [remove_all;remove_power(remove2)];
                ARG.ICA.ratio = ratio;
        end    
        
        %------------------------------------------------------------------
        % EYE MOVEMENT/COMPONENT CORRELATIONS
        % compute correlations with eye movements and suggest components absed
        % on this
        switch ARG.Remove
            case {'all'}
                EyeAnalysis =  eegck_analyseEOG(ft_selectdata(data_eog, 'channel',{'VEOG','HEOG','REOG'}),0);
                M = eegck_trials2mat(comp);
                EyeAnalysis.Saccade  = EyeAnalysis.Saccade';
                EyeAnalysis.Movement  = EyeAnalysis.Movement';
                CC=[];
                for e=1:size(M,1)
                    tmp = (sq(M(e,:,:)));
                    tmp = ck_filt(tmp,dataX.fsample,[30,80],'band',6);
                    tmp = abs(ck_filt_hilbert(tmp))';
                    CC(e,1) = corr(tmp(:),EyeAnalysis.Saccade(:));
                    CC(e,2) = corr(tmp(:),EyeAnalysis.Movement(:));
                end
                ARG.ICA.all_eyecorr = CC;
                thr = std(CC(:));
                CC = max(abs(CC),[],2);
                remove_eye_corr = find((CC>ARG.ICA_eyecorr));
                ARG.ICA.remove_eye_corr = remove_eye_corr;
                EyeAnalysis=[];
                M=[];
        end
        
        %------------------------------------------------------------------
        % SUGGESTED COMPONENTS TO REMOVE
        switch ARG.Remove
            case {'fast'}
                % remove frontal topos only - automatic process
                remove = remove_eye;
                ARG.ICA.not_removed=[];
            case {'all'}
                % remove based on topo
                remove = [remove_eye;remove_all;remove_eye_corr];
                % find those with seem to have strange power but are not rejected and display
                ARG.ICA.not_removed  = setdiff(remove_power,remove);
        end
        remove = unique(remove);
        remove = sort(remove);
        fprintf('Suggested components to remove:  \n');
        for l=1:length(remove)
            fprintf('%d \n ',remove(l));
        end
        fprintf('\n');
        fprintf('%d \n', size(remove,1));
        
        %------------------------------------------------------------------
        % DISPLAY COMPONENTS
        figure(1);
        chld = get(gcf,'Children');
        ncomp = length(comp.label);
        for l=1:length(remove)
            set(get(chld(ncomp-remove(l)+1),'Title'),'Color',[1 0 0 ])
        end
        % red = removed
        for l=1:length(ARG.ICA.not_removed)
            set(get(chld(ncomp-ARG.ICA.not_removed(l)+1),'Title'),'Color',[0 1 0 ])
        end
        % green = not removed
        % black = not chosen
        saveppt(sprintf('Prepro S0%d',subj));
        close all
        
        %------------------------------------------------------------------
        % DISPLAY SPECTRA AND COMPONENTS
        % display spectra and components that may be conspicous but were not removed
        switch ARG.Remove
            case {'all'}
                
                figure(4);
                set(gcf,'Position',[ 0 0        1484        1000]);
                n = numel( ARG.ICA.not_removed)*2;
                nyplot = ceil(sqrt(n));  nxplot = ceil(n./nyplot);
                load('FtDummy');
                eegck_topcfg;
                for k=1:length( ARG.ICA.not_removed)
                    subplot(nxplot, nyplot,k*2-1); hold on
                    plot(fax,(log(Power(good,:))),'k');
                    plot(fax,log(Power( ARG.ICA.not_removed(k),:)),'r','LineWidth',2);
                    axis([fax(1) fax(end) min(log(Power(:))) max(log(Power(:)))]);
                    title(sprintf('component %d', ARG.ICA.not_removed(k)));
                    subplot(nxplot, nyplot,k*2);
                    EvpDummy.avg = comp.topo(:, ARG.ICA.not_removed(k));
                    ft_topoplotER(cfg_topo,EvpDummy);
                end
                suptitle(sprintf('S0%d, Day %d, Block %d',subj,dayInp,s))
                pause(1);
                saveppt(sprintf('Prepro S0%d',subj));
        end
        close all
        
        %------------------------------------------------------------------
        % REMOVE COMPONENTS
        cfg = [];
        cfg.component = remove;
        dataX = ft_rejectcomponent(cfg, comp);
        ARG.ICA.removed_components = remove;
        
        % just preserve the data before removing trials;
        dataTemp = dataX;
        
        
        %------------------------------------------------------------------
        % SUBJ 20 HAS SOME BAD CHANNELS ON DAY 1 - REMOVE THEM 
        if subj ==20 && kn ==1;
            fprintf('Removing bad channels.... \n'); 
            dataTemp = dataX;
            
            badchannels{1} = {'FT7','P9','P6','TP8','F7'};  % D1 B1
            badchannels{2} = {'FT7','TP8'};                 % D1 B2
            
            cfg = [];
            cfg.method = 'nearest';
            cfg.badchannel = badchannels{s};
            cfg.layout = 'biosemi64.lay';
            cfg.trials  = 'all';
            cfg.sens = 'biosemi64.lay';
            cfg_neighb = eegck_BiosemiNbrs(dataX);
            cfg.neighbours = cfg_neighb;
            [interp] = ft_channelrepair(cfg, dataX);
            dataX = interp; % interpolated data 
            fprintf('Removed Bad Channels \n'); 
            
        end
        
        
        %------------------------------------------------------------------
        % REMOVE TRIALS WITH EXCESSIVE AMPLITUDE
        out = eegck_maxampft(dataX,(1:64));
        cfg = [];
        cfg.trials = find(sum(abs(out)>120,1)==0);
        dataX = ft_redefinetrial(cfg, dataX);
        data_eog = ft_redefinetrial(cfg,data_eog);
        
        %------------------------------------------------------------------
        % APPEND EOG AND EEG DATA
        dataX = ft_appenddata([], dataX, data_eog);
        
        %------------------------------------------------------------------
        % REPLOT AFTER ICA
        XXtemp = zeros(length(dataX.trial),length(dataX.time{1}));
        for ii = 1:length(dataX.trial)
            XXtemp(ii,:) = mean(dataX.trial{ii}(:,:),1);
        end
        figure(),plot(dataX.time{1},XXtemp');
        xlabel('Time'); ylabel('Amp');
        box off
        title(sprintf('S0%d After ICA, Day %d, Block %d',subj,dayInp,s));
        saveppt(sprintf('Prepro S0%d',subj));
        close all
        
        %------------------------------------------------------------------
        % save data
        sname = sprintf('PreproICA_S0%d_D%d_B%d.mat',subj,dayInp,s);
        save(sname,'dataX','A','ARG');
        
        %------------------------------------------------------------------
        % EOG ANALYSIS
        EyeAnalysis =  eegck_analyseEOG(ft_selectdata(dataX, 'channel',{'VEOG','HEOG','REOG'}),0);
        j = find( (dataX.time{1}>=-0.5).*(dataX.time{1}<=1.2));             % select time region of interest – adapdt your paradigm e,g, -0.5 to 1.2
        MaskM =EyeAnalysis.Movement(:,j);                                   % sum over time
        MaskS =EyeAnalysis.Saccade(:,j);
        EyeAnalysis=[];
        
        % two thresholds to select trials based on number of bad time points
        ARG.EOGS = 11;
        ARG.EOGM = 20;
        
        % clean signal
        cfg = [];
        cfg.trials = find((sum(MaskM,2)<ARG.EOGM).*(sum(MaskS,2)<ARG.EOGS));
        dataX = ft_redefinetrial(cfg, dataX);
        ARG.EOGremove = cfg; 

        newName = sprintf('PreproICA_S0%d_D%d_B%d_EOG_out.mat',subj,dayInp,s);
        save(newName,'dataX','ARG','A');
        fprintf('Done:  Preprocessing Block %d Day %d \n',s,kn)
        
        %------------------------------------------------------------------
        % RECORD COMPONENTS REMOVED 
        fprintf(fileID,'Subject %d, Day %d, nBlock %d \n \n',subj,dayInp,s);
        fprintf(fileID,'Trials Removed: %1.0f \n',size(data.trial,2) - size(cfg.trials,2));
        fprintf(fileID,'Removed Components: \n');
        fprintf(fileID,'%1.0f \n',size(ARG.ICA.removed_components,1));
        fprintf(fileID,'%s',num2str(ARG.ICA.removed_components'));
        fprintf(fileID, '\n \n');
        
    end

end

fclose(fileID);
clearvars -except subj subs sub_list
fprintf('Done Preprocessing S0%d \n',subj)

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DISPLAY NOISE CLEANING FOR EACH BLOCK - INDIVIDUAL PARTICIPANTS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

dataList = dir('PreproICA_S*_EOG_out.mat');
S = 1;
c=1;co=1;
nfiles = length(dataList);

% Get data from the subject from the preprocessing file 
for file=1:nfiles
    sname = dataList(file).name;
    Data = load(sname,'ARG'); 
    Stats_sup(co,:) = Data.ARG.ICA.support;
    Stats_sup2(co) = median(Data.ARG.ICA.support(Data.ARG.ICA.removed_components));
    Stats_thr(co,:,:) = Data.ARG.ICA.art_corr;
    Stats_ratio(co,:) = Data.ARG.ICA.ratio;
    Stats_eyec(co,:,:) = Data.ARG.ICA.all_eyecorr;
    Stats_remove(co,:) = [length(Data.ARG.ICA.remove_eye_corr),length(Data.ARG.ICA.removed_components)];
    Sub_list(co,:) = [S,file];
    co=co+1;
end

cvec = 'rgbkmcrgbkmcrgbkmcrgbkmcrgbkmcrgbkmc';

%--------------------------------------------------------------------------
% PLOT RESULTS 
N = size(dataList,1);
L =  (1 :length(dataList)); % axis length
figure('units','normalized','outerposition',[0 0 0.7 0.8]) 

% correlation with templates
subplot(2,2,1); hold on;
for k=1:N
    tmp = max(abs(sq(Stats_thr(k,:,:))),[],2);
    plot(k,tmp,[cvec(Sub_list(k,2)) 'o'])
end
title('Template correlation');
xlim([0 length(Sub_list)+1])
set(gca,'XTick',L,'XTickLabel',Sub_list(L,2),'TickDir','out')
line([0 N+1],[Data.ARG.ICA_artif_thr Data.ARG.ICA_artif_thr])
xlabel('block')

% Power ratio
subplot(2,2,2); hold on;
for k=1:N
    tmp =Stats_ratio(k,:);
    plot(k,tmp,[cvec(Sub_list(k,2)) 'o'])
end
title('Power ratio')
axis([0 N+1 0 30]);                                                 % we care most abotu those with low ratio
line([0 N+1],[Data.ARG.ICA_spectrum_thr Data.ARG.ICA_spectrum_thr])
set(gca,'XTick',L,'XTickLabel',Sub_list(L,2),'TickDir','out')
xlabel('block')

% Eye Correlation
subplot(2,2,3); hold on;
for k=1:N
    tmp =Stats_eyec(k,:);
    plot(k,tmp,[cvec(Sub_list(k,2)) 'o'])
end
title('Eye movement correlation')
xlim([0 length(Sub_list)+1])
line([0 N+1],[Data.ARG.ICA_eyecorr Data.ARG.ICA_eyecorr])
set(gca,'XTick',L,'XTickLabel',Sub_list(L,2),'TickDir','out')
xlabel('block')

% Number of removed components 
subplot(2,2,4); hold on;
for k=1:N
    plot(k,Stats_remove(k,1),[cvec(Sub_list(k,1)) 'o'])
    plot(k,Stats_remove(k,2),[cvec(Sub_list(k,1)) 's'])
end
plot(Stats_remove(:,1),'k');
plot(Stats_remove(:,2),'r');
xlim([0 length(Sub_list)+1])
title('# removed components (total/eyeC)');
set(gca,'XTick',L,'XTickLabel',Sub_list(L,2),'TickDir','out')
xlabel('block');

suptitle(sprintf('S0%d',subj));
saveppt(sprintf('Prepro S0%d',subj));
close all



%--------------------------------------------------------------------------
% E0G GRAPHS FOR DIFFERENT CONDITIONS (STILL INDIVIDUAL SUBJECTS)
conditions1 = {'Aud', 'Aud', 'VH','AVH','AVH','AVL'};
conditions2 = {'VH','VL','VL','AVL','VH','VL'};

cond1 = [3 3 1 4 4 7];
cond2 = [1 2 2 7 1 2];

% LOOP THROUGH CONDITIONS 
for c2 = 1:length(cond1)
    
    cond = [cond1(c2),cond2(c2)];
    conditions = {conditions1{c2},conditions2{c2}};
    
    for k = 1:length(dataList)
        load(dataList(k).name)
        
        %------------------------------------------------------------------
        % Detect EOG Events
        EyeAnalysis =  eegck_analyseEOG(ft_selectdata(dataX, 'channel',{'VEOG','HEOG','REOG'}));
        m = EyeAnalysis.Movement;
        s = EyeAnalysis.Saccade;
        
        % EOG for each condition 
        EOG_Stats = zeros(2,2,length(dataX.trial{1}));
        for c=1:2
            trls = find(dataX.trialinfo(:,3)==(cond(c)));
            EOG_Stats(1,c,:) = mean(m(trls,:));
            EOG_Stats(2,c,:) = mean(s(trls,:));
        end
        
        j = find( (dataX.time{1}>=-0.5).*(dataX.time{1}<=1.2));
        tax = dataX.time{1};
        % filename = files(k).name
        % save(filename,'dataX','ARG','A','EOG_Stats','tax');
        
        % VEOG
        close all
        figure('units','normalized','outerposition',[0 0 0.8 0.7])
        subplot(2,2,1)
        plot(tax,sq(mean(EOG_Stats(1,:,:),2)),'k');
        a = max(sq(mean(EOG_Stats(1,:,:),2)));
        title('VEOG');
        box off
        
        % REOG
        subplot(2,2,2)
        plot(tax,sq(mean(EOG_Stats(2,:,:),2)),'k');
        b = max(sq(mean(EOG_Stats(2,:,:),2)));
        title('REOG');
        box off
        
        % VEOG comparing the two 
        subplot(2,2,3)
        plot(tax,sq(EOG_Stats(1,1,:)));
        c = max(sq(EOG_Stats(1,1,:)));
        hold on
        plot(tax,sq(EOG_Stats(1,2,:)),'r');
        d = max(sq(EOG_Stats(1,2,:)));
        title('VEOG comp');
        legend({sprintf('%s',conditions{1}),sprintf('%s',conditions{2})},'Location','NorthWest')
        box off
        
        % REOG comparing the two. 
        subplot(2,2,4)
        plot(tax,sq(EOG_Stats(2,1,:)));
        e = max(sq(EOG_Stats(2,1,:)));
        hold on
        plot(tax,sq(EOG_Stats(2,2,:)),'r');
        f = max(sq(EOG_Stats(2,2,:)));
        title('REOG comp');
        legend({sprintf('%s',conditions{1}),sprintf('%s',conditions{2})},'Location','NorthWest')
        box off
        
        suptitle(sprintf('S0%d Block %d %s vs %s',subj, k, conditions{1}, conditions{2}))
        saveppt('EOG')
        close all
        
        E0G_Group(k,:,:,:) = EOG_Stats;
        
    end
    
    
    %----------------------------------------------------------------------
    % INDIVIUDAL FIGURES, ACROSS BLOCKS (individual blocks) 
    
    figure('units','normalized','outerposition',[0 0 0.8 0.7])

    % VEOG
    subplot(2,2,1)
    plot(tax,sq(mean(E0G_Group(:,1,:,:),3)),'k');
    title('VEOG');
    box off
    
    % REOG
    subplot(2,2,2)
    plot(tax,sq(mean(E0G_Group(:,2,:,:),3)),'k');
    title('REOG');
    box off
    
    % COMP 
    subplot(2,2,3)
    m = sq(mean(E0G_Group(:,1,:,:),1));
    s = sq(sem(E0G_Group(:,1,:,:),1));
    errorbar(tax,m(1,:),s(1,:),'r')
    hold on
    errorbar(tax,m(2,:),s(2,:),'b')
    title(sprintf('%s vs %s VEOG',conditions{1},conditions{2}));
    legend({sprintf('%s',conditions{1}),sprintf('%s',conditions{2})},'Location','NorthWest')
    box off
    
    % COMP REOG
    subplot(2,2,4)
    m = sq(mean(E0G_Group(:,2,:,:),1));
    s = sq(sem(E0G_Group(:,2,:,:),1));
    errorbar(tax,m(1,:),s(1,:),'r')
    hold on
    errorbar(tax,m(2,:),s(2,:),'b')
    title(sprintf('%s vs %s REOG',conditions{1},conditions{2}));
    box off
    suptitle(sprintf('S0%d All Blocks %s vs %s',subj, conditions{1},conditions{2}))
    saveppt('EOG')
    
    
    close all
    clc
end

fprintf('Done Noise Cleaning/EOG S0%d \n',subj)





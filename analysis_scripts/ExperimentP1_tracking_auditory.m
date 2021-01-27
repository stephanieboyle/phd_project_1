%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Steph B - PILOT AUDITORY EXPERIMENT
% - Presents 2 different rates of flickering auditory sounds (8 clicks.900ms,
%   14 clicks.900ms)
% - ONLY HAS AUDITORY TRIALS - runs for 60 trials.
% - If int = 0, no background auditory noise.
% - Presents visual noise screen with fix cross.
% - Gives some results on performance at end.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear
close all
load allTrials7; % stimuli (event rates 8:14)
load comp7; % comparison stimuli (event rate 11)
rand('seed', sum(100 * clock)); % seed random number generator
cl = clock;

%==========================================================================
% TRIALS
% aRel = 0.001:0.010:0.06;  % set the different reliabilities
int = 0;
% cond(:,1) = 1:length(aRel); % set how many reliability levels you are testing
cond(:,1) = 1;
rateLow = 1; % set the lower comparison rate
rateHigh = 7; % set the higher comparison rate
trials = 30; % set the number of trials you want to test
allT = (trials*length(cond))*2; % all trials
allFrames = 75; % frame rate
all = zeros(allT,8); % create the matrix

% create all trials:
% all(:,1) = repmat(aRel',(trials*2),1);  % put the reliability levels in
all(1:(length(cond)*trials),1) = rateLow; % put the low rates in
all(((length(cond)*trials)+1):end,1) = rateHigh; % put the high rates in
all(:,2) = repmat(cond,(trials*2),1); % put the conditions in
%==========================================================================


%==========================================================================
% STIMULI
% take some random stimuli out for testing
indAud = datasample(1:1000,length(all))'; % get a random list of stimuli to take
all(:,3) = indAud;
indAudComp = datasample(1:7000,length(all))'; % get a random list of comparison stimuli to take
all(:,4) = indAudComp;

% shuffle
ind2 = randperm(length(all));
all = all(ind2,:); % shuffled order

% make your experimental stimuli in one matrix:
trackAudT = zeros(length(all),allFrames);
for k = 1:length(all);
    trackAudT(k,:) = allTrials(all(k,3),:,(all(k,1))); % this takes the random stimuli, all rows of it, from either rate or 5 depending on what rPerm(j) is for that trial
end

% make your comparison stimuli
trackAudComp = zeros(length(all),allFrames);
for k = 1:length(all);
    trackAudComp(k,:) = allComp(all(k,4),:); % no 3rd dim rate - all 11.sec
end

% check:
for k = 1:length(all)
    temp = length(find(trackAudT(k,:)==2));
    all(k,5) = temp;
    temp2 = length(find(trackAudComp(k,:)==2));
    all(k,6)  = temp2;
end

% Program which stream has more/less for later:
for k = 1:length(all)
    if all(k,1) == rateLow % if rate is 3 (10 flashes.sec)
        all(k,7) = 2; % then the correct response is second stream more
    elseif all(k,1) == rateHigh; % if the rate is 5 (12 flashes.sec)
        all(k,7) = 1; % then the correct response is first stream more
    end
end
%==========================================================================


%==========================================================================
% IMAGE
Bgnd = 60;
noiseC = 50;
h = 600; % image height
w = 800; % image width
A = round(rand(h,w)*noiseC)+Bgnd; % noise image

imgE{1} = A; % noise image
% imgE{2} = A; % noise image
imgC{1} = A; % noise image
imgC{2} = A; % noise  image
%==========================================================================


%==========================================================================
% SOUND
load('new.mat'); % NEW.MAT CONTAINS A 12MS CLICK SOUND
Rate = 44100; % new rate upstairs in the lab
pauses(1) = 11.965; % ms pause
pauses = pauses/1000; % convert to seconds
Tone =[]; % start with nothing
silent = zeros(round(Rate*pauses),1);
snt = (length(silent)-1);
d = length(silent);
c= 1;
%==========================================================================


%==========================================================================
% RELIABILITIES

% for k = 1:length(aRel)
%     int(k) = aRel(k);
% end
% Sounds will have to be made in the actual experiment later on as they
% change from trial to trial.
%==========================================================================

clear allComp allTrials ind2 A colPos h indAud indAudComp ratesT rPerm rowPos trialUse  ;


%==========================================================================
% FIXATION CROSS
%  set the size of the arms of our fixation cross
fixCrossDimPix = 10;
%  set the coordinates (these are all relative to zero we will let
% the drawing routine center the cross in the center of our monitor
xCoords = [-fixCrossDimPix fixCrossDimPix 0 0];
yCoords = [0 0 -fixCrossDimPix fixCrossDimPix];
allCoords = [xCoords; yCoords];
% Set the line width for our fixation cross
lineWidthPix = 2;
%==========================================================================

% ask for input.
Subj = input('Subject: ','s'); % enter subject number
Block = input('Block: ');
Day = input('Day: ');

% save: Kayserlab folder computer
sname = sprintf('C:/Kayserlab/Stephanie B/Project1Lab/log/TrackAud_%s_B%d_D%d_%02d%02d_%02d%02d.mat',Subj,Block,Day,cl(2),cl(3),cl(4),cl(5));

% define Keyboard:
leftKey = KbName('left');
rightKey = KbName('right');
counter = 1; % need this for feedback later
timeStim = zeros(length(all),2); % record times
trialUse = ones(1,allFrames); % for visual

%--------------------------------------------------------------------------
% SCREEN STUFF
AssertOpenGL;
screenNumber = max(Screen('Screens'));
white = WhiteIndex(screenNumber); black = BlackIndex(screenNumber); grey = white / 2; % Define black, white and grey for font
[window, windowRect] = Screen('OpenWindow', screenNumber, Bgnd); % fullscreen
Screen('Flip', window); % Flip to clear
[screenXpixels, screenYpixels] = Screen('WindowSize', window); % Get the size of the on screen window
x=screenXpixels; % define screen length x
y=screenYpixels; % define screen length y

% Screen('TextSize', window, 16); % Set the text size
ifi = Screen('GetFlipInterval', window); % Query the frame duration
[xCenter, yCenter] = RectCenter(windowRect); % Get the centre coordinate of the window
topPriorityLevel = MaxPriority(window); % Query the maximum priority level
Screen('BlendFunction', window, 'GL_SRC_ALPHA', 'GL_ONE_MINUS_SRC_ALPHA'); % Set up alpha-blending for smooth (anti-aliased) lines
Screen('TextFont',window, 'Courier New');
Screen('TextSize',window, 19);
vbl1 = Screen('Flip', window); % gets the flip interval
%--------------------------------------------------------------------------


%--------------------------------------------------------------------------
% set up PsychPortAudio
% Rate = 22050;
Rate = 44100;
InitializePsychSound(1); %force low latency
PsychPortAudio('Verbosity', 10);
deviceid=-1; %default
mode=1; %playback only
reqlatencyclass=2; % Request latency mode 2, which used to be the best one in our measurement:
buffersize = 0;     % Pointless to set this. Auto-selected to be optimal.
suggestedLatencySecs = [];
channels=2;
%%% open audio device
pahandle = PsychPortAudio('Open', deviceid, mode, reqlatencyclass, Rate, channels, buffersize, suggestedLatencySecs);
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
% MAKE THE TEXTURES:
texE{1} =Screen('MakeTexture', window, imgE{1});
%--------------------------------------------------------------------------


%==========================================================================
% EXPERIMENT
%==========================================================================
HideCursor
intC = int;

for k = 1:allT;
    
    %     if all(j,2) ==1
    %         int = aRel(1); % highest reliability/lowest noise
    %
    %     elseif all(j,2) ==2
    %         int = aRel(2);
    %
    %     elseif all(j,2) ==3
    %         int = aRel(3);
    %
    %     elseif all(j,2) ==4
    %         int = aRel(4);
    %
    %     elseif all(j,2) ==5
    %         int = aRel(5);
    %
    %     elseif all(j,2) ==6
    %         int = aRel(6); % lowest reliability/highest noise
    %
    %     end
    
    %----------------------------------------------------------------------
    Tone = [];
    ToneC = [];
    silent = zeros(round(Rate*pauses),1);
    snt = (length(silent)-1);
    d = length(silent);
    
    c= 1; % COUNTER
    % EXP TONE
    for s = 1:allFrames
        if trackAudT(k,s) == 1
            Tone(c:c+snt,1) = silent;
            c= c+d;
        elseif trackAudT(k,s)==2
            %             Tone(c:c+snt,1) = click;
            Tone(c:c+snt,1) = new;
            c = c+d;
        end
    end
    
    clear c
    c = 1; % RESET COUNTER
    
    % COMP TONE
    for s = 1:allFrames
        if trackAudComp(k,s) ==1
            ToneC(c:c+snt,1) = silent;
            c = c+d;
        elseif trackAudComp(k,s) ==2
            %             ToneC(c:c+snt,1) = click;
            ToneC(c:c+snt,1) = new;
            c = c+d;
        end
    end
    
    Tone = Tone+randn(size(Tone))*int;
    ToneC = ToneC+randn(size(ToneC))*intC;
    
    % convert to Stereo sound
    Tone(:,2) = Tone;
    ToneC(:,2) = ToneC;
    %----------------------------------------------------------------------
    
    %----------------------------------------------------------------------
    % START EXPERIMENT:
    if k==1
        DrawFormattedText(window, 'Press Any Key To Begin', 'center', 'center', white);
        Screen('Flip', window);
        KbStrokeWait;
        pause(1);
    end
    %----------------------------------------------------------------------
    
    %----------------------------------------------------------------------
    % FIXATION CROSS
    Screen('DrawTexture', window, texE{1});
    Screen('DrawLines', window, allCoords, lineWidthPix, white, [xCenter yCenter], 1);
    Screen('Flip', window);
    %----------------------------------------------------------------------
    
    %----------------------------------------------------------------------
    % PAUSE AND LOAD SOUND AT SAME TIME BEFORE NEXT TRIAL
    % load sound into memory - do at trial start
    PsychPortAudio('FillBuffer', pahandle, Tone'); % channelsxtime
    % this sets upt he sound - but does not play
    PsychPortAudio('Start', pahandle, 1, inf, 0);
    pause(rand*0.3); % pause
    %----------------------------------------------------------------------
    
    
    %----------------------------------------------------------------------
    % EXPERIMENTAL STREAM
    %---------------------------------------------------------------------
    % start playing sound
    StartTime=PsychPortAudio('RescheduleStart', pahandle, 0, 1); soundStart = tic; % Start playback and collect start time estimate
    
    % visual bit
    %     for s = 1:allFrames
    Screen('DrawTexture', window, texE{1}); % just show nothing visually.
    [vblact,StimulusOnsetTime,FlipTimeStamp] = Screen('Flip', window);
    %     end
    % stops playing sound - but wait for sound to finish
    PsychPortAudio('Stop', pahandle, 1);timeStim(k,1) = toc(soundStart);
    %----------------------------------------------------------------------
    
    
    %----------------------------------------------------------------------
    % PAUSE & LOAD SOUND FOR NEXT TRIAL:
    pause(0.1)
    PsychPortAudio('FillBuffer', pahandle, ToneC'); % channelsxtime
    % this sets upt he sound - but does not play
    PsychPortAudio('Start', pahandle, 1, inf, 0);
    pause(0.3+rand*0.3); % pause between 300-700ms
    %----------------------------------------------------------------------
    
    
    %----------------------------------------------------------------------
    % COMPARISON STREAM
    %----------------------------------------------------------------------
    % start playing sound
    StartTime=PsychPortAudio('RescheduleStart', pahandle, 0, 1);    soundStart = tic; % Start playback and collect start time estimate
    Screen('DrawTexture', window, texE{1}); % <- with a different texture for each k
    [vblact,StimulusOnsetTime,FlipTimeStamp] = Screen('Flip', window);
    % stops playing - but wait for sound to finish
    PsychPortAudio('Stop', pahandle, 1); timeStim(k,2) = toc(soundStart); % save total sound time
    %----------------------------------------------------------------------
    
    %----------------------------------------------------------------------
    pause(0.2);
    %----------------------------------------------------------------------
    
    %----------------------------------------------------------------------
    % RESPONSE
    Screen('DrawTexture', window, texE{1});
    DrawFormattedText(window, 'Which stream had more clicks?','center', white);
    DrawFormattedText(window, ' first stream [left]       second stream [right] ', 'center', 'center', white);
    Screen('Flip', window);
    
    respToBeMade = true;
    while respToBeMade
        [keyIsDown,secs, keyCode] = KbCheck;
        if keyCode(leftKey)
            resp = 1; % first stream had more
            respToBeMade = false;
        elseif keyCode(rightKey)
            resp = 2; % second stream had more
            respToBeMade = false;
        end
    end
    
    all(k,8) = resp;
    %----------------------------------------------------------------------
    
    %----------------------------------------------------------------------
    % FEEDBACK
    if all(k,7)== all(k,8);
        Screen('DrawTexture', window, texE{1});
        DrawFormattedText(window, 'Correct :) ','center','center', white);
        Screen('Flip', window);
        WaitSecs(rand+0.2);
    elseif all(k,7)~= all(k,8);
        Screen('DrawTexture', window, texE{1});
        DrawFormattedText(window, 'Incorrect :( ','center','center',white);
        Screen('Flip', window);
        WaitSecs(rand+0.2);
    end
    %---------------------------------------------------------------------
    
    %---------------------------------------------------------------------
    % BREAKS
    if k == counter+30;
        Screen('DrawTexture', window, texE{1});
        DrawFormattedText(window,'Break...', 'center', 200, white);
        DrawFormattedText(window, 'Press Any Key To Begin Again', 'center', 'center', white);
        Screen('Flip', window);
        KbStrokeWait;
        pause(1);
        counter = counter+30;
    end
    %----------------------------------------------------------------------
    
    %------------------------------------------------------------------------
    % FIXATION CROSS
    Screen('DrawTexture', window, texE{1});
    Screen('DrawLines', window, allCoords, lineWidthPix, white, [xCenter yCenter], 1);
    Screen('Flip', window);
    %------------------------------------------------------------------------
    
    
    save(sname,'all');
    clear Tone ToneC
    pause(1);
end


Screen('CloseAll')
PsychPortAudio('CloseAll')
ShowCursor

% ALL table:
% [:,1] = Rate (3 or 5)
% [:,2] = condition (1:6)
% [:,3] = original matrix row
% [:,4] = original comparison matrix row
% [:,5] = correct response
% [:,6] = actual response


%--------------------------------------------------------------------------
% % Results
% trackResultsA = proportion correct [all data, rate3, rate5]

trackResultsA(1,1) = length(find(all(:,7)== all(:,8)))/allT;

cInd = find(all(:,1)==rateLow);
data1 = all(cInd,:);
trackResultsA(1,2) = length(find(data1(:,7)== data1(:,8)))/trials;

cInd2 = find(all(:,1)==rateHigh);
data2 = all(cInd2,:);
trackResultsA(1,3) = length(find(data2(:,7)== data2(:,8)))/trials;

disp(trackResultsA(1))

save(sname,'trackResultsA', 'all', 'timeStim','int','data1','data2');


%% analysis for >1 auditory reliability
% trackResultsA = zeros(length(aRel),3,3); % make the response table
%
% for k = 1:3
% trackResultsA(:,1,k) = aRel'; % fill in the first row of each with the reliabilities
% end
%
% % make the tables for storing results for fitting
% data = zeros(trials*2,size(all,2),length(aRel));
% data2 = zeros(trials,size(all,2),length(aRel));
% data3 = zeros(trials,size(all,2),length(aRel));
%
%
% % fill separate out the results by reliability level
% for m = 1:length(aRel)
%     cInd = find(all(:,2)==m); % for each rel level, separate out the data
%     data(:,:,m) = all(cInd,:); % (:,:,1) = first rel level, (:,:,2) = second rel level... etc.
%
%     % separate out by rel and rate
%     cInd2 = find(all(:,2)==m & all(:,1)==3); % for each rel level, for RATE 3, separate out the data
%     data2(:,:,m) = all(cInd2,:);
%
%     cInd3 = find(all(:,2)==m & all(:,1)==5); % for each rel level, for RATE 5, separate out the data
%     data3(:,:,m) = all(cInd3,:);
%
%     trackResultsA(m,2,1)= (length(find(data(:,5,m)==data(:,6,m))))/(trials*2); % rates 3 and 5 together
%     trackResultsA(m,2,2)= (length(find(data2(:,5,m)==data2(:,6,m))))/(trials); % rate 3
%     trackResultsA(m,2,3)= (length(find(data3(:,5,m)==data3(:,6,m))))/(trials); % rate 5
%
%
% end
%
%
% % add in the number of trials
% trackResultsA(:,3,1) = repmat(trials*2,length(aRel),1); % (:,:,1) = all rates together
% trackResultsA(:,3,2) = repmat(trials,length(aRel),1); %
% trackResultsA(:,3,3) = repmat(trials,length(aRel),1);
%
%
% save(sname,'trackResultsA', 'all', 'timeStim');
%
%
% % trackResultsA (:,:,1)  = results for rates 3 and 5 combined
% % trackResultsA (1:6,1,1) = the 6 reliability levels
% % trackResultsA (1:6,2,1) = the results for each reliability level
% % trackResultsA (1:6,3,1) = number of trials
%
% % trackResultsA (:,:,2) and (:,:,3) = stores the same info as above,
% % but for rate 3 (:,:,2)  and rate 5 (:,:,3) separately.
%
%
% % ALL RATES FIGURE:
% figure; plotpd(trackResultsA(:,:,1),'color','black');
% hold on
% % add psychometric curve
% shape = 'cumulative Gaussian';
% n_intervals = 2;
% prefs = batch('shape',shape,'n_intervals',n_intervals,'runs', 999);
% outputPrefs = batch('write_pa','pa','write_th','th');
% T1 = psignifit(trackResultsA(:,:,1), [prefs outputPrefs]);
% plotpf(shape,pa.est,'color','red');
% % drawHeights = psi(shape, pa.est, th.est);
% % line(th.lims, ones(size(th.lims,1), 1) * drawHeights, 'color', 'red')
% saveas(gcf,sprintf('trackAud_%s',Subj), 'fig')
% saveas(gcf,sprintf('trackAud_%s',Subj), 'tiff')
%
%
% % RATES SEPARATED FIGURE:
% figure;
% for k = 2:3
%     if k == 2
%         color = [0 0 1] % blue
%     elseif k == 3
%         color = [1 0 0] % red
%     end
%
%     % BLUE = RATE 3 (10 flashes per second)
%     % RED = RATE 5 (12 flashes per second)
%
%     plotpd(trackResultsA(:,:,k),'color',color)
% %     plotpd(trackResultsA(:,:,2),'color','blue')
% %     plotpd(trackResultsA(:,:,3),'color','red')
%     hold on
%     % add psychometric curve
%     shape = 'cumulative Gaussian';
%     n_intervals = 2;
%     prefs = batch('shape',shape,'n_intervals',n_intervals,'runs', 999);
%     outputPrefs = batch('write_pa','pa','write_th','th');
%     T1 = psignifit(trackResultsA(:,:,k), [prefs outputPrefs]);
%     plotpf(shape,pa.est,'color',color);
%
% %     drawHeights = psi(shape, pa.est, th.est);
% %     line(th.lims, ones(size(th.lims,1), 1) * drawHeights, 'color', color)
%     hold on
% end
% hold off
%
%
% saveas(gcf,sprintf('trackAudRates_%s',Subj), 'fig')
% saveas(gcf,sprintf('trackAudRates_%s',Subj), 'tiff')

% trackResultsA % display results


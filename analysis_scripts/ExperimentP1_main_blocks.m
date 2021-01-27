%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% STEPH B RELIABILITY EXPERIMENT - EEG/BEHAVIOURAL
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Rate discrimination task
% Shows flashing squares or flickering sounds or both together.
% There are:
% visual (2 conditions), auditory (1 condition) and
% multisensory (6 conditions) trials.
% 2 reliability levels for visual (high vs.low contrast square). Auditory
% reliability is always high.
% 3 conflict levels as well - 0/+2/-2. 0 conflict means auditory and visual
% rate are the same, +2 conflict means the visual rate is faster, and -2
% conflict means the auditory rate is faster.
% Total: 51 individual conditions.
% Event rates in this version: there are 7 (which are set to 8:14
% flashes/clicks per 900ms right now).
%
% EXPERIMENT: Presents an experimental stream that has a variable amount of
% flashes/clicks/flashes&clicks and then a comparison stream that always
% has 11 flashes/clicks/flashes&clicks. The task for the participants is to
% say which stream (experimental or comparison stream) has more 'events'.
% RUNS AT SCREEN RESOLUTION OF 800x600 AND MONITOR REFRESH RATE OF 85Hz.
%--------------------------------------------------------------------------


clear
close all
rand('seed', sum(100 * clock));                                             % Random number generator seed
cl = clock;

%==========================================================================
% INPUT
sqHigh = input('Target Intensity High Rel:');                               % set visual square contrast HIGH reliability (get it from tracking_visual exp)
sqLow = input('Target Intensity Low Rel:');                                 % set visual square contrast LOW reliability (get it from tracking_visual exp)
Subj = input('Subject: ','s');                                              % enter subject number
Block = input('Block: ');                                                   % enter block number
Day = input('Day: ');                                                       % enter day
Do_trigger = input('EEG? Yes: 1, No: 0 \n' );                               % Only set if EEG experiment

% % % sname: Kayserlab folder computer
sname = sprintf('C:/Kayserlab/Stephanie B/Project1Lab/log/Exp_%s_B%d_D%d_%02d%02d_%02d%02d.mat',Subj,Block,Day,cl(2),cl(3),cl(4),cl(5));
% %==========================================================================
% sname = sprintf('W:/Project 1/Exp_%s_B%d_D%d_%02d%02d_%02d%02d.mat',Subj,Block,Day,cl(2),cl(3),cl(4),cl(5)); 

%==========================================================================
% CHANGEABLE VARIABLES
trials = 10;                                                                % # trials of each con
blocks = 5;                                                                 % # of blocks 
rates = 7;                                                                  % # of rates 
conditionNumber = 9;                                                        % # of conditions (v/a/ms)
indivConditions = 51;                                                       % # of individual conditions
allFrames = 75;                                                             % number of frames in stimuli
trialsOne = rates*conditionNumber;                                          % number of trials needed to get 1 trial of each condition

infoP = zeros(trialsOne,13);                                                % a matrix to store the info you need (this will get x times bigger later when you repeat the matrix (with x being the number of trials))
int = 0;                                                                    % background intensity of AUDITORY noise (can change this to be less/more)
%==========================================================================


%==========================================================================
% SET UP ALL THE CONDITIONS FOR THIS PARTICIPANT

% CONDITIONS -------------------------------
% there are 7 rates, and the 9 conditions have all 7. So need 7 repeats of
% 1:9
conditions = zeros(rates,conditionNumber);                                  % initialise it
for k = 1:conditionNumber                                                   % conditions 1:9
    conditions(:,k) = ones(rates,1)*k;                                      % gives a 7,9 matrix
end
conditions = conditions(:);                                                 % turns it into a 63,1 matrix. All conditions, 7 times, in 1 column.
infoP(:,2) = conditions;                                                    % store conditions
% starting at 2 so we can use the first column to store the trial number
% later

%--------------------------------------------------------------------------
% RATES
allR = repmat((1:rates)',conditionNumber,1);                                % this gives a 63,1 column with rates listed 1:7,1:7...
infoP(:,3) = allR;                                                          % this stores 1:7 for each of the 9 conditions.
for k = 1:length(infoP)
    if infoP(k,2) ==1 || infoP(k,2) ==2;                                    % if the conditions are 1 or 2 - then it is a unisensory visual trial
        infoP(k,4) = 1;                                                     % 1 = VISUAL
    elseif infoP(k,2) ==3;                                                  % if the condition is 3 - then it is a unisensory auditory trial
        infoP(k,4) =2;                                                      % 2 = AUDITORY
    elseif infoP(k,2) >3;                                                   % if the condition is 4:9 - then it is a multisensory trial.
        infoP(k,4) =3;                                                      % 3 = MULTISENSORY
    end
end
%--------------------------------------------------------------------------


%--------------------------------------------------------------------------
% CONFLICT
for k = 1:length(infoP)
    if infoP(k,2) <=3 || infoP(k,2) == 4 || infoP(k,2) == 7;                % if the condition is less than 3 (all unisensory conditions), or is 4 or 7 - it is a 0 conflict trial
        infoP(k,5) =1;                                                      % 1 = 0 conflict
    elseif infoP(k,2) ==5 || infoP(k,2) == 8;                               % if the condition is a 5 or 8 = +2 conflict (visual faster than aud)
        infoP(k,5) =2;                                                      % 2 = +2 conflict
    elseif infoP(k,2) == 6 || infoP(k,2) == 9;                              % if the condition is a 6 or 9 = -2 conflict (visual slower than aud)
        infoP(k,5) = 3;                                                     % 3 = -2 conflict
    end
end
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
% RELIABILITY
for k = 1:length(infoP)
    if infoP(k,2) ==1 || infoP(k,2) >=3 && infoP(k,2) <7;                   % if the condition is 1,3,4,5 or 6 = high reliability
        infoP(k,6) =1;                                                      % 1 = high reliability
    elseif infoP(k,2) ==2 || infoP(k,2) >=7 && infoP(k,2) <=9;              % if the condition is 2, 7 8 or 9 = low reliability
        infoP(k,6) = 2;                                                     % 2 = low reliability
    end
end
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
% TAKE OUT MS RATES THAT DON'T EXIST
ind = find(infoP(:,3) == 1 & infoP(:,4) == 3| infoP(:,3) == 7 & infoP(:,4) ==3);
% so find rates 1 and 7, for all conditions that are labelled as
% multisensory. MS conditions are coded as 3, so pick those ones.
infoP(ind,:) = []; % delete them from the table
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
% INDIVIDUAL CONDITIONS
infoP(:,7) = 1:indivConditions;                                             % Add in individual conditions
%==========================================================================

%==========================================================================
% Repeat matrix to get n trials of each condition:
allT = length(infoP)*trials;
infoP = repmat(infoP,trials,1);
%==========================================================================

%==========================================================================
% ASSIGN RANDOM STIMULI
load experimental_stream_stimuli;                                           % exp stimuli
load comparison_stream_stimuli;                                             % comparison stimuli
[r c p] = size(allTrials);                                                  % get the size
[r1 c1] = size(allComp);                                                    % get the size

% experimental stimuli
ordExp = 1:r;
ordExp = randperm(ordExp(r))';                                              % make a random order for exp stim

% comparison stimuli
ordC = 1:r1;
ordC = randperm(ordC(r1))';                                                 % random order comp

trialOrd = zeros(allT,allFrames,p);                                         % initialise vector experimental
compOrd = zeros(allT,allFrames);                                            % initialise vector comparison

ordExp = ordExp(1:allT,1);                                                  % just take out the first x many exp stimuli
ordC = ordC(1:allT,1);                                                      % just take out the first x many comparison stim

trialOrd(:,:,:) = allTrials(ordExp,:,:);                                    % put the exp stim into a matrix
compOrd(:,:) = allComp(ordC,:);                                             % put the comp stim into a matrix

% % CHECK THE STIMULI ARE CORRECT ----- UNCOMMENT TO DO SO:
% stim = zeros(1,allT,p); % check experimental stimuli
% for m = 1:p
%     for n = 1:allT
%         stim(1,n,m) = length(find(trialOrd(n,:,m)==2)); % a stimuli is coded as a 2,
%         % so if you find where the 2's are, it tells you how many flashes
%         % there should be. So for (:,:,1), stim should all = 8, (:,:,2)
%         % =9... etc.
%     end
% end
%
% stimC = zeros(1,allT); % check comparison stim
% for n = 1:allT
%     stimC(1,n) = length(find(compOrd(n,:)==2));
%     % stimC should just equal 11.
% end

infoP(:,8) = ordExp;                                                        % store the experimental index
infoP(:,9) = ordC;                                                          % store the comparison index

tInd = 1:allT;
tInd = randperm(tInd(allT));                                                % randomise the order
infoP = infoP(tInd,:);                                                      % shuffle the whole matrix now everything is stored.

trialOrd(:,:,:) = trialOrd(tInd,:,:);                                       % shuffle the exp stim
compOrd(:,:) = compOrd(tInd,:);                                             % shuffle the comp stim

% clear the variables you don't need anymore.
clearvars -except allFrames allT compOrd conditionNumber infoP trialOrd sqHigh sqLow Do_trigger sname blocks Bgnd Day trials indivConditions
%==========================================================================

%==========================================================================
% SOUNDS
load('experimental_sound.mat');
Rate = 44100;                                                               % new rate upstairs in the lab (normally 22050)
pauses(1) = 11.965;                                                         % ms pause
pauses = pauses/1000;                                                       % convert to seconds
Tone =[];                                                                   % start with nothing
silent = zeros(round(Rate*pauses),1);
snt = (length(silent)-1);
d = length(silent);
c= 1;
%==========================================================================

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

%==========================================================================
% TRIALS
trialEmpty = ones(1,allFrames);                                             % Empty visual sequence
toneEmpty = ones(1,allFrames);                                              % empty sound:
%==========================================================================

%==========================================================================
% INITIALISE TRIGGERS
addpath('C:\toolbox');
% init
config_io;
global cogent;
address = hex2dec('D010');
if Do_trigger, io32(cogent.io.ioObj,address,0);end
%==========================================================================


%==========================================================================
% define the keyboard keys we want to use:
leftKey = KbName('left');
rightKey = KbName('right');
%==========================================================================


%==========================================================================
% SET UP SCREEN
Bgnd = 60;
AssertOpenGL;
screenNumber = max(Screen('Screens'));
% Define black, white and grey for font:
white = WhiteIndex(screenNumber); black = BlackIndex(screenNumber); grey = white / 2;
% Window:
[window, windowRect] = Screen('OpenWindow', screenNumber, Bgnd);            % fullscreen
Screen('Flip', window);                                                     % Flip to clear
[screenXpixels, screenYpixels] = Screen('WindowSize', window);              % Get the size of the on screen window
x=screenXpixels;
y=screenYpixels;
ifi = Screen('GetFlipInterval', window);                                    % Query the frame duration
[xCenter, yCenter] = RectCenter(windowRect);                                % Get the centre coordinate of the window
topPriorityLevel = MaxPriority(window);                                     % Query the maximum priority level
Screen('BlendFunction', window, 'GL_SRC_ALPHA', 'GL_ONE_MINUS_SRC_ALPHA');  % Set up alpha-blending for smooth (anti-aliased) lines
Screen('TextFont',window, 'Courier New');
Screen('TextSize',window, 14);
vbl1 = Screen('Flip', window);
%==========================================================================


%==========================================================================
% SET UP PSYCHPORTAUDIO
Rate = 44100;
InitializePsychSound(1);                                                    %force low latency
PsychPortAudio('Verbosity', 10);
deviceid=-1;                                                                %default
mode=1;                                                                     %playback only
reqlatencyclass=2;                                                          % Request latency mode 2, which used to be the best one in our measurement:
buffersize = 0;                                                             % Pointless to set this. Auto-selected to be optimal.
suggestedLatencySecs = [];
freq = Rate;
channels=2;
% open audio device:
pahandle = PsychPortAudio('Open', deviceid, mode, reqlatencyclass, Rate, channels, buffersize, suggestedLatencySecs);
%==========================================================================


%==========================================================================
% IMAGES
Bgnd = 60;
noiseC = 50;
h = screenYpixels;
w = screenXpixels;
rowPos = h/2;                                                               % middle
colPos = w/2;                                                               % middle
sq = 50;                                                                    % size of square

%Image
x1 = colPos - sq;                                                           % left side of sq
x2 = colPos + sq;                                                           % right side of sq
y1 = rowPos - sq;                                                           % top of sq
y2 = rowPos + sq;                                                           % bottom of sq

A = round(rand(h,w))*noiseC+Bgnd;                                           % noise image
B = A;
C = A;

B(y1:y2,x1:x2) = B(y1:y2,x1:x2)+sqHigh;                                     % HIGH RELIABILITY IMAGE
C(y1:y2,x1:x2) = C(y1:y2,x1:x2)+sqLow;                                      % LOW RELIABILITY IMAGE

% experimental image
imgE{1} = A;                                                                % noise
imgE{2} = B;                                                                % high reliability image
imgE{3} = C;                                                                % low reliability image

% comparison image
imgC{1} = A;
imgC{2} = B;                                                                % always high reliability
%==========================================================================


%==========================================================================
% TEXTURES
% experimental stream:
texE{1} =Screen('MakeTexture', window, imgE{1});
texE{2} =Screen('MakeTexture', window, imgE{2});                            % high reliability exp image
texE{3} =Screen('MakeTexture', window, imgE{3});                            % low reliability exp image

% comparison stream:
texC{1} =Screen('MakeTexture', window, imgC{1});
texC{2} =Screen('MakeTexture', window, imgC{2});
%==========================================================================


%==========================================================================
% EXPERIMENTAL LOOP
% set up some stuff first:
feedback = 1:2;                                                             % to give random feedback after each trial
durStim = zeros(allT,4);                                                    % initialise vector to store durations
strtTimes = zeros(allT,6,6);                                                % initialise vector to store start times

blockBreaks = trials*indivConditions/blocks; 
counter3 = 0; 
np = 1; 
blockNo = 1:blocks; 

np2 = 4;                                                % trigger counter 
%==========================================================================
% ACTUALLY START THE EXPERIMENT...
%==========================================================================

for k = 1:allT
    
    infoP(k,1) = k; % store the trial number in the table.
    HideCursor
    
    %----------------------------------------------------------------------
    % SET TRIGGER VALUES
    trial = np2+1;
    infoP(k,10) = trial;
    
    if k==200 || k == 400;
        np2 = 4;
        trial = np2+1;
        infoP(k,10) = trial;
    end
    
    
%     %----------------------------------------------------------------------
%     % TRIGGER VALUES
%     % because we have too many trials, triggers can't go that high. So need
%     % to reset it before it reaches that high.
%     if k < 200, trial = k+4; infoP(k,10) = trial; % start trigger counter
%         % at 5, and keep going until 203 (k=199+4)
%     elseif k ==200, trial =5; infoP(k,10) = trial; % when you reach trial
%         % 200, reset the trigger counter to 5.
%     elseif k >200, trial = trial+1; infoP(k,10) = trial; % once it has been
%         % reset to 5, you can just add 1 to the trigger value.
%     end
%     % so trial 1= trigger 5, trial 2 = trigger 6...... trial 200 = starts
%     % at 5 again, 201 = 6, etc....
%     
    
    % to check/put them back in the right order
    %     for k = 1:255
    %         if k <200
    %             store2(1,k) = store(1,k)-4;
    %         elseif k >=200
    %             store2(1,k) = store(1,k)+195;
    %         end
    %     end
    %----------------------------------------------------------------------
    
    %----------------------------------------------------------------------
    % CONFLICT
    switch infoP(k,5) % conflict
        case 1 % 1 = 0 conflict
            rUseV = infoP(k,3); % visual rate
            rUseA = infoP(k,3); % auditory rate
            infoP(k,11) = rUseV; % store visual rate
            infoP(k,12) = rUseA; % store auditory rate
            
        case 2 % 2 rate conflict ( visual rate is faster)
            rUseV = infoP(k,3)+1; % visual rate is faster
            rUseA = infoP(k,3)-1; % auditory rate is slower
            infoP(k,11) = rUseV; % store visual rate
            infoP(k,12) = rUseA; % store auditory rate
            
        case 3 % -2 rate conflict (auditory rate is faster)
            rUseV = infoP(k,3)-1; % visual rate is slower
            rUseA = infoP(k,3)+1; % auditory rate is faster
            infoP(k,11) = rUseV; % store visual rate
            infoP(k,12) = rUseA; % store auditory rate
    end
    %----------------------------------------------------------------------
    
    %----------------------------------------------------------------------
    % CONDITION
    switch infoP(k,4) % SET THE CONDITION
        case 1 % VISUAL UNISENSORY
            trialUse = trialOrd(k,:,rUseV); % flashing square EXPERIMENTAL
            compUse = compOrd(k,:); % flashing square COMPARISON
            toneUse = toneEmpty; % no sound EXPERIMENTAL
            toneUseComp = toneEmpty; % no sound COMPARISON
            
        case 2 % AUDITORY UNISENSORY
            trialUse = trialEmpty; % no image EXPERIMENTAL
            compUse = trialEmpty; % no image COMPARISON
            toneUse = trialOrd(k,:,rUseA); % flickering sound EXPERIMENTAL
            toneUseComp = compOrd(k,:); % flickering sound COMPARISON
            
        case 3 % MULTISENSORY
            trialUse = trialOrd(k,:,rUseV); % flashing square EXPERIMENTAL
            compUse = compOrd(k,:); % flashing square COMPARISON
            toneUse = trialOrd(k,:,rUseA); % flickering sound EXPERIMENTAL
            toneUseComp = compOrd(k,:); % flickering sound COMPARISON
    end
    %----------------------------------------------------------------------
    
    %----------------------------------------------------------------------
    % RELIABILITY
    if infoP(k,6) == 2; % if it is a low reliability trial change stim to 3
        % to use texture {3}
        indR = find(trialUse(:,:)==2);
        trialUse(:,indR)=3;
    end
    %----------------------------------------------------------------------
    
    
    %----------------------------------------------------------------------
    % Make the sound
    Tone = [];
    ToneC = [];
    
    % exp sound
    c= 1; % COUNTER
    % EXP TONE
    for s = 1:allFrames
        if toneUse(1,s) == 1 % if there is a 1 in the stim
            Tone(c:c+snt,1) = silent; % that means it's silent
            c= c+d; % change the counter
        elseif toneUse(1,s)==2 % if there is a 2 in the stim
            Tone(c:c+snt,1) = new; % that means there is a click
            c = c+d; % change the counter
        end
    end
    
    clear c
    c = 1;
    % comparison tone - same as above
    for s = 1:allFrames
        if toneUseComp(1,s) ==1
            ToneC(c:c+snt,1) = silent;
            c = c+d;
        elseif toneUseComp(1,s) ==2
            % ToneC(c:c+snt,1) = click;
            ToneC(c:c+snt,1) = new;
            c = c+d;
        end
    end
    
    %     % add in the background noise
    %     Tone = Tone+randn(size(Tone))*int;
    %     ToneC = ToneC+randn(size(ToneC))*int;
    
    % convert to Stereo sound
    Tone(:,2) = Tone;
    ToneC(:,2) = ToneC;
    %----------------------------------------------------------------------
    
    
    %----------------------------------------------------------------------
    % START EXPERIMENT
    if k == 1
        DrawFormattedText(window, ['There are ' num2str(blocks) ' blocks'], 'center', 'center', white);
        Screen('Flip', window);
        pause(1);
        DrawFormattedText(window, 'Press Left Arrow for first stream has more events', 'center',200, white);
        DrawFormattedText(window, 'Press Right Arrow for second stream has more events', 'center', 'center', white);
        DrawFormattedText(window, 'Press Any Key To Begin', 'center', 600, white);
        Screen('Flip', window);
        KbStrokeWait;
        pause(1);
    end
    %----------------------------------------------------------------------
    
    
    %----------------------------------------------------------------------
    % START TRIAL:
    strtTimes(k,:,1) = clock; % record trial start time
    %----------------------------------------------------------------------
    
    
    %======================================================================
    % TRIGGER: TRIAL START
    if Do_trigger, io32(cogent.io.ioObj,address,trial); end;
    %======================================================================
    
    
    %----------------------------------------------------------------------
    % noise screen texture:
    Screen('DrawTexture', window, texE{1});
    Screen('DrawLines', window, allCoords, lineWidthPix, white, [xCenter yCenter], 1);
    Screen('Flip', window);
    %----------------------------------------------------------------------
    
    
    %----------------------------------------------------------------------
    % LOAD SOUND FOR NEXT TRIAL:
    % load sound into memory
    PsychPortAudio('FillBuffer', pahandle, Tone'); % channelsxtime
    % this sets upt he sound - but does not play
    PsychPortAudio('Start', pahandle, 1, inf, 0);
    %----------------------------------------------------------------------
    
    
    %======================================================================
    % RESET START TRIGGER
    pause(0.1);
    if Do_trigger, io32(cogent.io.ioObj,address,0);end
    %======================================================================
    
    
    %----------------------------------------------------------------------
    pause(0.5+rand*0.5); % pause between 500 and 1100ms
    %----------------------------------------------------------------------
    
    
    %----------------------------------------------------------------------
    % EXPERIMENTAL STREAM
    % sound:
    strtTimes(k,:,2) = clock; % start time for AUDITORY EXPERIMENTAL
    soundStart = tic; % start the timing for AUDITORY DURATION
    StartTime=PsychPortAudio('RescheduleStart', pahandle, 0, 1);  % Start playback and collect start time estimate
    %----------------------------------------------------------------------
    
    
    %======================================================================
    % EEG TRIAL EXP STIMULI TRIGGER:
    if Do_trigger, io32(cogent.io.ioObj,address,1); end; % exp trigger is set to 1
    %======================================================================
    
    
    %----------------------------------------------------------------------
    % STIM
    strtTimes(k,:,3) = clock; % start time for VISUAL EXPERIMENTAL
    visualStart = tic; % start the timing for VISUAL DURATION
    % visual bit:
    for s = 1:allFrames
        % trialUse = a series of 1s and 2s or 1s and 3s that are stimuli.
        Screen('DrawTexture', window, texE{trialUse(1,s)}); % <- with a different texture for each s. 1 is noise, 2 is image.
        [vblact StimulusOnsetTime FlipTimeStamp] = Screen('Flip', window);
    end
    durStim(k,1) = toc(visualStart); % end VISUAL DURATION
    % stops playing sound - but wait for sound to finish
    PsychPortAudio('Stop', pahandle, 1);    durStim(k,2) = toc(soundStart); % end AUDITORY DURATION
    %----------------------------------------------------------------------
    
    
    %----------------------------------------------------------------------
    % put the noise screen back up before the pause
    Screen('DrawTexture', window, texE{1});
    Screen('Flip', window);
    %----------------------------------------------------------------------
    
    
    %----------------------------------------------------------------------
    % LOAD SOUND FOR NEXT TRIAL:
    PsychPortAudio('FillBuffer', pahandle, ToneC'); % channelsxtime
    % this sets upt he sound - but does not play
    PsychPortAudio('Start', pahandle, 1, inf, 0);
    %----------------------------------------------------------------------
    
    
    %======================================================================
    % RESET TRIGGER
    pause(0.1)
    % end trigger:
    if Do_trigger, io32(cogent.io.ioObj,address,0);end
    %======================================================================
    
    
    %----------------------------------------------------------------------
    pause(0.2+rand*0.2); % pause between 300-700ms
    %----------------------------------------------------------------------
    
    
    %----------------------------------------------------------------------
    % COMPARISON STREAM
    % sound:
    strtTimes(k,:,4) = clock; % start time for AUDITORY COMPARISON
    soundStart = tic; % start timing for AUDITORY DURATION
    StartTime=PsychPortAudio('RescheduleStart', pahandle, 0, 1); % Start playback and collect start time estimate
    %----------------------------------------------------------------------
    
    
    %======================================================================
    % EEG TRIAL COMP STIMULI TRIGGER
    if Do_trigger, io32(cogent.io.ioObj,address,2); end; % comp trigger is set to 2
    %======================================================================
    
    
    %----------------------------------------------------------------------
    % STIMULI
    strtTimes(k,:,5) = clock; % start time for VISUAL COMPARISON
    visualStart = tic; % start timing for VISUAL DURATION
    % visual bit :
    for s = 1:allFrames
        Screen('DrawTexture', window, texC{compUse(1,s)}); % <- with a different texture for each s
        [vblact StimulusOnsetTime FlipTimeStamp] = Screen('Flip', window);
    end
    durStim(k,3) = toc(visualStart); % end duration for VISUAL COMPARISON
    % stops playing - but wait for sound to finish
    PsychPortAudio('Stop', pahandle, 1);  durStim(k,4) = toc(soundStart);
    %----------------------------------------------------------------------
    
    
    %----------------------------------------------------------------------
    % put the noise screen back up before the pause
    Screen('DrawTexture', window, texE{1});
    Screen('Flip', window);
    %----------------------------------------------------------------------
    
    
    %======================================================================
    % EEG TRIAL END TRIGGER
    pause (0.1)
    if Do_trigger,io32(cogent.io.ioObj,address,0);  end; % when the comp stream is done - put a 0
    pause(0.1); % pause
    %======================================================================
    
    strtTimes(k,:,6) = clock; % time of trial end
    
    
    %----------------------------------------------------------------------
    % RESPONSE
    Screen('DrawTexture', window, texE{1});
    DrawFormattedText(window, 'Which stream had more events?','center', white);
    DrawFormattedText(window, ' first stream [left]       second stream [right] ', 'center', 'center', white);
    Screen('Flip', window);
    
    % wait for key press:
    respToBeMade = true;
    while respToBeMade
        [keyIsDown,secs, keyCode] = KbCheck;
        if keyCode(leftKey)
            resp = 1; % first stream had more 'events'
            respToBeMade = false;
        elseif keyCode(rightKey)
            resp = 2; % second stream had more 'events'
            respToBeMade = false;
        end
    end
    infoP(k,13) = resp;
    %----------------------------------------------------------------------
    
    
    %----------------------------------------------------------------------
    % FEEDBACK
    if infoP(k,3) < 4 && infoP(k,13) ==2 || infoP(k,3)>4 && infoP(k,13) ==1;
        Screen('DrawTexture', window, texE{1});
        DrawFormattedText(window, 'Correct :) ','center','center', white);
        Screen('Flip', window);
        pause(0.3);
        cmdDisp = [num2str(k), 'Correct'];
        disp(cmdDisp)
    elseif infoP(k,3) < 4 && infoP(k,13) ==1 || infoP(k,3)>4 && infoP(k,13) ==2;
        Screen('DrawTexture', window, texE{1});
        DrawFormattedText(window, 'Incorrect :( ','center','center', white);
        Screen('Flip', window);
        pause(0.3);
        cmdDisp = [num2str(k), 'Incorrect'];
        disp(cmdDisp)
        % because exp rate 11 is the same is comparison rate 11, there is
        % no correct answer, so needs random feedback for performance:
    elseif infoP(k,3)==4;
        fb11 = randsample(feedback,1); % pick either 1 or 2
        if fb11 ==1 % if it's 1, tell them they are correct.
            Screen('DrawTexture', window, texE{1});
            DrawFormattedText(window, 'Correct :) ','center','center', white);
            Screen('Flip', window);
            pause(0.3);
            cmdDisp = [num2str(k), 'None'];
            disp(cmdDisp)
        elseif fb11 ==2 % if it's 2, tell them they are wrong.
            Screen('DrawTexture', window, texE{1});
            DrawFormattedText(window, 'Incorrect :( ','center','center', white);
            Screen('Flip', window);
            pause(0.3);
            cmdDisp = [num2str(k), 'None'];
            disp(cmdDisp)
        end
    end
     
    Screen('DrawTexture', window, texE{1});
    Screen('DrawLines', window, allCoords, lineWidthPix, white, [xCenter yCenter], 1);
    Screen('Flip', window);
    %----------------------------------------------------------------------
    
    
%     %----------------------------------------------------------------------
%     % BLOCKS - old one 
%     if k == counter+50 && k<allT;
%         Screen('DrawTexture', window, texE{1});
%         DrawFormattedText(window,'End of Block', 'center', 200, white);
%         DrawFormattedText(window, 'Press Any Key To Begin Again', 'center', 'center', white);
%         Screen('Flip', window);
%         KbStrokeWait;
%         counter = counter+51;
%         disp('Block')
%     elseif k ==allT
%         Screen('DrawTexture', window, texE{1});
%         DrawFormattedText(window,'End', 'center', 200, white);
%         Screen('Flip', window);
%         disp('end')
%         pause(0.2)
%     end
%     %----------------------------------------------------------------------
    
    
    
        %----------------------------------------------------------------------
    % BLOCKS
    if k == counter3+blockBreaks && k<allT;
        Screen('DrawTexture', window, texE{1});
        DrawFormattedText(window,sprintf('End of Block %d',blockNo(np)), 'center', 200, white);
        DrawFormattedText(window, 'Press Any Key To Begin Again', 'center', 'center', white);
        Screen('Flip', window);
        KbStrokeWait;
        counter3 = counter3+blockBreaks;
        np = np+1; 
        disp('Block')
    elseif k ==allT
        Screen('DrawTexture', window, texE{1});
        DrawFormattedText(window,'End', 'center', 200, white);
        Screen('Flip', window);
        disp('end')
        pause(0.2)
    end
    %----------------------------------------------------------------------
    
    
    clear trialUse compUse toneUse  toneUseComp Tone ToneC cmdDisp % reset these for the next trial
    save(sname,'infoP');
    
    %----------------------------------------------------------------------
    pause(0.5+rand*0.1); % 
    %----------------------------------------------------------------------
    
    np2 = np2+1; % trigger counter
    disp(trial)    
end


infoP(:,14) = Day; % add in what day 
save(sname,'infoP','sqHigh','sqLow','durStim', 'strtTimes');
Screen('CloseAll')
PsychPortAudio('Close');
ShowCursor



%--------------------------------------------------------------------------
% RESULTS BIT
%--------------------------------------------------------------------------


resultsAll = zeros(3,7);
cond = {'Vis H','Vis L','Aud','Ms High','Ms Low'};
conds = [1 2 3 4 7]; 
for m = 1:length(conds)
    for k = 1:7
        Ind = infoP(infoP(:,2)==conds(m),:);
        
        if k <=4
            resultsAll(m,k) = length(find(Ind(:,3)==k & Ind(:,13)==2))/length(find(Ind(:,3)==k));
        elseif k >4
            resultsAll(m,k) = length(find(Ind(:,3)==k & Ind(:,13)==1))/length(find(Ind(:,3)==k));
        end
    end
    figure(m)
    plot(resultsAll(m,:),'o-')
    ylim([0 1]);xlim([0 8]);
%     hline(0.5,'color','r');
    title(cond(m))
end

disp(resultsAll);

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % ANALYSIS FOR BLOCKS - see how performance is at end of each block
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % INFOP TABLE
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %       1              2               3         4         5
% % [trial number, condition (1:9), rate (1:7), modality, conflict,
%
% %       6                 7             8               9         10
% % reliability, individual condition, index Exp, index Comp, trigger value,

%
% %       11          12            13
% % visual rate, auditory rate, response
% %
%
% indResp = length(find(infoP(:,3)==4)); % we don't want to count the '11' rate - as these are always wrong by design
% correct = (length(find(infoP(:,3)<4 & infoP(:,13)==2 | infoP(:,3)>4 & cinfoP(:,13)==1)))/(allT-indResp)
%
% % Above line: finds where the exp stream rate is less than 4 (so, less than
% % 11) AND the participant responded that the second stream was faster (correct trial).
% % And find where the rate is more than 4 (so, more than 11) and the participant
% % responded that the first stream was faster (correct trial).
% % Then divide it by the number of trials (minus the trials (indResp) that are
% % always incorrect).

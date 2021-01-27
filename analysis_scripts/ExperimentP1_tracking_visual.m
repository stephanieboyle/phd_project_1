%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% tracking block visual
% Run n trials of different reliabilities to see what to set visual
% thresholds at before main experiment.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%--------------------------------------------------------------------------
% STUFF
clear
close all
 
load allTrials7; % stimuli (event rates 8:14)
load comp7; % comparison stimuli (event rate 11)
rand('seed', sum(100 * clock));
cl = clock;
int = 0;
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
% INPUT
Subj = input('Subject: ','s'); % enter subject number
Block = input('Block: ');
Day = input('Day: ');

% % % save: Kayserlab folder computer
 sname = sprintf('C:/Kayserlab/Stephanie B/Project1Lab/log/TrackVis_%s_B%d_D%d_%02d%02d_%02d%02d.mat',Subj,Block,Day,cl(2),cl(3),cl(4),cl(5));
% %--------------------------------------------------------------------------
% 

%--------------------------------------------------------------------------
% TRIALS
% vRel = 10:8:55; % set up different reliabilities
vRel = 10:15:70; 
% conditions
cond(:,1) = 1:length(vRel); % set how many reliability levels you are testing

% rates
rateLow = 1; % set the lower comparison rate (1=8)
rateHigh = 7; % set the higher comparison rate (7=14)

% trials
trials = 10; % set the number of trials you want to test
allT = (trials*length(cond))*2; % all trials
allFrames = 75; % frame rate
all = zeros(allT,8); % create the matrix

% create all trials:
% all(:,1) = repmat(vRel',(trials*2),1);  % put the reliability levels in
all(1:(length(cond)*trials),1) = rateLow; % put the low rates in
all(((length(cond)*trials)+1):end,1) = rateHigh; % put the high rates in
all(:,2) = repmat(cond,(trials*2),1); % put the conditions in
%--------------------------------------------------------------------------


%--------------------------------------------------------------------------
% STIMULI
% take some random stimuli out for testing
indVis = datasample(1:1000,length(all))'; % get a random list of stimuli to take
all(:,3) = indVis; % store it
indVisComp = datasample(1:7000,length(all))'; % get a random list of comparison stimuli to take
all(:,4) = indVisComp; % store it

% shuffle matrix
ind2 = randperm(length(all));
all = all(ind2,:); % shuffled order

% make your experimental stimuli in one matrix:
trackVisT = zeros(length(all),allFrames);
for k = 1:length(all);
    trackVisT(k,:) = allTrials(all(k,3),:,(all(k,1))); % this takes the random stimuli, all rows of it, from either rate or 5 depending on what rPerm(j) is for that trial
end

% make your comparison stimuli
trackVisComp = zeros(length(all),allFrames);
for k = 1:length(all);
    trackVisComp(k,:) = allComp(all(k,4),:); % no 3rd dim rate - all 11.sec
end

% check:
for k = 1:length(all)
    temp = length(find(trackVisT(k,:)==2));
    all(k,5) = temp;
    temp2 = length(find(trackVisComp(k,:)==2));
    all(k,6)  = temp2;
end

% Program which stream has more/less for later:
for k = 1:length(all)
    if all(k,1) == rateLow % if rate is low
        all(k,7) = 2; % then the correct response is second stream more
    elseif all(k,1) == rateHigh; % if the rate is high
        all(k,7) = 1; % then the correct response is first stream more
    end
end
%--------------------------------------------------------------------------


%--------------------------------------------------------------------------
% SCREEN STUFF
Bgnd = 60;
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
% IMAGES
Bgnd = 60;
noiseC = 50;
rowPos = screenYpixels/2; % middle
colPos = screenXpixels/2; % middle
sq = 50; % size of square

x1 = colPos - sq; % left side of sq
x2 = colPos + sq; % right side of sq
y1 = rowPos - sq; % top of sq
y2 = rowPos + sq; % bottom of sq

A = round(rand(screenYpixels,screenXpixels)*noiseC)+Bgnd; % noise image
B = A;
%--------------------------------------------------------------------------


%--------------------------------------------------------------------------
% RELIABILITIES
imgE = cell(1,length(vRel));
imgC = cell(1,1);

imgE{1} = A; % noise image experimental
imgC{1} = A; % noise image comparison

for k = 2:length(vRel)+1
    B = A; % set B to plain noise image
    B(y1:y2,x1:x2) = B(y1:y2,x1:x2)+vRel(k-1); % make a square that is
    % of a certain 'reliability' level (vRel)
    imgE{k} = B; % img{2:7} = different reliabilities from lowest to highest.
end

imgC{2} = imgE{length(imgE)}; % always high reliability square for the comparison image (high reliability is stored in img{1}
%--------------------------------------------------------------------------


%--------------------------------------------------------------------------
% define Keyboard 
leftKey = KbName('left');
rightKey = KbName('right');
counter = 1; % need this for feedback later
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
% TEXTURES
texE{1} = Screen('MakeTexture', window, imgE{1}); % noise image
for k = 1:length(imgE)
    texE{k+1} = Screen('MakeTexture', window, imgE{k}); % noise image
end

% comparison stream:
texC{1} =Screen('MakeTexture', window, imgC{1}); % noise image
texC{2} =Screen('MakeTexture', window, imgC{2}); % high reliability image
%--------------------------------------------------------------------------

timeStim = zeros(length(all),2); % to test timing

%--------------------------------------------------------------------------
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
%--------------------------------------------------------------------------


%==========================================================================
% EXPERIMENT
%==========================================================================
HideCursor

for k = 1:length(all)
    
    trialUse = trackVisT(k,:); % pick out the first exp stimuli to use
    compUse = trackVisComp(k,:); % pick out the first comp stim to use
    trial = k;
    
    %----------------------------------------------------------------------
    % RELIABILITY
    
    switch all(k,2)
        case 1 % lowest reliability (stored in texE{2})
            indR = find(trialUse(:,:)==2);
            trialUse(:,indR)=2;
        case 2 % second lowest reliability (stored in texE{3})
            indR = find(trialUse(:,:)==2);
            trialUse(:,indR)=3;
        case 3 % third lowest reliability (stored in texE{4})
            indR = find(trialUse(:,:)==2);
            trialUse(:,indR)=4;
        case 4 % fourth lowest reliability (stored in texE{5})
            indR = find(trialUse(:,:)==2);
            trialUse(:,indR)=5;
        case 5 % fifth lowest reliability (stored in texE{6})
            indR = find(trialUse(:,:)==2);
            trialUse(:,indR)=6;
        case 6 % sixth lowest reliability (stored in texE{7})
            indR = find(trialUse(:,:)==2);
            trialUse(:,indR)=7;
    end
    
    %----------------------------------------------------------------------
    % Actual Experiment:
    if k==1
        DrawFormattedText(window, 'Press Any Key To Begin', 'center', 'center', white);
        Screen('Flip', window);
        KbStrokeWait;
        pause(1);
    end
    %----------------------------------------------------------------------
    
    
    %----------------------------------------------------------------------
    % FIXATION CROSS
    Screen('DrawTexture',window,texE{1});
    Screen('DrawLines', window, allCoords, lineWidthPix, white, [xCenter yCenter], 1);
    Screen('Flip', window);
    %----------------------------------------------------------------------

    %----------------------------------------------------------------------
    pause(0.5+rand*0.5); % pause between 500 and 1100ms
    %----------------------------------------------------------------------

    %----------------------------------------------------------------------
    % EXPERIMENTAL STREAM
    % visual bit
    visualStart = tic;
    for s = 1:allFrames
        Screen('DrawTexture', window, texE{trialUse(s)}); % <- with a different texture for each k
        [vblact,StimulusOnsetTime,FlipTimeStamp] = Screen('Flip', window);
    end
    timeStim(k,1) = toc(visualStart);
    Screen('DrawTexture',window,texE{1}); % put noise screen back up
    Screen('Flip', window);
    pause(0.4+rand*0.3); % pause between 300-700ms
    %----------------------------------------------------------------------
    
    
    %----------------------------------------------------------------------
    % COMPARISON STREAM
    visualStart = tic;
    for s = 1:allFrames
        Screen('DrawTexture', window, texC{compUse(s)}); % <- with a different texture for each s
        [vblact,StimulusOnsetTime,FlipTimeStamp] = Screen('Flip', window);
    end
    timeStim(k,2) = toc(visualStart);
    Screen('DrawTexture',window,texE{1}); % put noise screen back up
    Screen('Flip', window);
    pause(0.2); % pause
    %----------------------------------------------------------------------
    
    
    %----------------------------------------------------------------------
    % RESPONSE
    Screen('DrawTexture',window,texE{1});
    DrawFormattedText(window, 'Which stream had more flashes?','center', white);
    DrawFormattedText(window, ' first stream [left]       second stream [right] ', 'center', 'center', white);
    Screen('Flip', window);
    
    respToBeMade = true;
    while respToBeMade
        [keyIsDown,secs, keyCode] = KbCheck;
        if keyCode(leftKey)
            resp = 1; % first stream
            respToBeMade = false;
        elseif keyCode(rightKey)
            resp = 2; % second stream
            respToBeMade = false;
        end
    end
    
    all(k,8) = resp;
    %----------------------------------------------------------------------
    
    
    %----------------------------------------------------------------------
    % GIVE FEEDBACK
    if all(k,7)== all(k,8);
        Screen('DrawTexture',window,texE{1});
        DrawFormattedText(window, 'Correct :) ','center','center', white);
        Screen('Flip', window);
        pause(0.3);
    elseif all(k,7) ~= all(k,8);
        Screen('DrawTexture',window,texE{1});
        DrawFormattedText(window, 'Incorrect :( ','center','center', white);
        Screen('Flip', window);
        pause(0.3);
    end
    %----------------------------------------------------------------------
    
    %----------------------------------------------------------------------
    % BLOCKS
    if k == counter+30;
        Screen('DrawTexture',window,texE{1});
        DrawFormattedText(window, 'Break...', 'center', 200, white);
        DrawFormattedText(window, 'Press Any Key To Begin Again', 'center', 'center', white);
        Screen('Flip', window);
        KbStrokeWait;
        pause(1);
        counter = counter+30;
    end
    %----------------------------------------------------------------------
    
    clear trialUse trialUseComp 
    save(sname,'all');
    pause(1)
end


Screen('CloseAll')
PsychPortAudio('CloseAll')
ShowCursor

%
% % ALL table:
% % [:,1] = Rate (low or high)
% % [:,2] = condition (1:6)
% % [:,3] = original matrix row
% % [:,4] = original comparison matrix row
% % [:,5] = exp rate check
% % [:,6] = comp rate check
% % [:,7] = correct response
% % [:,8] = actual response



%--------------------------------------------------------------------------
% RESULTS BIT
%--------------------------------------------------------------------------
trials = length(find(all(:,2)==1));

trackResultsV = zeros(length(vRel),3);
trackResultsV(:,1) = vRel;
trackResultsV(:,3) = trials;

for k = 1:length(vRel)
    trackResultsV(k,2) = length(find(all(:,2)==k & all(:,7)==all(:,8)))/trials;
end


figure(1);clf;
Stimulus = trackResultsV(:,1)';
Performance = trackResultsV(:,2)';
plot(Stimulus,Performance,'o')
% options for fminsearch function
options = optimset;
options.Display = 'off';
options.MaxFunEvals = 100000;
options.MaxIter = 100000;

% fit the parameters of the Gaussian cumulative function defined in
% local_cumulgauss
params = fminsearch(@(x) local_cumulgauss_1afc(Stimulus,Performance,x),[4,3,2]);
% compute Gaussian with this parameters
Gauss = normcdf(Stimulus,params(1),params(2))*params(3);
hold on;
plot(Stimulus,Gauss,'r');

% params(1) is threshold for 50%, params(2) is the slope
% find Stimulus value giving 75% performance
% compute Gauss based on finer sampling:
dx = (Stimulus(end)-Stimulus(1))/200;
xnew = [Stimulus(1):dx:Stimulus(end)];
Gauss = normcdf(xnew,params(1),params(2))*params(3);
% find 75% point
[~,o] = min(abs(Gauss-0.75));
THR = (xnew(o));

line([THR THR],[0 Gauss(o)]);
line([Stimulus(1) THR],[Gauss(o) Gauss(o)])
% saveas(gcf,sprintf('trackVisRates_%s_D%d_ckfunct',Subj,Day), 'fig')


% %%
% % 
% % psignifit:
% figure; plotpd(trackResultsV,'color','black');
% hold on
% add psychometric curve
% shape = 'cumulative Gaussian';
% n_intervals = 1;
% prefs = batch('shape',shape,'n_intervals',n_intervals,'runs', 999);
% outputPrefs = batch('write_pa','pa','write_th','th');
% T1 = psignifit(trackResultsV(:,:,1), [prefs outputPrefs]);
% plotpf(shape,pa.est,'color','red');
% drawHeights = psi(shape, pa.est, th.est);
% line(th.lims, ones(size(th.lims,1), 1) * drawHeights, 'color', 'red')
% saveas(gcf,sprintf('trackVisRates_%s_D%d_psign',Subj,Day), 'fig')
% % saveas(gcf,sprintf('trackVis_%s',Subj), 'tiff')
% 










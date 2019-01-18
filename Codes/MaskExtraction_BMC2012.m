 clc
 close all; 
 clear variables;
 %% reading video file and save it as a matrix
 FolderName = {'Video_002','Video_003','Video_004',...
     'Video_006','Video_007','Video_008'};
 l = length(FolderName);
 Disp = 0;
 savePathSt = fullfile('.','Result','BMC2012');
 VidPathSt = fullfile('..', 'Data');
%  counter = 1;
 for counter = 1:1:length(FolderName)
     % reading original video frames
     path = fullfile(VidPathSt,FolderName{counter}, strcat(FolderName{counter},'.avi'));
     vid = VideoReader(path);
     % starting at specific time
     vid.CurrentTime = 0; % in seconds
     i=1;
     while hasFrame(vid)
         vidFrame = readFrame(vid);
    %      vidFrame = vidFrame(x_min:x_max, y_min:y_max, :);
         imArray(:,:,:,i) = vidFrame;
         i = i+1;
     end
% reading the background model

    %% save the binary mask as video
    power =1;
    coef = 1;
    ForegEn = coef * E .^ power;
%     clear E
    % ForegEn = Foreground;
    Th = (1/4) * max(max(ForegEn));
    % thresholding
    ForegMask = ForegEn > Th;
    % morphologocal processing
    ForegMask = imopen(ForegMask, strel('rectangle', [3,3]));
    ForegMask = imclose(ForegMask, strel('rectangle', [5, 5]));
    ForegMask = imfill(ForegMask, 'holes');
    ForegMask = 255* uint8(reshape(ForegMask,height, width, []));
    v = VideoWriter(fullfile(savePath,'forground.avi'), 'Grayscale AVI');
    v.FrameRate = 10;
    % v.Colormap = 'Grayscale AVI';
    open(v)
    writer.FrameRate = vid.FrameRate;
    writeVideo(v, ForegMask);
    close(v);
    for j=1:1:size(ForegMask,3)
    FileName = strcat(num2str(j), '.bmp');
    path = fullfile(savePath, FileName);
    imwrite(ForegMask(:, :, j), path);
    end
    save(fullfile(savePath,'elapse-time.txt'), 'tElapsed','-ascii');
 end
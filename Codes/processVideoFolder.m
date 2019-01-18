%THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
%AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
%IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
%DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
%FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
%DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
%SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
%CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
%OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
%OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

% Nil Goyette
% University of Sherbrooke
% Sherbrooke, Quebec, Canada. April 2012

function confusionMatrix = processVideoFolder(videoPath, binaryFolder)
    % A video folder should contain 2 folders ['input', 'groundtruth']
	% and the "temporalROI.txt" file to be valid. The choosen method will be
	% applied to all the frames specified in \temporalROI.txt
    
    range = readTemporalFile(videoPath);
    idxFrom = range(1);
    idxTo = range(2);
    display(['Processing ', videoPath, char(10), 'Saving to ', binaryFolder, char(10), 'From frame ' ,  num2str(idxFrom), ' to ',  num2str(idxTo), char(10)]);
	
	% Create binary images with your method
	% TODO Choose between Matlab code OR executable on disk?
	
	% TODO If matlab code, create the function and call it
	% YourMethod(videoPath, binaryFolder, idxFrom, idxTo);

	% TODO If executable on disk, change the path and add parameters if desired
    %[status, result] = system(['/path/to/executable' ' ' videoPath ' ' binaryFolder]);
    %if status ~= 0,
    %   disp('There was an error while calling your executable.');
    %   disp(['result =' result '\n\nStopping executtion.']);
	%   exit
    %end

    % Compare your images with the groundtruth and compile statistics
    groundtruthFolder = fullfile(videoPath, 'groundtruth');
    confusionMatrix = compareImageFiles(groundtruthFolder, binaryFolder, idxFrom, idxTo);
end

function range = readTemporalFile(path)
    % Reads the temporal file and returns the important range
    
    fID = fopen([path, '\temporalROI.txt']);
	if fID < 0
        disp(ferror(fID));
        exit(0);
    end
	
    C = textscan(fID, '%d %d', 'CollectOutput', true);
    fclose(fID);
    
    m = C{1};
    range = m';
end

function confusionMatrix = compareImageFiles(gtFolder, binaryFolder, idxFrom, idxTo)
    % Compare the binary files with the groundtruth files.
    
    extension = '.jpg'; % TODO Change extension if required
    threshold = strcmp(extension, '.jpg') == 1 || strcmp(extension, '.jpeg') == 1;
    
    imBinary = imread(fullfile(binaryFolder, ['bin', num2str(idxFrom, '%.6d'), extension]));
    int8trap = isa(imBinary, 'uint8') && min(min(imBinary)) == 0 && max(max(imBinary)) == 1;
    
    confusionMatrix = [0 0 0 0 0]; % TP FP FN TN SE
    for idx = idxFrom:idxTo
        fileName = num2str(idx, '%.6d');
        imBinary = imread(fullfile(binaryFolder, ['bin', fileName, extension]));
        if size(imBinary, 3) > 1
            imBinary = rgb2gray(imBinary);
        end
        if islogical(imBinary) || int8trap
            imBinary = uint8(imBinary)*255;
        end
        if threshold
            imBinary = im2bw(imBinary, 0.5);
            imBinary = im2uint8(imBinary);
        end
        imGT = imread(fullfile(gtFolder, ['gt', fileName, '.png']));
        
        confusionMatrix = confusionMatrix + compare(imBinary, imGT);
    end
end

function confusionMatrix = compare(imBinary, imGT)
    % Compares a binary frames with the groundtruth frame
    
    TP = sum(sum(imGT==255&imBinary==255));		% True Positive 
    TN = sum(sum(imGT<=50&imBinary==0));		% True Negative
    FP = sum(sum((imGT<=50)&imBinary==255));	% False Positive
    FN = sum(sum(imGT==255&imBinary==0));		% False Negative
    SE = sum(sum(imGT==50&imBinary==255));		% Shadow Error
	
    confusionMatrix = [TP FP FN TN SE];
end

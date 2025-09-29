% =========================================================================
%                             train_model.m
% =========================================================================
% Use this script ONLY for training. It will automatically save and
% resume from checkpoints.
% =========================================================================

clear;
close all;
clc;

%% Step 1: Load Data
disp('Step 1: Loading train and test datasets...');
base_folder = 'C:\Users\dauru\OneDrive\Documents\MATLAB\proj\chest_xray'; 
train_folder = fullfile(base_folder, 'train');
validation_folder = fullfile(base_folder, 'test');

imdsTrain = imageDatastore(train_folder, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
imdsValidation = imageDatastore(validation_folder, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
disp('Datasets loaded successfully.');

%% Step 2: Run Two-Stage Training
model_filename = 'resnet50_finetuned_model.mat';

% --- STAGE 1: TRANSFER LEARNING WITH RESNET-50 ---
disp('--- Starting Stage 1: Transfer Learning ---');

% --- CORRECTED LOGIC TO RESUME FROM CHECKPOINT ---
checkpointPath_S1 = 'stage1_checkpoints';
if ~exist(checkpointPath_S1, 'dir'), mkdir(checkpointPath_S1); end

% Find the latest checkpoint file in the directory
latestCheckpoint = dir(fullfile(checkpointPath_S1, '*.mat'));
if ~isempty(latestCheckpoint)
    % If a checkpoint exists, load it to resume training
    [~, idx] = max([latestCheckpoint.datenum]); % Find the newest file
    latestFile = fullfile(checkpointPath_S1, latestCheckpoint(idx).name);
    
    disp(['Resuming Stage 1 training from checkpoint: ' latestFile]);
    data = load(latestFile); % Load contents into a structure
    lgraph = layerGraph(data.net); % Access the network from the 'net' field
else
    % If no checkpoint, start a new training session from the original ResNet-50
    disp('No Stage 1 checkpoint found. Starting new training...');
    net = resnet50;
    lgraph = layerGraph(net);
    numClasses = numel(categories(imdsTrain.Labels));
    newFCLayer = fullyConnectedLayer(numClasses, 'Name','new_fc', 'WeightLearnRateFactor',10, 'BiasLearnRateFactor',10);
    newClassLayer = classificationLayer('Name','new_classoutput');
    lgraph = replaceLayer(lgraph, 'fc1000', newFCLayer);
    lgraph = replaceLayer(lgraph, 'ClassificationLayer_fc1000', newClassLayer);
end
% --- END OF CORRECTED LOGIC ---

% Prepare data (this must be done in both cases)
inputSize = lgraph.Layers(1).InputSize(1:2);
augimdsTrain = augmentedImageDatastore(inputSize, imdsTrain, 'ColorPreprocessing', 'gray2rgb');
augimdsValidation = augmentedImageDatastore(inputSize, imdsValidation, 'ColorPreprocessing', 'gray2rgb');

% Set options and train
options_stage1 = trainingOptions('sgdm', 'MiniBatchSize',16, 'MaxEpochs',20, 'InitialLearnRate',1e-4, 'Shuffle','every-epoch', 'ValidationData',augimdsValidation, 'ValidationFrequency',10, 'Verbose',false, 'Plots','training-progress', 'CheckpointPath', checkpointPath_S1);
[trainedNet_Stage1, ~] = trainNetwork(augimdsTrain, lgraph, options_stage1);
disp('--- Stage 1 training complete. ---');

% --- STAGE 2: FINE-TUNING THE TRAINED RESNET-50 ---
disp('--- Starting Stage 2: Fine-Tuning ---');
lgraph_finetune = layerGraph(trainedNet_Stage1);
layersToUnfreeze = {'res5c_branch2c', 'res5c_branch2b', 'res5b_branch2c', 'res5b_branch2b'};
for i = 1:numel(layersToUnfreeze)
    layerName = layersToUnfreeze{i};
    layer = lgraph_finetune.Layers(arrayfun(@(x) strcmp(x.Name, layerName), lgraph_finetune.Layers));
    if isa(layer, 'matlab.cnn.layer.Convolution2DLayer')
        layer.WeightLearnRateFactor = 1;
        layer.BiasLearnRateFactor = 1;
    end
    lgraph_finetune = replaceLayer(lgraph_finetune, layerName, layer);
end
disp("Deeper layers have been unfrozen.");
checkpointPath_S2 = 'stage2_checkpoints';
if ~exist(checkpointPath_S2, 'dir'), mkdir(checkpointPath_S2); end
options_finetune = trainingOptions('sgdm', 'MiniBatchSize',16, 'MaxEpochs',10, 'InitialLearnRate',1e-5, 'Shuffle','every-epoch', 'ValidationData',augimdsValidation, 'ValidationFrequency',10, 'Verbose',false, 'Plots','training-progress', 'CheckpointPath', checkpointPath_S2);
[trainedNet_Finetuned, ~] = trainNetwork(augimdsTrain, lgraph_finetune, options_finetune);
disp('--- Fine-tuning complete! ---');

disp(['Saving final model to ', model_filename, '...']);
save(model_filename, 'trainedNet_Finetuned');
disp('Model saved successfully.');
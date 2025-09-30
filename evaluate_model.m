clear;
close all;
clc;

%% Step 1: Load the Final Model and Data
disp('Step 1: Loading final model and validation data...');
load('resnet50_finetuned_model.mat');
finalNet = trainedNet_Finetuned;

base_folder = 'C:\Users\dauru\OneDrive\Documents\MATLAB\proj\chest_xray'; 
validation_folder = fullfile(base_folder, 'test');
imdsValidation = imageDatastore(validation_folder, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
disp('Model and data loaded.');

%% Step 2: Evaluate the Model
disp('Step 2: Evaluating final model performance...');
inputSize = finalNet.Layers(1).InputSize;
augimdsValidation = augmentedImageDatastore(inputSize(1:2), imdsValidation, 'ColorPreprocessing', 'gray2rgb');
YPred = classify(finalNet, augimdsValidation);
YValidation = imdsValidation.Labels;
accuracy = mean(YPred == YValidation);
disp(['Final Model Accuracy: ', num2str(accuracy * 100, '%.2f'), '%']);
figure;
confusionchart(YValidation, YPred);
title('Confusion Matrix for Test Data');

%% Step 3: Test on a Single New Image
disp('Step 3: Performing test on a single new image...');
newImage = imread('C:\Users\dauru\OneDrive\Documents\MATLAB\proj\chest_xray\person5_bacteria_19.jpeg');
inputSize = finalNet.Layers(1).InputSize;
if size(newImage, 3) == 3
    imgGray = rgb2gray(newImage);
else
    imgGray = newImage;
end
imgResized = imresize(imgGray, [inputSize(1) inputSize(2)]);
imgForNet = cat(3, imgResized, imgResized, imgResized);
[label, confidence] = classify(finalNet, imgForNet);
figure;
imshow(newImage);
title(['Prediction: ', char(label), ' | Confidence: ', num2str(max(confidence)*100, '%.2f'), '%']);
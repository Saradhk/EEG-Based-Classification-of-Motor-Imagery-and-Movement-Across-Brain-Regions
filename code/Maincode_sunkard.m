% Step 1: Setup
addpath('F:\eeglab\eeglab2024.2.1\'); % Add EEGLAB path
eeglab; % Start EEGLAB

% Define subject IDs
numSubjects = 52;
subjectIDs = arrayfun(@(x) sprintf('s%02d', x), 1:numSubjects, 'UniformOutput', false);

% Define conditions
conditions = {'movement','imagery'};

% Define channel names
channelNames = { 'FP1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', ...
    'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', ...
    'CP1', 'P1', 'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'IZ', ...
    'OZ', 'POZ', 'PZ', 'CPZ', 'FPZ', 'FP2', 'AF8', 'AF4', 'AFZ', ...
    'FZ', 'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', ...
    'FCZ', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', ...
    'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2','O3', 'O4', 'O5', 'O6'};

% Define selected electrodes

%'P4'- TP7 - 'CP4', 'FC4'-'C4'-'C3','T7', 'P3','FC3', 'FCZ','CP3', 'CPZ', 'PZ','T8' 
selectedElectrodes = { 'C3', 'CZ', 'FC3', 'FCZ','CP3', 'CPZ', 'P3', 'PZ','T7', 'T8'};
selectedIndices = find(ismember(channelNames, selectedElectrodes));

% Define electrode groups based on Brodmann areas
electrodeGroups = struct();
electrodeGroups.BA6 = {'FC3', 'FC4', 'FCZ'};  % Example frontal electrodes
electrodeGroups.BA4 = {'C3', 'C4', 'CZ'};   % Example central electrodes
electrodeGroups.BA5 = {'CP3', 'CP4', 'CPZ'};% Example parietal electrodes
electrodeGroups.BA7 = {'P3', 'P4', 'PZ'};
electrodeGroups.BA20to22 = {'T7', 'T8'};   % Example temporal electrodes

% Find indices for each group (based on selectedElectrodes, not channelNames)
groupIndices = struct();
fields = fieldnames(electrodeGroups);
for f = 1:numel(fields)
    groupIndices.(fields{f}) = find(ismember(selectedElectrodes, electrodeGroups.(fields{f})));

% Ensure indices exist
 if isempty(groupIndices.(fields{f}))
    fprintf('Warning: Group %s has no matching electrodes in selectedElectrodes.\n', fields{f});
 end

end

% Initialize feature matrix and labels
features = [];
labels = [];

% Initialize storage for averaged PSD
avg_psd = cell(1, length(conditions)); % Store PSD values per condition
freqs_used = []; % Store frequency values (will be same for all)

% Step 2: Process each subject
for s = 1 :numSubjects
    subjectFile = sprintf('F:\\MAT\\%s.mat', subjectIDs{s}); % Path to subject's .mat file
    load(subjectFile); % Load EEG data
    for i = 1:length(conditions)
       
        % Assign EEG data and sampling rate
        if strcmp(conditions{i}, 'movement')
    EEG.data = mean(cat(3, eeg.movement_left, eeg.movement_right), 3);
   elseif strcmp(conditions{i}, 'imagery')
    EEG.data = mean(cat(3, eeg.imagery_left, eeg.imagery_right), 3);
       end
        EEG.srate = eeg.srate; % Assign sampling rate
        
        % Retain only selected electrodes
        EEG.data = EEG.data(selectedIndices, :);
        EEG.nbchan = length(selectedIndices);
        
        % Define time window
        epochStartTime = 0; % Start time in seconds
        epochEndTime = 7; % End time in seconds
        epochStartSample = epochStartTime * EEG.srate + 1;
        epochEndSample = epochEndTime * EEG.srate;
        
        % Extract the time segment
        EEG.data = EEG.data(:, epochStartSample:epochEndSample);
        
        % Apply baseline correction
        baselineTime = [0 1]; % Baseline time in seconds
        baselineStartSample = baselineTime(1) * EEG.srate + 1;
        baselineEndSample = baselineTime(2) * EEG.srate;
        
        if baselineStartSample >= 1 && baselineEndSample <= size(EEG.data, 2)
            baselineData = mean(EEG.data(:, baselineStartSample:baselineEndSample), 2);
            EEG.data = EEG.data - baselineData; % Apply baseline correction
        end
      % Define Notch Filter Parameters
        notchFreq = 60; % Adjust to 60 Hz if needed
        Q = 30; % Quality factor, adjust for different bandwidths
        wo = notchFreq / (EEG.srate / 2); % Normalized frequency
        [b, a] = iirnotch(wo, wo / Q); % Create Notch filter

% Apply Notch Filter to Each Channel
  for ch = 1:size(EEG.data, 1)
      EEG.data(ch, :) = filtfilt(b, a, EEG.data(ch, :)); % Zero-phase filtering
  end
  % **Compute PSD for Each Subject**
     psd_values_subject{i} = zeros(length(selectedElectrodes), length(freqs_used)); 

% Now preallocate with correct dimensions

    for ch = 1:size(EEG.data, 1)
            [psd_values, freqs] = pwelch(EEG.data(ch, :), [], [], [], EEG.srate);
            % Store frequency values only once
            if isempty(freqs_used)
                freqs_used = freqs;
            end

            if isempty(psd_values_subject{i})
                psd_values_subject{i} = zeros(length(selectedElectrodes), length(psd_values)); 
            end

            psd_values_subject{i}(ch, :) = psd_values(:).'; % Ensure row format
    end

    % Accumulate PSD values across subjects
        if isempty(avg_psd{i})
            avg_psd{i} = psd_values_subject{i};
        else
            avg_psd{i} = avg_psd{i} + psd_values_subject{i};
        end

        % **Step 4: Feature Extraction**
        theta_band = mean(arrayfun(@(ch) bandpower(EEG.data(ch, :), EEG.srate, [4 8]), 1:size(EEG.data, 1)));
        alpha_band = mean(arrayfun(@(ch) bandpower(EEG.data(ch, :), EEG.srate, [8 12]), 1:size(EEG.data, 1)));
        beta_band  = mean(arrayfun(@(ch) bandpower(EEG.data(ch, :), EEG.srate, [12 30]), 1:size(EEG.data, 1)));
        gamma_band = mean(arrayfun(@(ch) bandpower(EEG.data(ch, :), EEG.srate, [30 100]), 1:size(EEG.data, 1)));

        % **Extract Time-Domain Features**
        mean_val = mean(EEG.data, 2);      % Mean
        variance_val = var(EEG.data, 0, 2); % Variance
        skewness_val = skewness(EEG.data, 1, 2); % Skewness
        kurtosis_val = kurtosis(EEG.data, 1, 2); % Kurtosis
        rms_val = rms(EEG.data, 2);         % RMS
        zcr_val = sum(diff(sign(EEG.data), 1, 2) ~= 0, 2) / size(EEG.data, 2); % Zero-crossing rate
        entropy_val = arrayfun(@(ch) wentropy(EEG.data(ch, :), 'shannon'), 1:size(EEG.data, 1));

       % **Compute Hjorth Parameters**
       first_derivative = diff(EEG.data, 1, 2);
       second_derivative = diff(first_derivative, 1, 2);

       activity = variance_val; % Hjorth Activity
       mobility = std(first_derivative, 0, 2) ./ sqrt(activity); % Hjorth Mobility
       complexity = std(second_derivative, 0, 2) ./ std(first_derivative, 0, 2); % Hjorth Complexity

       % **Create feature vector (combine time and frequency features)**
         feature_vector = [alpha_band, theta_band, beta_band, gamma_band, entropy_val, ...
                  mean_val', variance_val', skewness_val', kurtosis_val', ...
                  rms_val', zcr_val', mobility', complexity'];

% Store features and labels
     features = [features; feature_vector];
     if strcmp(conditions{i}, 'movement')
          labels = [labels; 1]; % movement
         elseif strcmp(conditions{i}, 'imagery')
          labels = [labels; 2]; % imagery
    end
      
    end 
end
% Compute the final average by dividing by the number of subjects
for i = 1:length(conditions)
    avg_psd{i} = avg_psd{i} / numSubjects;
end

% **Plot Averaged PSD for Each Condition**
figure('Name', 'Averaged PSD Across Subjects'); 
colors = {'r', 'g', 'b', 'm', 'k','c','y'}; % Colors for different electrodes

for i = 1:length(conditions)
    subplot(2,2,i); % Arrange subplots in a 2x2 grid
    hold on;
    for ch = 1:size(avg_psd{i}, 1)
        plot(freqs_used, 10*log10(avg_psd{i}(ch, :)), ...
             'Color', colors{mod(ch-1, length(colors)) + 1}, 'LineWidth', 1.5);
    end
    xlabel('Frequency (Hz)');
    ylabel('Power/Frequency (dB/Hz)');
    title(['Averaged PSD - ', conditions{i}]);
    legend(selectedElectrodes, 'Location', 'northeast');
    grid on;
    hold off;
end
%per

features_raw = features; % Save original features
% Initialize accuracy storage for groups
groupAccuracies = struct();

% Loop through each group
for f = 1:numel(fields)
    groupName = fields{f};
    groupIdx = groupIndices.(groupName);
    
    % Debugging: Print indices
    fprintf('Processing Group: %s, Indices: ', groupName);
    disp(groupIdx);
    
    % Ensure indices are valid positive integers
    adjustedIndices = groupIdx(groupIdx > 0 & groupIdx <= size(features, 2));

    if isempty(adjustedIndices)
        warning('Skipping Group %s: No valid indices after adjustment.', groupName);
        continue;
    end

    % Double-check if indices are within the feature matrix size
    if max(adjustedIndices) > size(features, 2)
        warning('Skipping Group %s: Indices exceed available feature count.', groupName);
        continue;
    end

    % Attempt feature selection safely
    try
        groupFeatures = features(:, adjustedIndices);
    catch ME
        warning('Skipping Group %s due to index error: %s', groupName, ME.message);
        continue;
    end
% Step 5: Divide Data into 70% Training and 30% Testing Sets**
rng(1); % Set random seed for reproducibility
numSamples = size(features, 1);
idx = randperm(numSamples);
numTrain = round(0.7 * numSamples);
trainIdx = idx(1:numTrain);
testIdx = idx(numTrain+1:end);

trainFeatures = features(trainIdx, :);
trainLabels   = labels(trainIdx);
testFeatures  = features(testIdx, :);
testLabels    = labels(testIdx);
% Your further processing here (e.g., classification)
   
    trainGroupFeatures = groupFeatures(trainIdx, :);
    testGroupFeatures = groupFeatures(testIdx, :);
    
    % Train an SVM model for this electrode group
    groupModel = fitcecoc(trainGroupFeatures, trainLabels, ...
        'Coding', 'onevsall', 'Learners', templateSVM('KernelFunction', 'linear', 'Standardize', true));
    
    % Predict and calculate accuracy
    groupPredictions = predict(groupModel, testGroupFeatures);
    groupAccuracies.(groupName) = mean(groupPredictions == testLabels) * 100;
end
    
% Display accuracies for each group

disp('Accuracy per electrode group:');
disp(groupAccuracies);

% Extract electrode group names and accuracies
groupNames = fieldnames(groupAccuracies);
accuracies = cellfun(@(g) groupAccuracies.(g), groupNames);

% Create Bar Plot
figure;
bar(accuracies);

% Customize Plot
set(gca, 'XTickLabel', groupNames, 'XTickLabelRotation', 45); % Rotate x-axis labels
xlabel('Brain Areas (Brodmann Regions)');
ylabel('SVM Classification Accuracy (%)');
title('Comparison of Classification Accuracy per Brain Area');
ylim([0 100]); % Set y-axis limits
grid on;

% Display values on top of bars
for i = 1:length(accuracies)
    text(i, accuracies(i) + 2, sprintf('%.2f%%', accuracies(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold');
end
figure;
boxplot(accuracies, groupNames);
xlabel('Brain Areas (Brodmann Regions)');
ylabel('SVM Classification Accuracy (%)');
title('Accuracy Distribution Across Brain Areas');
grid on;

% Extract accuracy values for Brodmann areas
areaNames = fieldnames(groupAccuracies);  % Get Brodmann area names
accuracyValues = cellfun(@(x) groupAccuracies.(x), areaNames);  % Convert struct to array

% Define X and Y positions for each Brodmann area (manually or systematically)
numAreas = length(areaNames);
[X, Y] = meshgrid(1:numAreas, 1:numAreas);  % Create grid coordinates
Z = reshape(accuracyValues, [numAreas, 1]);  % Reshape for plotting
Z = repmat(Z, 1, numAreas);  % Expand into a matrix for surf plot

% Surface plot
figure;
surf(X, Y, Z);

% Customize appearance
colormap(jet);  % Use a color gradient
colorbar;       % Show color scale
shading interp; % Smooth color transitions
xlabel('Brodmann Area Index');
ylabel('Electrode Groups');
zlabel('Accuracy (%)');
title('3D Surface Plot of Accuracy per Brodmann Area');

% Add text labels to axes
xticks(1:length(areaNames));
xticklabels(areaNames);
yticks(1:length(areaNames));
yticklabels(areaNames);


%pca
[coeff, features_pca, ~, ~, explained] = pca(features);
cumulative_variance = cumsum(explained);
numComponents = find(cumulative_variance > 95, 1); % Keep 95% variance
features = features_pca(:, 1:numComponents); % Reduce feature size

% Apply Min-Max Normalization (Scale to 0-1 range)
features = (features - min(features)) ./ (max(features) - min(features));

% Convert labels to categorical format
labels = categorical(labels);


% Step 5: Divide Data into 70% Training and 30% Testing Sets**
rng(1); % Set random seed for reproducibility
numSamples = size(features, 1);
idx = randperm(numSamples);
numTrain = round(0.7 * numSamples);
trainIdx = idx(1:numTrain);
testIdx = idx(numTrain+1:end);

trainFeatures = features(trainIdx, :);
trainLabels   = labels(trainIdx);
testFeatures  = features(testIdx, :);
testLabels    = labels(testIdx);

                      % Step 6: Train SVM Model

% Define Kernel Types to Test
kernels = {'linear', 'polynomial', 'rbf'};
svm_results = struct(); % Store results
svm_models = struct(); % Store SVM models

for i = 1:length(kernels)
    kernelType = kernels{i};
    
    % Train SVM Model with the Given Kernel
    if strcmp(kernelType, 'polynomial')
        model = fitcecoc(trainFeatures, trainLabels, ...
            'Coding', 'onevsall', 'Learners', templateSVM('KernelFunction', kernelType, 'PolynomialOrder', 3, 'Standardize', true));
    else
        model = fitcecoc(trainFeatures, trainLabels, ...
            'Coding', 'onevsall', 'Learners', templateSVM('KernelFunction', kernelType, 'Standardize', true));
    end
    
    % Predict on Test Data
    predicted_labels = predict(model, testFeatures);
    
    % Compute Accuracy
    accuracy = mean(predicted_labels == testLabels) * 100;
    svm_results.(kernelType) = accuracy;
    svm_models.(kernelType) = model; % Store model
    
    % Display Results
    fprintf('SVM Kernel: %s, Accuracy: %.2f%%\n', kernelType, accuracy);
end

% Find the Best Kernel
[bestAcc, idx] = max(struct2array(svm_results));
bestKernel = kernels{idx};
bestSVMModel = svm_models.(bestKernel);

fprintf('\nBest SVM Kernel: %s with Accuracy: %.2f%%\n', bestKernel, bestAcc);

% Predict using the best SVM model
predicted_svm = predict(bestSVMModel, testFeatures);

% Train k-NN Model
knnModel = fitcknn(trainFeatures, trainLabels, 'NumNeighbors', 5);
predicted_knn = predict(knnModel, testFeatures);
knn_accuracy = mean(predicted_knn == testLabels) * 100;

disp(['k-NN Accuracy: ', num2str(knn_accuracy), '%']);

% Confusion Matrices
figure;
confusionchart(testLabels, predicted_svm, 'RowSummary', 'row-normalized', 'ColumnSummary', 'column-normalized');
title(['Confusion Matrix for Best SVM Kernel (', bestKernel, ')']);

figure;
confusionchart(testLabels, predicted_knn, 'RowSummary', 'row-normalized', 'ColumnSummary', 'column-normalized');
title('Confusion Matrix for k-NN');


% Calculate accuracy for each class
%accuracies_per_class = zeros(1, length(classNames)); 

%for i = 1:length(classNames)
    %classIdx = testLabels == classNames(i);  % Ensure classNames is used correctly
    %predicted_class = predicted_labels(classIdx);
    %actual_class = testLabels(classIdx);
    %accuracies_per_class(i) = sum(predicted_class == actual_class) / numel(actual_class);
%end


% Step 7: Train Random Forest Model**
numTrees = 100; % Number of trees
rfModel = TreeBagger(numTrees, trainFeatures, trainLabels, ...
    'OOBPrediction', 'On', 'Method', 'classification');

% Evaluate Random Forest Model Performance
predicted_rf = predict(rfModel, testFeatures);
predicted_rf = categorical(predicted_rf); % Convert to categorical for comparison
rf_accuracy = sum(predicted_rf == testLabels) / numel(testLabels);
disp(['Random Forest Accuracy: ', num2str(rf_accuracy * 100), '%']);

classNames = unique(testLabels); % Automatically finds unique class labels
 % Ensures correct comparisons

%classNames = [1, 2]; % 1 = Movement, 2 = Imagery

% Initialize accuracy storage
accuracies_per_class = zeros(1, length(classNames)); 

%accuracies_per_class = zeros(1, length(classNames)); 

for i = 1:length(classNames)
    classIdx = testLabels == classNames(i);  % Ensure classNames is used correctly
    predicted_class = predicted_labels(classIdx);
    actual_class = testLabels(classIdx);
    %accuracies_per_class(i) = sum(predicted_class == actual_class) / numel(actual_class);
    if numel(actual_class) > 0 % Prevent division by zero
        accuracies_per_class(i) = sum(predicted_class == actual_class) / numel(actual_class);
    else
        accuracies_per_class(i) = NaN; % Handle cases where no data exists
    end
end

% Convert to percentage
accuracies_per_class = accuracies_per_class * 100; 

% Display results with proper class names
classNamesStr = ["Movement", "Imagery"]; % Define readable labels
disp("Accuracy per class (%):");
disp(array2table(accuracies_per_class, 'VariableNames', cellstr(classNamesStr)));

% Plot accuracy per class
figure;
bars = bar(accuracies_per_class, 'FaceColor', [1 0.5 0]);

%bar(accuracies_per_class); % Already in percentage
xticks(1:length(classNamesStr)); 
xticklabels(classNamesStr); % Use readable labels
ylabel('Accuracy (%)');
title('Classification Accuracy per Class');
grid on;







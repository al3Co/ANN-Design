clc
clear
close all

load('resultsANN2.mat')

% rearrange table sorting the accuracy
[rClass,indClass] = sortrows(rClassANN,9,'descend');
k = 2;  % k = 2 means to plot from class_Results

% prepare data (Accuracy - Time)
data = [rClass.Acc rClass.Time];
[m,nT] = size(data);

for n = 1:nT
    %function to get values to bar plot
    [movCat, accM] = getBarData(rClass.DeepLearn, data(:,n));
    [kindCat, accK] = getBarData(rClass.batch_size, data(:,n));
    [unitCat, accU] = getBarData(rClass.units, data(:,n));
    [epoCat, accE] = getBarData(rClass.epochs, data(:,n));
    % plot data
    plotResults(movCat, accM, kindCat, accK, unitCat, accU, epoCat, accE, n, k);
    hold on
end
fprintf('Qty. of Networks: %d different networks.\n',m);
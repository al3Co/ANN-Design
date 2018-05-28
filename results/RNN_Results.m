% RNN_Results
clc
clear
close all

load('resultsANN2.mat')

% rearrange table sorting the accuracy
[rANNr,indANN] = sortrows(rRNN,11,'descend');
k = 3;  % k = 3 means to plot from RNN_Results 

% prepare data (Accuracy - Time)
data = [rANNr.Acc rANNr.Time];
[m,nT] = size(data);

for n = 1:nT
    %function to get values to bar plot
    [movCat, accM] = getBarData(rANNr.Movement, data(:,n));
    [kindCat, accK] = getBarData(rANNr.Kind, data(:,n));
    [unitCat, accU] = getBarData(rANNr.units, data(:,n));
    [epoCat, accE] = getBarData(rANNr.epochs, data(:,n));
    % plot data
    plotResults(movCat, accM, kindCat, accK, unitCat, accU, epoCat, accE, n, k);
    hold on
end
fprintf('Qty. of Networks: %d different networks.\n',m);

load('/home/disam/Documents/GitHub/FlexSensor/WorkSpaces/sync250418/allMovements.mat')
allM = allMovements;
clear allMovements;

flexS = [allM.A0 allM.A1 allM.A2 allM.A3 allM.A4 allM.A5 allM.A6 allM.A7...
        allM.A8 allM.A9 allM.A10 allM.A11 allM.A12 allM.A13 allM.A14];
names = {'A0' 'A1' 'A2' 'A3' 'A4' 'A5' 'A6' 'A7' 'A8' 'A9' 'A10' 'A11'...
        'A12' 'A13' 'A14'};
[coeff,score,latent,tsquared,explained,mu] = pca(flexS,'VariableWeights','variance');

Xcentered = score*coeff';

biplot(coeff(:,1:2),'scores',score(:,1:2),'varlabels',names);

% A0 = [0.027103; -0.0297; -0.00146];
% A1 = [-0.0249; 0.02498; -0.003953];
% A2 = [-0.00359; 0.0599; 0.0544];
% A3 = [-0.03087; -0.00632; -0.04124];
% A4 = [0.016596; -0.01542; 0.0299];
% A5 = [0.016665; -0.0083995; 0.019626];
% A6 = [0.04608; 0.03377; 0.04028];
% A7 = [0.014664; -0.0134; -0.004481];
% A8 = [-0.031389; -0.00726; 0.03168];
% A9 = [0.02222; 0.00789; 0.005937];
% A10 = [-0.0073625; -0.026465; 0.00326223];
% A11 = [-0.014221; -0.02208; 0.03606];
% A12 = [-0.016203; -0.035678; -0.003414];
% A13 = [0.014085; -0.00084229; -0.015404];
% A14 = [-0.010591; 0.001871; 0.0034823];
% T = table(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14);
% writetable(T,'PCA_SensorsData.csv','Delimiter',',','QuoteStrings',true)
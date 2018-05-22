function plotResults(movCat, accM, kindCat, accK, unitCat, accU, epoCat, accE, n, k)
    % function to plot data, value n is the figure number
    figure(n)
    
    tit = 'Accuracy';
    t1 = 'Movement';
    t2 = 'Sensor';
    
    if n == 2; tit = 'Time (s)'; end
    if k == 2; t1 = 'Network'; t2 = 'Batch size'; end
    
    subplot(2,2,1)
    bar(movCat, accM)
    ylabel(tit)
    title(t1)

    subplot(2,2,2)
    bar(kindCat, accK)
    ylabel(tit)
    title(t2)

    subplot(2,2,3)
    bar(unitCat, accU)
    ylabel(tit)
    title('Units')

    subplot(2,2,4)
    bar(epoCat, accE)
    ylabel(tit)
    title('Epochs')
end
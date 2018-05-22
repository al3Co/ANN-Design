function [gL, acc] = getBarData(value, AccVector)
    % function to obtain the first element of "value" with the corresponding value of the vector
    [movVal, gN, gL] = grp2idx(value);
    acc = [];
    increment = 1;
    for element = 1:length(gN)
        for val = 1:length(AccVector)
            if string(value(val)) == string(gN(element))
                acc(increment,1) = AccVector(val);
                increment = increment +1;
                break
            end
        end

    end
end
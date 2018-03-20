function labels = update_labels(y, K)
    %
    % Takes the vector of inputs y and converts it into the format suitable for
    % the neural network.
    %
    
    l = length(y);
    labels = zeros(l, K);

    for i = 1:l
        labels(i, y(i)) = 1;
    end
end

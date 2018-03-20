function [X_norm, mu, sigma] = normalize(X)
    %
    % Normalizes the features of X
    % @return X_norm: the normalized dataset
    % @return mu: the mean of each column in the dataset
    % @return sigma: the standard deviation of each column in the dataset
    %

    mu = mean(X);
    sigma = std(X);
    X_norm = (X - mu) ./ sigma;

end

function [X_norm, mu, sigma] = normalize(X)
    %
    % Normalizes the features of X
    % @return X_norm: the normalized dataset
    % @return mu: the mean of each column in the dataset
    % @return sigma: the standard deviation of each column in the dataset
    %

    sigma_epsilon = 0.000001;
    mu = mean(X);
    sigma = max(std(X), sigma_epsilon);
    X_norm = (X - mu) ./ sigma;

end

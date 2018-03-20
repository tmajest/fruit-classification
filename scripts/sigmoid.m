function g = sigmoid(z)
    %
    % Computes the sigmoid function of the matrix z.
    %

    g = 1.0 ./ (1.0 + exp(-z));

end

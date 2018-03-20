function g = sigmoid_gradient(z)
    %
    % Calculates the sigmoid gradient function for the input z.
    %

    s = sigmoid(z);
    g = s .* (1 - s);

end

function J = nn_cost_function(hypothesis, y, Theta1, Theta2, lambda)
    %
    % Computes the cost function for the neural network.
    %

    p1 = -y .* log(hypothesis);
    p2 = (1 - y2) .* log(1 - a3);

    % Calculate regularization terms
    reg1 = sum(sum(Theta1(:, 2:end) .^ 2));
    reg2 = sum(sum(Theta2(:, 2:end) .^ 2));
    reg = (lambda / (2 * m)) * (reg1 + reg2);

    J = (sum(sum(p1 - p2)) / m) + reg;

end

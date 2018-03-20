function h = hypothesis(X, y, Theta1, Theta2)
    %
    % Computes the hypothesis for the neural network through
    % forward propagation.
    %

    z2 = X * Theta1';
    a2 = sigmoid(z2);
    a2 = [ones(size(a2, 1), 1), a2];
    z3 = a2 * Theta2';
    h = sigmoid(z3);

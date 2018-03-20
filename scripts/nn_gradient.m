function [grad, hypothesis] = nn_gradient(X, y, Theta1, Theta2)
    %
    % Computes the gradient for the neural network.
    %

    % Forward propagation
    z2 = X * Theta1';
    a2 = sigmoid(z2);
    a2 = [ones(size(a2, 1), 1), a2];
    z3 = a2 * Theta2';
    a3 = sigmoid(z3);

    % Backwards propagation to compute gradients
    d3 = a3 - y;
    d2 = (d3 * Theta2(:, 2:end)) .* sigmoid_gradient(z2);
    grad1 = (d2' * X) / m;
    grad2 = (d3' * a2) / m;

    % Add regularization to gradients
    temp = Theta1;
    temp(:, 1) = zeros(size(temp, 1), 1);
    grad1 = grad1 + ((lambda / m) * temp1);

    temp = Theta2;
    temp(:, 1) = zeros(size(temp, 1), 1);
    grad2 = grad2 + ((lambda / m) * temp);

    grad = [grad1(:); grad2(:)];
    hypothesis = a3;

function [J grad] = nn_compute(...
    theta_vec, ...
    input_layer_size, ...
    hidden_layer_size, ...
    num_labels, ...
    X, y, lambda)
    
    %
    % Computes the cost function and gradient for the neural network.
    %

    % Extract theta1 and theta2
    [Theta1, Theta2] = unroll_theta(
        theta_vec, 
        input_layer_size, 
        hidden_layer_size, 
        num_labels);

    m = size(X, 1);

    % Compute gradient
    %----------------------------------------------------------

    % Forward propagation
    z2 = X * Theta1';
    a2 = sigmoid(z2);
    a2 = [ones(size(a2, 1), 1), a2];
    z3 = a2 * Theta2';
    a3 = sigmoid(z3);

    % Backwards propagation
    d3 = a3 - y;
    d2 = (d3 * Theta2(:, 2:end)) .* sigmoid_gradient(z2);
    grad1 = (d2' * X) / m;
    grad2 = (d3' * a2) / m;

    % Add regularization to gradients
    temp = Theta1;
    temp(:, 1) = zeros(size(temp, 1), 1);
    grad1 = grad1 + ((lambda / m) * temp);

    temp = Theta2;
    temp(:, 1) = zeros(size(temp, 1), 1);
    grad2 = grad2 + ((lambda / m) * temp);

    grad = [grad1(:); grad2(:)];

    % Compute cost function
    %----------------------------------------------------------
    p1 = -y .* log(a3);
    p2 = (1 - y) .* log(1 - a3);

    % Calculate regularization terms
    reg1 = sum(sum(Theta1(:, 2:end) .^ 2));
    reg2 = sum(sum(Theta2(:, 2:end) .^ 2));
    reg = (lambda / (2 * m)) * (reg1 + reg2);

    J = (sum(sum(p1 - p2)) / m) + reg;
end

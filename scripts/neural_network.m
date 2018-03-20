
function [Theta1, Theta2, cost, mu, sigma] = neural_network(X, y, lambda, K)
    %
    % Trains the neural network.
    %

    num_neurons_1 = size(X, 2);
    num_neurons_2 = 50;
    num_neurons_3 = K;

    [X_norm, mu, sigma] = normalize(X);
    X_norm = [ones(size(X_norm, 1), 1), X_norm];
    y2 = update_labels(y, K);
    
    Theta1 = random_theta(num_neurons_1, num_neurons_2);
    Theta2 = random_theta(num_neurons_2, num_neurons_3);
    init_theta_vec = [Theta1(:); Theta2(:)];

    fflush(stdout);

    compute_func = @(t) nn_compute(
        t, ...
        num_neurons_1,
        num_neurons_2,
        K,
        X_norm, y2, lambda);

    options = optimset('GradObj', 'on', 'MaxIter', 50);
    [theta_vec, cost] = fmincg(compute_func, init_theta_vec, options);

    [Theta1, Theta2] = unroll_theta(
        theta_vec, 
        num_neurons_1,
        num_neurons_2,
        K);

end

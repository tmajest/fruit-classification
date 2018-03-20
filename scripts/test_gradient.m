function [errors] = test_gradient()
    %
    % Run a test to calculate if the computed gradient is roughly equal to
    % the numerical gradient.
    %
    
    % Initialize test
    il = 10;
    hl = 20;
    nl = 15;
    m = 100;
    lambda = 1;

    Theta1 = random_theta(il, hl);
    Theta2 = random_theta(hl, nl);

    X = randi(20, m, il);
    X2 = [ones(size(X, 1), 1) X];
    y = (mod(1:m, nl) + 1)';
    y2 = update_labels(y, nl);

    theta_vec = [Theta1(:); Theta2(:)];
    cost_func = @(t)nn_compute(t, il, hl, nl, X2, y2, lambda);

    % Check gradient
    [cost grad] = cost_func(theta_vec);
    numgrad = computeNumericalGradient(cost_func, theta_vec);
    diff = norm(numgrad - grad) / norm(numgrad + grad);

    disp([numgrad grad]);
    fprintf('Left: numerical gradient. Right: computed gradient.\n');
    fprintf('Difference: %g', diff);

end

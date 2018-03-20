function [Theta1, Theta2, cost, mu, sigma] = run_nn(lambda)
    [X, y] = load_data('training');
    [Theta1, Theta2, cost, mu, sigma] = neural_network(X, y, lambda, 2);
end

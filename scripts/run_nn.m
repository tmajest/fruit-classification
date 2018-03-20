function [Theta1, Theta2, cost] = run_nn(lambda)
    display('running nn');
    [X, y] = load_data();

    display('Starting nn');
    [Theta1, Theta2, cost] = neural_network(X, y, lambda, 2);
end

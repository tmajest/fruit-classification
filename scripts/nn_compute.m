function [J gradient] = nn_compute(...
    theta_vec, ...
    input_layer_size, ...
    hidden_layer_size, ...
    num_labels, ...
    X, y, lambda)
    
    %
    % Computes the cost function and gradient for the neural network.
    %

    display('iteration of nn_compute');

    % Extract theta1 and theta2
    display('extracting theta');
    [Theta1, Theta2] = unroll_theta(
        theta_vec, 
        input_layer_size, 
        hidden_layer_size, 
        num_labels);

    m = size(X, 1);

    display('calculating gradient');
    [grad, hypothesis] = nn_gradient(X, y, Theta1, Theta2);

    fflush(stdout);

    display('calculating cost function');
    J = nn_cost_function(hypothesis, y, Theta1, Theta2, lambda);

    fflush(stdout);

end

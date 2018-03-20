function [Theta1, Theta2] = unroll_theta(...
    theta_vec, ...
    input_layer_size, ...
    hidden_layer_size, ...
    num_labels)

    %
    % Unrolls the theta vector into two Theta matrixes.
    %

    Theta1 = reshape(
        theta_vec(1:hidden_layer_size * (input_layer_size + 1)), ...
        hidden_layer_size,
        input_layer_size + 1);

    Theta2 = reshape(
        theta_vec((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
        num_labels,
        hidden_layer_size + 1);
end

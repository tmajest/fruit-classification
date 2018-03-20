
function Theta = random_theta(input_size, output_size)
    %
    % Initializes random small values for theta
    %

    epsilon_init = 0.12;
    Theta = rand(output_size, input_size + 1) * 2 * epsilon_init - epsilon_init;
end

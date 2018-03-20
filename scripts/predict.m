function p = predict(X, Theta1, Theta2)
    %
    % Predicts the values for X given Theta1 and Theta2.
    %
    m = size(X, 1);
    num_labels = size(Theta2, 1);

    h1 = sigmoid([ones(m, 1), X] * Theta1');
    h2 = sigmoid([ones(m, 1), h1] * Theta2');
    [_ p] = max(h2, [], 2);

end

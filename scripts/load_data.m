function [X, y] = load_data()
    %
    % Loads the Apple Braeburn and Banana images into a dataset X with features y.
    %

    % Constants
    IMAGE_WIDTH = 100;
    IMAGE_HEIGHT = 100;
    IMAGE_DEPTH = 3; % Each pixel has r, g, and b value
    APPLE_LABEL = 1;
    BANANA_LABEL = 2;

    % Load files
    apple_path = '../training/Apple Braeburn/';
    banana_path = '../training/Banana/';

    apple_files = dir([apple_path '*.jpg']);
    banana_files = dir([banana_path '*.jpg']);

    %num_apples = length(apple_files);
    %num_bananas = length(banana_files);
    num_apples = 10;
    num_bananas = 10;

    display('initializing x and y datasets');

    % Initialize dataset
    X = zeros(
        num_apples + num_bananas, ...
        IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_DEPTH);

    y = zeros(num_apples + num_bananas, 1);

    % Load apple images
    display('loading apples');
    for i = 1:num_apples
        filename = [apple_path apple_files(i).name];
        I = imread(filename);
        X(i, :) = I(:)';
        y(i) = APPLE_LABEL;
    end

    % Load banana images
    display('loading bananas');
    for i = 1:num_bananas
        filename = [banana_path banana_files(i).name];
        I = imread(filename);
        X(num_apples + i, :) = I(:)';
        y(num_apples + i) = BANANA_LABEL; 
    end
end

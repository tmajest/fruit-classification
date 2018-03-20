function show_fruit(X, index)
    image_data = uint(X(index, 1:30000));
    image_matrix = reshape(image_data, 100, 100, 3);
    imshow(image_matrix);
end

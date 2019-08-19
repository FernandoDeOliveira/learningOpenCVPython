def sliding_window(image, step, window_size):
    for y in range(0, image.shape[0], step):
        for x in range(0, image.shape[1], step):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

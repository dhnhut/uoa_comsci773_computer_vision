import numpy as np


def gaussian_filtering(image, image_width, image_height):
    # fixed Gaussian kernel
    gaussian_kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
    kernel_size = gaussian_kernel.shape[0]

    extended_image = np.pad(image, pad_width=1, mode="edge")

    filtered_image = np.zeros_like(image, dtype=np.float64)
    for i in range(image_height):
        for j in range(image_width):
            region = extended_image[i : i + kernel_size, j : j + kernel_size]
            filtered_image[i, j] = np.sum(region * gaussian_kernel)
    return filtered_image


def sobel_derivative(image, image_width, image_height, direction="x"):
    # 3x3 Sobel kernels
    g_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]) / 8
    g_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]) / 8

    kernel_size = len(g_x)

    extended_image = np.pad(image, pad_width=1, mode="edge")

    filtered_image = np.zeros_like(image, dtype=np.float64)
    for i in range(image_height):
        for j in range(image_width):
            region = extended_image[i : i + kernel_size, j : j + kernel_size]
            if direction == "x":
                filtered_image[i, j] = np.sum(region * g_x)
            else:
                filtered_image[i, j] = np.sum(region * g_y)
    return filtered_image


def compute_image_derivatives(image, image_width, image_height):
    I_x = sobel_derivative(image, image_width, image_height, direction="x")
    I_y = sobel_derivative(image, image_width, image_height, direction="y")
    return I_x**2, I_y**2, I_x * I_y


def compute_single_cornerness_score(ix_square, iy_square, ixiy, r, c, alpha):
    det = ix_square[r, c] * iy_square[r, c] - ixiy[r, c] ** 2
    trace = ix_square[r, c] + iy_square[r, c]
    return det - alpha * (trace**2)


def kernel_sum(image, image_width, image_height):
    kernel_size = 3
    extended_image = np.pad(image, pad_width=1, mode="edge")
    square_image = np.zeros_like(image, dtype=np.float64)
    for i in range(image_height):
        for j in range(image_width):
            region = extended_image[i : i + kernel_size, j : j + kernel_size]
            square_image[i, j] = np.sum(region)
    return square_image


def cornerness_score_matrix(
    ix_square, iy_square, ixiy, alpha, image_width, image_height
):
    sum_ix_square = kernel_sum(ix_square, image_width, image_height)
    sum_iy_square = kernel_sum(iy_square, image_width, image_height)
    sum_ixiy = kernel_sum(ixiy, image_width, image_height)

    cornerness_matrix = np.zeros_like(ix_square, dtype=np.float64)

    for r in range(image_height):
        for c in range(image_width):
            cornerness_matrix[r, c] = compute_single_cornerness_score(
                sum_ix_square, sum_iy_square, sum_ixiy, r, c, alpha
            )
    return cornerness_matrix


def compute_cornerness_score(
    ix_square, iy_square, ixiy, alpha, threshold, image_width, image_height
):
    top = 1000

    corners = (
        cornerness_score_matrix(
            ix_square, iy_square, ixiy, alpha, image_width, image_height
        )
        .flatten()
        .tolist()
    )
    corners = [x for x in corners if x > threshold]
    corners.sort(reverse=True)
    return corners[:top]  # Return top 1000 corners


def non_maximum_suppression(corner_response, image_width, image_height):
    nms_c = np.zeros_like(corner_response, dtype=np.float64)

    for r in range(image_height):
        for c in range(image_width):
            current_score = corner_response[r, c]
            if current_score == 0:
                continue  # Skip if the cornerness score is zero

            # Define the neighborhood (3x3 window)
            r_start = max(0, r - 1)
            r_end = min(image_height, r + 2)
            c_start = max(0, c - 1)
            c_end = min(image_width, c + 2)

            # Get the maximum score in the neighborhood
            neighborhood = corner_response[r_start:r_end, c_start:c_end]
            max_score = np.max(neighborhood)

            # If the current score is the maximum in the neighborhood, keep it; otherwise, set it to zero
            if current_score == max_score:
                nms_c[r, c] = current_score

    return nms_c

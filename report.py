import numpy as np


def compute_homo(matched_corner_pairs):
    A = np.zeros((2 * len(matched_corner_pairs), 9))

    for i, (pt1, pt2) in enumerate(matched_corner_pairs):
        x1, y1 = pt1
        x2, y2 = pt2

        A[2 * i] = [0, 0, 0, x1, y1, 1, -y2 * x1, -y2 * y1, -y2]
        A[2 * i + 1] = [x1, y1, 1, 0, 0, 0, -x2 * x1, -x2 * y1, -x2]

    print("Matrix A:")
    print(A)

    U, D, Vt = np.linalg.svd(A)
    h = Vt[-1]
    H = h.reshape(3, 3)
    return H


# # Assignment 1 Phase 3 (Theory) - Question 1
# # S1 = {(1, 1, 1), (2, 3, 1), (4, 2, 1), (3, 5, 1))}
# # S2 = {(3, 4, 1), (5, 8, 1), (9, 6, 1), (7, 12, 1)},
theory_pairs = np.array(
    [
        [[1, 1], [3, 4]],
        [[2, 3], [5, 8]],
        [[4, 2], [9, 6]],
        [[3, 5], [7, 12]],
    ]
)

print("Theory Homography:")
theory_H = compute_homo(theory_pairs)
print(theory_H)

# # Tutorial
# # S1 = {(2, 2, 1), (2, 4, 1), (6, 4, 1), (6, 2, 1))}
# # S2 = {(-2, -2, 1), (-1, -4, 1), (-6, -1, 1), (-6, -5, 1)},
tutorial_pairs = np.array(
    [
        [[2, 2], [-2, -2]],
        [[2, 4], [-1, -4]],
        [[6, 4], [-6, -1]],
        [[6, 2], [-6, -5]],
    ]
)
print("\nTutorial Homography:")
tutorial_H = compute_homo(tutorial_pairs)
print(tutorial_H)


A = np.array(
    [
        [0, 0, 0, 1, 1, 1, -4, -4, -4],
        [1, 1, 1, 0, 0, 0, -3, -3, -3],
        [0, 0, 0, 2, 3, 1, -16, -24, -8],
        [2, 3, 1, 0, 0, 0, -10, -15, -5],
        [0, 0, 0, 4, 2, 1, -24, -12, -6],
        [4, 2, 1, 0, 0, 0, -36, -18, -9],
        [0, 0, 0, 3, 5, 1, -36, -60, -12],
        [3, 5, 1, 0, 0, 0, -21, -35, -7],
    ]
)


print("Validate: Ah = 0")

windows_H = np.array(
    [
        [-0.04614434, 0.22572712, 0.42838206],
        [-0.30096949, 0.55579463, 0.55579463],
        [-0.05016158, 0.02508079, 0.22773574],
    ]
)

# coderunner_H = np.array(
#     [
#         [-0.33336481, -0.23483464, -0.33336481],
#         [0.17612598, -0.80303409, -0.04926509],
#         [0.01956955, -0.03913911, -0.20582151],
#     ]
# )

coderunner_H = np.array(
    [
        [-0.06505653, 0.23126432, 0.43000037],
        [-0.30835243, 0.55164832, 0.55164832],
        [-0.05139207, 0.02569604, 0.22443209],
    ]
)


print(np.dot(A, windows_H.flatten()))
print(np.dot(A, coderunner_H.flatten()))
print(np.dot(A, theory_H.flatten()))

print(np.allclose(np.dot(A, windows_H.flatten()), np.zeros(8)))
print(np.allclose(np.dot(A, coderunner_H.flatten()), np.zeros(8)))
print(np.allclose(np.dot(A, theory_H.flatten()), np.zeros(8)))

print(np.linalg.norm(np.dot(A, windows_H.flatten())))
print(np.linalg.norm(np.dot(A, coderunner_H.flatten())))
print(np.linalg.norm(np.dot(A, theory_H.flatten())))

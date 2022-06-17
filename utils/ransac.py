import numpy as np


def ransac_cylinder(points: np.array, num_iter: int = 10, thresh: float = 0.05, max_radius=0.4):
    """Cylinder model fitting on the points by RANSAC

    Parameters
    ----------
    points : `np.ndarray`
        points
    num_iter : `int`
        Number of iterations for RANSAC algorithm
    thresh : `float`
        Threshold of error allowed for inliers

    Returns
    -------
    (x_b, y_b, r_b) : `tuple`
        Estimated parameters (coordinate of the center and the radius)
    best_score : `int`
        Number of inliers
    """

    # For the designated number of times do
    x_b, y_b, r_b = 0, 0, 0
    best_score = -1
    points[:, 2] = 0
    for i in range(num_iter):
        # Sample point
        indices = np.random.randint(0, points.shape[0], 3)
        s = points[indices, :]

        # Calculate cylinder (circle) model
        x1, x2, x3 = s[0][0], s[1][0], s[2][0]
        y1, y2, y3 = s[0][1], s[1][1], s[2][1]

        x = ((y1 - y2)*(x3**2 - x1**2 + y3**2-y1**2) - (y1-y3)*(x2**2 -
             x1**2+y2**2-y1**2)) / (2*(x1-x2) * (y1-y3)-2*(x1-x3)*(y1-y2))
        y = ((x1 - x3)*(x2**2 - x1**2 + y2**2-y1**2) - (x1-x2)*(x3**2 -
             x1**2+y3**2-y1**2)) / (2*(x1-x2) * (y1-y3)-2*(x1-x3)*(y1-y2))

        r = np.sqrt((x - x1)**2 + (y - y1)**2)

        # Evaluate
        diffs = points[:, :2] - np.array([x, y])
        diffs = np.abs(np.linalg.norm(diffs, axis=1) - r)

        score = (diffs < thresh).sum()

        if score > best_score and r < max_radius:
            best_score = score
            x_b, y_b, r_b = x, y, r

    return (x_b, y_b, r_b), best_score

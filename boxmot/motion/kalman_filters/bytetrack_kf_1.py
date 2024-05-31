import numpy as np
import scipy.linalg

chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919
}

class KalmanFilter(object):
    def __init__(self):
        ndim, dt = 4, 1.

        # Create Extended Kalman filter model matrices.
        self._motion_mat = np.eye(ndim, 2 * ndim)

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

    def initiate(self, measurement):
        # Initialize the state with the provided measurement
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3]
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        # Run Extended Kalman filter prediction step
        state_size = len(mean)
        F = np.eye(state_size)
        F[:2, state_size//2:state_size//2+2] = np.eye(2)  # Identity matrix for position
        F[2, state_size//2+2] = 1.  # Identity matrix for aspect ratio
        F[3, state_size//2+3] = 1.  # Identity matrix for height

        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(mean, self._motion_mat.T)
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance
        
    def project(self, mean, covariance):
        # Project state distribution to measurement space
        H = np.zeros((4, len(mean)))
        H[:2, :2] = np.eye(2)  # Identity matrix for position
        H[2, 2] = 1.  # Identity matrix for aspect ratio
        H[3, 3] = 1.  # Identity matrix for height

        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]
        ]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(H, mean)
        covariance = np.linalg.multi_dot((H, covariance, H.T))
        return mean, covariance + innovation_cov

    def update(self, mean, covariance, measurement):
        # Run Extended Kalman filter correction step
        projected_mean, projected_cov = self.project(mean, covariance)

        K = np.linalg.multi_dot((covariance, self._update_mat.T, np.linalg.inv(projected_cov)))

        innovation = measurement - projected_mean
        mean = mean + np.dot(K, innovation)
        covariance = covariance - np.linalg.multi_dot((K, projected_cov, K.T))

        return mean, covariance

    def gating_distance(self, mean, covariance, measurements, only_position=False, metric='maha'):
        # Compute gating distance between state distribution and measurements
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        d = measurements - mean
        if metric == 'gaussian':
            return np.sum(d * d, axis=1)
        elif metric == 'maha':
            cholesky_factor = np.linalg.cholesky(covariance)
            z = scipy.linalg.solve_triangular(
                cholesky_factor, d.T, lower=True, check_finite=False, overwrite_b=True)
            squared_maha = np.sum(z * z, axis=0)
            return squared_maha
        else:
            raise ValueError('invalid distance metric')

    def multi_predict(self, mean, covariance):
        # Run Extended Kalman filter prediction step (Vectorized version)
        state_size = len(mean[0])
        F = np.eye(state_size)
        F[:2, state_size//2:state_size//2+2] = np.eye(2)  # Identity matrix for position
        F[2, state_size//2+2] = 1.  # Identity matrix for aspect ratio
        F[3, state_size//2+3] = 1.  # Identity matrix for height

        std_pos = [
            self._std_weight_position * mean[:, 3],
            self._std_weight_position * mean[:, 3],
            1e-2 * np.ones_like(mean[:, 3]),
            self._std_weight_position * mean[:, 3]
        ]
        std_vel = [
            self._std_weight_velocity * mean[:, 3],
            self._std_weight_velocity * mean[:, 3],
            1e-5 * np.ones_like(mean[:, 3]),
            self._std_weight_velocity * mean[:, 3]
        ]
        sqr = np.square(np.r_[std_pos, std_vel]).T

        motion_cov = []
        for i in range(len(mean)):
            motion_cov.append(np.diag(sqr[i]))
        motion_cov = np.asarray(motion_cov)

        mean = np.dot(mean, self._motion_mat.T)
        left = np.dot(self._motion_mat, covariance).transpose((1, 0, 2))
        covariance = np.dot(left, self._motion_mat.T) + motion_cov

        return mean, covariance



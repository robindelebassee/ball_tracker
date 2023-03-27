import numpy as np
import matplotlib.pyplot as plt


class KalmanFilter(object):
    def __init__(self, dt, a_x, a_y, x_measure_std, y_measure_std, acceleration_std):
        '''
        :param dt: time step
        :param a_x: acceleration wrt x direction
        :param a_y: acceleration wrt y direction
        :param x_measure_std: standard deviation of the measurement in x direction
        :param y_measure_std: standard deviation of the measurement in the y direction
        :param acceleration_std: standard deviation of the acceleration
        '''

        self.dt = dt
        self.acceleration = np.matrix([[a_x], [a_y]])

        # intiial state
        self.state = np.zeros((4, 1))

        # transition matrix
        self.transition_matrix = np.matrix(
            [
                [1, 0, self.dt, 0],
                [0, 1, 0, self.dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])

        # control matrix
        self.control_matrix = np.matrix(
            [
                [(self.dt**2)/2, 0],
                [0, (self.dt**2)/2],
                [self.dt, 0],
                [0, self.dt]
            ])

        # map state to measurement
        self.state_to_measurement = np.matrix([[1, 0, 0, 0],
                                               [0, 1, 0, 0]])

        # process covariance
        self.process_cov = np.matrix(
            [
                [(self.dt**4)/4, 0, (self.dt)**3/2, 0],
                [0, (self.dt)**4/4, 0, (self.dt)**3/2],
                [(self.dt)**3/2, 0, self.dt**2, 0],
                [0, (self.dt)**3/2, 0, self.dt**2]
            ]) * acceleration_std**2

        # measurment covariance
        self.measurement_cov = np.matrix([[x_measure_std**2, 0],
                                          [0, y_measure_std**2]])

        # covariance matrix
        self.cov = np.eye(4)

    def predict(self):

        # update state
        self.state = np.dot(self.transition_matrix, self.state) + \
            np.dot(self.control_matrix, self.acceleration)

        # error covariance
        self.cov = np.dot(np.dot(self.transition_matrix, self.cov),
                          self.transition_matrix.T) + self.process_cov

        return self.state[:2]

    def optimize(self, z):

        S = np.dot(self.state_to_measurement, np.dot(
            self.cov, self.state_to_measurement.T)) + self.measurement_cov

        # Kalman Gain
        K = np.dot(np.dot(self.cov, self.state_to_measurement.T),
                   np.linalg.inv(S))

        self.state = np.round(
            self.state + np.dot(K, (z - np.dot(self.state_to_measurement, self.state))))

        I = np.eye(4)

        self.cov = (I - (K * self.state_to_measurement)) * self.cov

        return self.state[:2]

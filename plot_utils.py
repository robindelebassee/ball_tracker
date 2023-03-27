import matplotlib.pyplot as plt
import numpy as np


def plot_trajectories(x_coord, y_coord):
    frames_nb = np.arange(len(x_coord))

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    ax[0].plot(frames_nb, x_coord, '.', color='m')
    ax[0].set_xlabel('frames')
    ax[0].set_ylabel('x-direction')

    ax[1].plot(frames_nb, y_coord, '.', color='c')
    ax[1].set_xlabel('frames')
    ax[1].set_ylabel('y-direction')

    fig.suptitle('Trajectories of the ball wrt to x,y direction', fontsize=15)
    plt.savefig('images/trajectories.png')


def plot_trajectories_kalman(x_coord, y_coord, tracked_pos):
    # predicted x positions
    pred_x_pos = np.array([round(tracked_pos[i][0])
                          for i in range(len(tracked_pos))])

    # predicted y positions
    pred_y_pos = np.array([round(tracked_pos[i][1])
                          for i in range(len(tracked_pos))])

    # Create an array of the frames
    frames = np.arange(len(tracked_pos))

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    ax[0].plot(x_coord, frames, '.', color='r', label='Result from tracker')
    ax[0].plot(pred_x_pos, frames, '.', color='m', label="Estimated by Kalman")

    ax[1].plot(y_coord, frames, '.', color='g', label='Result from tracker')
    ax[1].plot(pred_y_pos, frames,  '.', color='c',
               label="Estimated by Kalman")

    ax[0].set_xlabel('frames')
    ax[0].set_ylabel('x-direction')
    ax[1].set_xlabel('frames')
    ax[1].set_ylabel('y-direction')

    ax[0].legend(loc="upper left")
    ax[1].legend(loc="upper left")

    fig.suptitle(
        'Trajectories of the ball wrt to x,y direction - Kalman filter results', fontsize=15)
    plt.savefig('images/trajectories_kalman.png')

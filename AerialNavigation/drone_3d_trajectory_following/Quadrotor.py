"""
Class for plotting a quadrotor

Author: Daniel Ingram (daniel-s-ingram)
"""

from math import cos, sin
import numpy as np
import matplotlib.pyplot as plt

class Quadrotor():
    def __init__(self, x=0, y=0, z=0, roll=0, pitch=0, yaw=0, size=0.25, show_animation=True):
        # Define points representing the quadrotor's structure in its local coordinate frame
        self.p1 = np.array([size / 2, 0, 0, 1]).T
        self.p2 = np.array([-size / 2, 0, 0, 1]).T
        self.p3 = np.array([0, size / 2, 0, 1]).T
        self.p4 = np.array([0, -size / 2, 0, 1]).T

        # Initialize lists to store the quadrotor's trajectory data
        self.x_data = []
        self.y_data = []
        self.z_data = []
        self.show_animation = show_animation

        if self.show_animation:
            plt.ion()  # Enable interactive mode for real-time plotting
            fig = plt.figure()
            # Add a key event to stop the simulation with the escape key
            fig.canvas.mpl_connect('key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])

            self.ax = fig.add_subplot(111, projection='3d')  # Create a 3D subplot

        # Set the initial pose of the quadrotor
        self.update_pose(x, y, z, roll, pitch, yaw)

    def update_pose(self, x, y, z, roll, pitch, yaw):
        # Update the pose attributes
        self.x = x
        self.y = y
        self.z = z
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw
        # Append the new position to the trajectory data
        self.x_data.append(x)
        self.y_data.append(y)
        self.z_data.append(z)

        if self.show_animation:
            self.plot()  # Plot the updated pose

    def transformation_matrix(self):
        # Compute the transformation matrix based on current pose
        x = self.x
        y = self.y
        z = self.z
        roll = self.roll
        pitch = self.pitch
        yaw = self.yaw
        return np.array(
            [[cos(yaw) * cos(pitch), -sin(yaw) * cos(roll) + cos(yaw) * sin(pitch) * sin(roll), sin(yaw) * sin(roll) + cos(yaw) * sin(pitch) * cos(roll), x],
             [sin(yaw) * cos(pitch), cos(yaw) * cos(roll) + sin(yaw) * sin(pitch)
              * sin(roll), -cos(yaw) * sin(roll) + sin(yaw) * sin(pitch) * cos(roll), y],
             [-sin(pitch), cos(pitch) * sin(roll), cos(pitch) * cos(roll), z]
             ])

    def plot(self):  # pragma: no cover
        T = self.transformation_matrix()  # Get the transformation matrix

        # Transform the points to the global coordinate frame
        p1_t = np.matmul(T, self.p1)
        p2_t = np.matmul(T, self.p2)
        p3_t = np.matmul(T, self.p3)
        p4_t = np.matmul(T, self.p4)

        plt.cla()  # Clear the current axes

        # Plot the transformed points
        self.ax.plot([p1_t[0], p2_t[0], p3_t[0], p4_t[0]],
                     [p1_t[1], p2_t[1], p3_t[1], p4_t[1]],
                     [p1_t[2], p2_t[2], p3_t[2], p4_t[2]], 'k.')

        # Draw lines representing the quadrotor's arms
        self.ax.plot([p1_t[0], p2_t[0]], [p1_t[1], p2_t[1]],
                     [p1_t[2], p2_t[2]], 'r-')
        self.ax.plot([p3_t[0], p4_t[0]], [p3_t[1], p4_t[1]],
                     [p3_t[2], p4_t[2]], 'r-')

        # Plot the trajectory of the quadrotor
        self.ax.plot(self.x_data, self.y_data, self.z_data, 'b:')

        # Set the plot limits
        plt.xlim(-5, 5)
        plt.ylim(-5, 5)
        self.ax.set_zlim(0, 10)

        plt.pause(0.001)  # Pause briefly to update the plot

import numpy as np
import math
from math import pi, sqrt, cos, sin
import random
import pyglet
from pyglet.gl import *
import itertools
import matplotlib.pyplot as plt


def main():
    window = SimWindow()
    glClearColor(.25, .25, .25, 1.0)
    pyglet.clock.schedule_interval(window.tick, window.sim_dt)
    pyglet.app.run()

    # Plot data
    velocity = np.array(window.velocity_history)
    velocity = velocity.T
    for i in range(len(velocity)):
        plt.plot(range(window.run_interval), velocity[i])
    plt.xlabel("Iterations")
    plt.ylabel("Velocity")
    plt.savefig("velocity_plot.png", bbox_inches='tight')
    plt.show()

    connectivity = np.array(window.connectivity_history)
    plt.plot(range(window.run_interval), connectivity)
    plt.xlabel("Iterations")
    plt.ylabel("Connectivity")
    plt.savefig("connectivity_plot.png", bbox_inches='tight')
    plt.show()


class SimWindow(pyglet.window.Window):
    def __init__(self):
        super(SimWindow, self).__init__(720, 720)
        self.number_of_sensor_nodes = 7
        self.run_interval = 10000
        self.neighbor_distance = 15.0
        self.sensor_range = self.neighbor_distance * 1.2
        self.obstacle_flag = False
        self.target_path = ''
        self.target_start_position = np.matrix((0, 0), np.float64).T
        # Non-Case Control Variables
        self.sim_dt = 1 / 60
        self.run_flag = True
        self.scale_flag = False
        self.current_scale = 1
        self.frame_number = 0
        self.sensor_nodes = []
        self.obstacle_nodes = []
        self.adjacency_matrix = np.zeros((self.number_of_sensor_nodes, self.number_of_sensor_nodes), np.float64)
        self.position_history = []
        self.center_mass_history = []
        self.velocity_history = []
        self.target_history = []
        self.connectivity_history = []
        self.number_of_obstacles = 0
        # Set variables for different Cases
        if self.target_path == '':
            self.target_start_position = np.matrix((0, 0), np.float64).T
        self.target_node = TargetNode(q=self.target_start_position)
        if self.obstacle_flag:
            if self.target_path == '':
                self.number_of_obstacles = 2
            elif self.target_path == 'circle':
                self.number_of_obstacles = 7
            elif self.target_path == 'wave':
                self.number_of_obstacles = 3
        self.beta_matrix = np.zeros((self.number_of_sensor_nodes, self.number_of_obstacles), np.float64)
        # Generate random sensor starting position
        for i in range(self.number_of_sensor_nodes):
            x = random.uniform(-150.0, 150.0)
            y = random.uniform(-150.0, 150.0)
            # Multiplied by 4 to scale to 180x180
            sensor_node = SensorNode(id=i, sensor_range=self.sensor_range * 4,
                                     neighbor_distance=self.neighbor_distance * 4,
                                     q=np.matrix([x, y]).T, target=self.target_node)
            self.sensor_nodes.append(sensor_node)
        self.populate_adjacency_matrix()

    def tick(self, dt):
        self.frame_number += 1
        if self.run_flag:
            # create and update obstacles
            if self.frame_number == 1 and self.number_of_obstacles > 0:
                self.spawn_obstacles()
            for i in range(self.number_of_obstacles):
                self.obstacle_nodes[i].update()
            self.populate_beta_matrix()

            # update target node
            self.target_node.update(self.target_path, dt=self.sim_dt)
            self.target_history.append(np.array(self.target_node.q))

            # update sensor node and record history
            position_holder = []
            velocity_holder = []
            for i in range(self.number_of_sensor_nodes):
                self.sensor_nodes[i].update(self.sim_dt)
                position_holder.append(np.array(self.sensor_nodes[i].q))
                velocity_holder.append(np.linalg.norm(self.sensor_nodes[i].p))
            self.populate_adjacency_matrix()

            self.position_history.append(position_holder)
            self.velocity_history.append(velocity_holder)
            self.connectivity_history.append(self.connectivity())

            center_mass = self.get_center_mass()

            # Draw frame
            self.clear()
            glPushMatrix()

            # Transform origin to center
            glTranslatef(self.width / 2.0, self.height / 2.0, 0.0)

            # check if window needs to be scaled and scale if needed
            self.window_fit_flock()
            if self.scale_flag:
                self.current_scale = self.current_scale / 2
                self.scale_flag = False
            glScalef(self.current_scale, self.current_scale, 0.0)

            # Draw sensor ranges
            for i in range(self.number_of_sensor_nodes):
                self.sensor_nodes[i].draw_sensor_range()

            # Draw node, center of mass, and target history
            ## self.draw_node_history(self.position_history, self.current_scale)
            ## self.draw_center_history(self.center_mass_history, self.current_scale)
            ## self.draw_target_history(self.target_history, self.current_scale)

            # Draw obstacle
            for i in range(self.number_of_obstacles):
                self.obstacle_nodes[i].draw_obstacle()

            # Draw neighbor connections
            self.draw_neighbor_connections()

            # Draw target
            self.target_node.draw_target_node(self.current_scale)

            # Draw center of mass
            self.center_mass_history.append(center_mass)
            self.draw_center_mass(center_mass, self.current_scale)

            # Draw sensor nodes
            for i in range(self.number_of_sensor_nodes):
                self.sensor_nodes[i].draw_node(self.current_scale)

            glPopMatrix()

            if self.frame_number >= self.run_interval:
                self.run_flag = False
            # save frame
            ## save_buffer = pyglet.image.get_buffer_manager().get_color_buffer()
            ## save_buffer.save('frames/screenshot_{}.png'.format(str(self.frame_number).zfill(3)))
        else:
            pyglet.app.exit()

    def spawn_obstacles(self):
        if self.target_path == '':
            self.obstacle_nodes.append(CircularObstacle(np.matrix((200.0, 200.0), np.float64).T))
            self.obstacle_nodes.append(CircularObstacle(np.matrix((400.0, 400.0), np.float64).T))
        elif self.target_path == 'circle':
            self.obstacle_nodes.append(CircularObstacle(np.matrix((800.0, 0.0), np.float64).T, radius=25))
            self.obstacle_nodes.append(CircularObstacle(np.matrix((600.0, 0.0), np.float64).T, radius=80))
            self.obstacle_nodes.append(CircularObstacle(np.matrix((0.0, 800.0), np.float64).T))
            self.obstacle_nodes.append(CircularObstacle(np.matrix((0.0, 600.0), np.float64).T))
            self.obstacle_nodes.append(CircularObstacle(np.matrix((0.0, -800.0), np.float64).T, radius=80))
            self.obstacle_nodes.append(CircularObstacle(np.matrix((0.0, -600.0), np.float64).T, radius=25))
            self.obstacle_nodes.append(CircularObstacle(np.matrix((-700.0, 0.0), np.float64).T, radius=100))
        elif self.target_path == 'wave':
            self.obstacle_nodes.append(CircularObstacle(np.matrix((250.0, -50.0), np.float64).T))
            self.obstacle_nodes.append(CircularObstacle(np.matrix((450.0, 50.0), np.float64).T))
            self.obstacle_nodes.append(CircularObstacle(np.matrix((650.0, -50.0), np.float64).T))

    def draw_center_history(self, center_history, scale_factor=1.0):
        if len(center_history) > 0:
            if scale_factor < 1.0:
                scale_factor += scale_factor ** 1.1
            for node in center_history:
                glPushMatrix()
                glTranslatef(node[0], node[1], 0.0)
                draw_circle(4 * (1 / scale_factor), [50, 255, 50, 255])
                glPopMatrix()

    def draw_node_history(self, node_history, scale_factor=1.0):
        if scale_factor < 1.0:
            scale_factor += scale_factor ** 1.1
        for time_step in node_history:
            for node in time_step:
                glPushMatrix()
                glTranslatef(node[0], node[1], 0.0)
                draw_circle(2 * (1 / scale_factor), [255, 50, 255, 75])
                glPopMatrix()

    def draw_target_history(self, target_history, scale_factor=1.0):
        if len(target_history) > 0:
            if scale_factor < 1.0:
                scale_factor += scale_factor ** 1.1
            for position in target_history:
                glPushMatrix()
                glTranslatef(position[0], position[1], 0.0)
                draw_circle(4 * (1 / scale_factor), [255, 255, 255, 255])
                glPopMatrix()

    def get_center_mass(self):
        sum = np.zeros((2, 1), np.float64)
        for i in self.sensor_nodes:
            sum += i.q
        return sum / len(self.sensor_nodes)

    def populate_adjacency_matrix(self):
        # Clear node neighbor lists
        for i in range(len(self.sensor_nodes)):
            self.sensor_nodes[i].neighbors = []
        # Populate adjacency matrix and node neighbor lists
        for i in range(len(self.sensor_nodes) - 1):
            for j in range(i + 1, len(self.sensor_nodes)):
                distance = self.sensor_nodes[i].distance_to_node(self.sensor_nodes[j])
                if distance <= self.sensor_nodes[i].sensor_range:
                    sigma_norm_ij = self.sensor_nodes[i].sigma_norm(self.sensor_nodes[j])
                    self.adjacency_matrix[i][j] = self.sensor_nodes[i].bump(sigma_norm_ij /
                                                                            self.sensor_nodes[i].sensor_range_alpha)
                    self.adjacency_matrix[j][i] = self.adjacency_matrix[i][j]
                    if self.adjacency_matrix[i][j] > 0.0:
                        self.sensor_nodes[i].neighbors.append(self.sensor_nodes[j])
                        self.sensor_nodes[j].neighbors.append(self.sensor_nodes[i])

    def populate_beta_matrix(self):
        for i in range(len(self.sensor_nodes)):
            self.sensor_nodes[i].beta_neighbors = []
            for k in range(self.number_of_obstacles):
                distance = self.sensor_nodes[i].distance_to_obstacle(self.obstacle_nodes[k])
                if distance <= self.sensor_nodes[i].obstacle_sense_range:
                    q_beta = self.sensor_nodes[i].q_beta_estimate(self.obstacle_nodes[k])
                    p_beta = self.sensor_nodes[i].p_beta_estimate(self.obstacle_nodes[k])
                    beta_agent = BetaNode(id=k, q=q_beta, p=p_beta)
                    sigma_norm_ik = self.sensor_nodes[i].sigma_norm(beta_agent)
                    b_ik = self.sensor_nodes[i].bump(sigma_norm_ik / self.sensor_nodes[i].obstacle_distance_beta)
                    self.beta_matrix[i][k] = b_ik
                    if self.beta_matrix[i][k] > 0.0:
                        self.sensor_nodes[i].beta_neighbors.append(beta_agent)

    def window_fit_flock(self):
        width = (self.width / 2) * (1 / self.current_scale)
        height = self.height / 2 * (1 / self.current_scale)
        for i in self.sensor_nodes:
            if i.q[1] >= width or i.q[0] >= height or i.q[1] <= -width or i.q[0] <= -height:
                self.scale_flag = True

    def draw_neighbor_connections(self, thickness=1.5, color=[0, 100, 255]):
        for node in self.sensor_nodes:
            for neighbor in node.neighbors:
                if node.id < neighbor.id:
                    # Get the beginning and end verteces
                    line_points = [node.q[0], node.q[1],
                                   neighbor.q[0], neighbor.q[1]]
                    pyglet.gl.glLineWidth(thickness)
                    pyglet.graphics.draw(2, GL_LINES,
                                         ('v2f', line_points),
                                         ('c3B', color * 2))

    def connectivity(self):
        return np.linalg.matrix_rank(self.adjacency_matrix, True) / self.number_of_sensor_nodes

    def draw_center_mass(self, location, scale_factor=1.0):
        glPushMatrix()
        glTranslatef(location[0], location[1], 0.0)
        if scale_factor < 1.0:
            scale_factor += scale_factor ** 1.1
        draw_circle(7 * (1 / scale_factor), [0, 0, 0, 255])
        glPopMatrix()


class CircularObstacle:
    def __init__(self, center=np.zeros((2, 1), np.float64), radius=50.0):
        self.center = center
        self.radius = radius
        self.p = np.zeros((2, 1), np.float64)
        self.color = [255, 255, 255, 255]

    def update(self, dt=1 / 100):
        self.center = self.center
        self.p = self.p

    def draw_obstacle(self):
        glPushMatrix()
        glTranslatef(self.center.item(0), self.center.item(1), 0.0)
        draw_circle(self.radius, [0, 0, 0, 255])
        draw_circle((7 / 8) * self.radius, self.color)
        draw_circle((5 / 8) * self.radius, [255, 0, 0, 255])
        glPopMatrix()


class TargetNode:
    def __init__(self, node_size=15.0, q=np.zeros((2, 1), np.float64)):
        self.node_size = node_size
        self.q = q
        self.p = np.zeros((2, 1), np.float64)
        self.time = 0.0

    def draw_target_node(self, scale_factor=1.0):
        glPushMatrix()
        glTranslatef(self.q[0], self.q[1], 0.0)
        if scale_factor < 1.0:
            scale_factor += scale_factor ** 1.1
        draw_circle(self.node_size * (1 / scale_factor), [0, 0, 0, 255])
        draw_circle((2 * self.node_size / 3) * (1 / scale_factor), [255, 255, 255, 255])
        draw_circle((self.node_size / 3) * (1 / scale_factor), [0, 0, 0, 255])
        glPopMatrix()

    def update(self, shape='', dt=1 / 100):
        if shape == '':
            pass
        elif shape == 'circle':
            r = 700
            # x = rcos(t) y = rsin(t)
            self.q[0] = r * cos(self.time)
            self.q[1] = r * sin(self.time)
            self.p[0] = -r * sin(self.time)
            self.p[1] = r * cos(self.time)
        elif shape == 'wave':
            speed = 2.0
            a = 350
            self.q[0] += speed
            self.q[1] = a * sin(3 * self.time)
            self.p[0] = speed
            self.p[1] = a * cos(3 * self.time)
        self.time += dt


class SensorNode:
    def __init__(self, node_size=4.0, sensor_range=72.0, neighbor_distance=60.0,
                 id=-1, q=np.zeros((2, 1), np.float64), target=TargetNode()):
        self.id = id
        self.node_size = node_size
        self.sensor_range = sensor_range
        self.sensor_range_alpha = self.sigma_norm(self.sensor_range)
        self.obstacle_sense_range = sensor_range
        self.obstacle_range_beta = self.sigma_norm(self.obstacle_sense_range)
        self.neighbor_distance = neighbor_distance
        self.neighbor_distance_alpha = self.sigma_norm(self.neighbor_distance)
        # Magic number ## just for testing
        self.obstacle_distance = neighbor_distance + 10
        self.obstacle_distance_beta = self.sigma_norm(self.obstacle_distance)

        self.q = q
        self.p = np.zeros((2, 1), np.float64)
        self.neighbors = []
        self.beta_neighbors = []
        self.target = target

    def update(self, dt=1 / 100):
        c1 = 30.0
        c2 = 2.0 * sqrt(c1)
        c3 = 2100.0
        c4 = 0.1 * sqrt(c3)
        u_alpha = c1 * self.gradient_function() + c2 * self.velocity_match()
        u_target = self.track_target()
        u_beta = c3 * self.gradient_beta() + c4 * self.velocity_beta()

        u = u_alpha + u_beta + u_target
        self.p += u * dt
        self.q += self.p * dt

    def gradient_function(self):
        sum = 0.0
        for j in self.neighbors:
            sum += self.phi_alpha(self.sigma_norm(j)) * self.n_ij(j)
        return sum

    def velocity_match(self):
        sum = 0.0
        for j in self.neighbors:
            sigma_norm_ij = self.sigma_norm(j)
            bumped = self.bump(sigma_norm_ij / self.sensor_range_alpha)
            sum += bumped * (j.p - self.p)
        return sum

    def gradient_beta(self):
        sum = 0.0
        for j in self.beta_neighbors:
            sum += self.phi_beta(self.sigma_norm(j)) * self.n_ij(j)
        return sum

    def velocity_beta(self):
        sum = 0.0
        for j in self.beta_neighbors:
            sigma_norm_ik = self.sigma_norm(j)
            b_ik = self.bump(sigma_norm_ik / self.obstacle_distance_beta)
            sum += b_ik * (j.p - self.p)
        return sum

    def track_target(self):
        c1 = 3.5
        c2 = 2 * sqrt(c1)
        return c1 * (self.target.q - self.q) + c2 * (self.target.p - self.p)

    def phi_alpha(self, z):
        return self.bump(z / self.sensor_range_alpha) * self.phi(z - self.neighbor_distance_alpha)

    def phi_beta(self, z):
        sigma_one = (z - self.obstacle_distance_beta) / sqrt(1 + (z - self.obstacle_distance_beta) ** 2)
        return self.bump(z / self.obstacle_distance_beta) * (sigma_one - 1)

    def phi(self, z, a=12.0, b=12.0):
        c = abs(a - b) / sqrt(4 * a * b)
        sigma_one = (z + c) / sqrt(1 + (z + c) ** 2)
        return 0.5 * ((a + b) * sigma_one + (a - b))

    def bump(self, distance_ratio, h=0.2):
        if 0 <= distance_ratio <= h:
            return 1.0
        elif h < distance_ratio <= 1.0:
            j = distance_ratio - h
            k = 1 - h
            l = pi * (j / k)
            return 0.5 * (1 + cos(l))
        else:
            return 0.0

    def n_ij(self, j, epsilon=0.1):
        diff_q = j.q - self.q
        return diff_q / sqrt(1 + epsilon * self.distance_to_node(j) ** 2)

    def q_beta_estimate(self, circular_obstacle):
        mu = self.mu_beta(circular_obstacle)
        center = circular_obstacle.center
        return mu * self.q + (1 - mu) * center

    def p_beta_estimate(self, circular_obstacle):
        mu = self.mu_beta(circular_obstacle)
        velocity_projection_matrix = self.p_matrix(circular_obstacle)
        return mu * velocity_projection_matrix * self.p

    def p_matrix(self, circular_obstacle):
        numerator = self.q - circular_obstacle.center
        denominator = self.distance_to_obstacle(circular_obstacle) + circular_obstacle.radius
        unit_normal = numerator / denominator
        return np.identity(2, np.float64) - unit_normal * unit_normal.T

    def mu_beta(self, circular_obstacle):
        return circular_obstacle.radius / (self.distance_to_obstacle(circular_obstacle) + circular_obstacle.radius)

    def sigma_norm(self, norm_input, epsilon=.1):
        if type(self) == type(norm_input) or type(norm_input) == type(BetaNode()):
            x = self.distance_to_node(norm_input)
        else:
            x = norm_input
        return (sqrt(1 + epsilon * x ** 2) - 1) / epsilon

    def distance_to_node(self, node):
        dif_x = self.q[0] - node.q[0]
        dif_y = self.q[1] - node.q[1]
        return sqrt(dif_x ** 2 + dif_y ** 2)

    def distance_to_obstacle(self, circular_obstacle):
        dif_x = self.q[0] - circular_obstacle.center.item(0)
        dif_y = self.q[1] - circular_obstacle.center.item(1)
        return sqrt(dif_x ** 2 + dif_y ** 2) - circular_obstacle.radius

    def draw_node(self, scale_factor=1.0):
        glPushMatrix()
        glTranslatef(self.q[0], self.q[1], 0.0)
        if scale_factor < 1.0:
            scale_factor += scale_factor ** 1.1
        draw_circle(self.node_size * (1 / scale_factor))
        glPopMatrix()

    def draw_sensor_range(self):
        glPushMatrix()
        glTranslatef(self.q[0], self.q[1], 0)
        draw_circle(self.sensor_range, [150, 255, 75, 25])
        glPopMatrix()


class BetaNode(SensorNode):
    def __init__(self, node_size=4.0,
                 id=-1, q=np.zeros((2, 1), np.float64), p=np.zeros((2, 1), np.float64)):
        self.node_size = node_size
        self.id = id
        self.q = q
        self.p = p


def draw_circle(radius, color=[255, 0, 0, 255]):
    number_of_triangles = 12
    angle = (2 * pi) / number_of_triangles

    x = [radius * cos(angle * i) for i in range(number_of_triangles + 1)]
    y = [radius * sin(angle * i) for i in range(number_of_triangles + 1)]
    points = [0, 0]
    points += list(itertools.chain(*zip(x, y)))
    # Magic number "+ 2" is the number of points greater
    # than the number of triangles needed to make the Triangle_FAN
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    pyglet.graphics.draw(number_of_triangles + 2, GL_TRIANGLE_FAN,
                         ('v2f', points),
                         ('c4B', color * (number_of_triangles + 2)))
    glDisable(GL_BLEND)


if __name__ == '__main__':
    main()

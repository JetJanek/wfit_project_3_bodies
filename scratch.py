# import math
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.animation import FuncAnimation
# import numpy as np
#
# class Planet:
#     def __init__(self, x, y, z, radius, color, mass, vx=0, vy=0, vz=0):
#         self.x = x
#         self.y = y
#         self.z = z
#         self.radius = radius
#         self.color = color
#         self.mass = mass
#         self.vx = vx
#         self.vy = vy
#         self.vz = vz
#         self.orbit = [(x, y, z)]
#
# def update_position(planet, dt):
#     planet.x += planet.vx * dt
#     planet.y += planet.vy * dt
#     planet.z += planet.vz * dt
#
# def update_velocity(planet, force, dt):
#     ax = force[0] / planet.mass
#     ay = force[1] / planet.mass
#     az = force[2] / planet.mass
#     planet.vx += ax * dt
#     planet.vy += ay * dt
#     planet.vz += az * dt
#
# def gravitational_force(planet1, planet2):
#     G = 10.0
#     dx = planet2.x - planet1.x
#     dy = planet2.y - planet1.y
#     dz = planet2.z - planet1.z
#     distance_squared = dx**2 + dy**2 + dz**2
#     distance = math.sqrt(distance_squared)
#     force_magnitude = G * planet1.mass * planet2.mass / distance_squared
#     force_x = force_magnitude * dx / distance
#     force_y = force_magnitude * dy / distance
#     force_z = force_magnitude * dz / distance
#     return (force_x, force_y, force_z)
#
# def simulate(planets, dt):
#     for planet in planets:
#         net_force = [0, 0, 0]
#         for other_planet in planets:
#             if planet != other_planet:
#                 force = gravitational_force(planet, other_planet)
#                 net_force[0] += force[0]
#                 net_force[1] += force[1]
#                 net_force[2] += force[2]
#         update_velocity(planet, net_force, dt)
#     for planet in planets:
#         update_position(planet, dt)
#         planet.orbit.append((planet.x, planet.y, planet.z))
#
# def animate(frame, planets, scatters, lines):
#     dt = 0.01
#     simulate(planets, dt)
#
#     for i, planet in enumerate(planets):
#         updated_points = np.array(planet.orbit).T
#         lines[i].set_data(updated_points[0], updated_points[1])
#         lines[i].set_3d_properties(updated_points[2])
#         scatters[i]._offsets3d = (np.array([planet.x]), np.array([planet.y]), np.array([planet.z]))
#
# # Tworzenie instancji Planet z początkowymi współrzędnymi i prędkościami
# v = 3
# L = 5  # Zmiana skali, aby lepiej wizualizować
#
# planet_A = Planet(1, 1, 2, 0.1, 'red', 10, 0, 0, 0)
# planet_B = Planet(L * 2, 1, 3, 0.1, 'green', 3, -v / 2, v * math.sqrt(3) / 2, 0)
# planet_C = Planet(0, 0, -2, 0.1, 'blue', 3, v, -v * math.sqrt(3) / 2, 0)
# # planet_D = Planet(L, 2, 0, 0.1, 'black', 3, -v / 3, v * math.sqrt(3), 0)
# # planet_E = Planet(3, 0, -4, 0.1, 'brown', 3, 2*v, -v * math.sqrt(3) / 4, 0)
# # planet_F = Planet(L, 4, 4, 0.1, 'yellow', 3, v / 3, v * math.sqrt(5), 0)
# # planet_G = Planet(2, 0, 0, 0.1, 'pink', 3, v/2, -v * math.sqrt(5) / 2, 0)
# # planet_H = Planet(1, 0, -1, 0.1, 'grey', 3, v, v, 0)
# # , planet_D, planet_E, planet_F, planet_G, planet_H
#
# # Tworzenie listy planet
# planets = [planet_A, planet_B, planet_C]
#
# # Konfiguracja animacji
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# scatters = [ax.scatter(planet.x, planet.y, planet.z, color=planet.color, s=planet.mass*50) for planet in planets]
# lines = [ax.plot([], [], [], color=planet.color, linewidth=1)[0] for planet in planets]
#
# ani = FuncAnimation(fig, animate, fargs=(planets, scatters, lines), frames=range(1000), interval=50)
#
# # Ustawianie ograniczeń osi
# min_limit = -L
# max_limit = L
# ax.set_xlim(min_limit, max_limit)
# ax.set_ylim(min_limit, max_limit)
# ax.set_zlim(min_limit, max_limit)
#
# # Konfiguracja interaktywności
# ax.view_init(elev=20, azim=30)
#
# # Wyświetlanie animacji
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.animation import FuncAnimation
#
# # Definicja stałych
# G = 1.0  # Stała grawitacyjna
#
# class Planet:
#     def __init__(self, x, y, z, vx, vy, vz, mass, color):
#         self.pos = np.array([x, y, z], dtype=float)
#         self.vel = np.array([vx, vy, vz], dtype=float)
#         self.mass = mass
#         self.color = color
#         self.orbit = [self.pos.copy()]
#
# def compute_accelerations(planets):
#     n = len(planets)
#     accelerations = [np.zeros(3) for _ in range(n)]
#
#     for i in range(n):
#         for j in range(n):
#             if i != j:
#                 diff = planets[j].pos - planets[i].pos
#                 dist = np.linalg.norm(diff)
#                 accelerations[i] += G * planets[j].mass * diff / dist**3
#
#     return accelerations
#
# def rk4_step(planets, dt):
#     n = len(planets)
#
#     pos = np.array([p.pos for p in planets])
#     vel = np.array([p.vel for p in planets])
#     masses = np.array([p.mass for p in planets])
#
#     def get_derivatives(pos, vel):
#         acc = np.zeros_like(pos)
#         for i in range(n):
#             for j in range(n):
#                 if i != j:
#                     diff = pos[j] - pos[i]
#                     dist = np.linalg.norm(diff)
#                     acc[i] += G * masses[j] * diff / dist**3
#         return vel, acc
#
#     k1_vel, k1_acc = get_derivatives(pos, vel)
#     k2_vel, k2_acc = get_derivatives(pos + k1_vel * dt / 2, vel + k1_acc * dt / 2)
#     k3_vel, k3_acc = get_derivatives(pos + k2_vel * dt / 2, vel + k2_acc * dt / 2)
#     k4_vel, k4_acc = get_derivatives(pos + k3_vel * dt, vel + k3_acc * dt)
#
#     new_pos = pos + dt / 6 * (k1_vel + 2 * k2_vel + 2 * k3_vel + k4_vel)
#     new_vel = vel + dt / 6 * (k1_acc + 2 * k2_acc + 2 * k3_acc + k4_acc)
#
#     for i in range(n):
#         planets[i].pos = new_pos[i]
#         planets[i].vel = new_vel[i]
#         planets[i].orbit.append(new_pos[i].copy())
#
# def animate(frame, planets, scatters, lines):
#     dt = 0.01
#     rk4_step(planets, dt)
#
#     for i, planet in enumerate(planets):
#         updated_points = np.array(planet.orbit).T
#         lines[i].set_data(updated_points[0], updated_points[1])
#         lines[i].set_3d_properties(updated_points[2])
#         scatters[i]._offsets3d = (planet.pos[0], planet.pos[1], planet.pos[2])
#
# # Tworzenie instancji Planet z początkowymi współrzędnymi i prędkościami
# v = 3
# L = 5
#
# planet_A = Planet(1, 1, 2, 0, 0, 0, 10, 'red')
# planet_B = Planet(L * 2, 1, 3, -v / 2, v * np.sqrt(3) / 2, 0, 3, 'green')
# planet_C = Planet(0, 0, -2, v, -v * np.sqrt(3) / 2, 0, 3, 'blue')
#
# # Tworzenie listy planet
# planets = [planet_A, planet_B, planet_C]
#
# # Konfiguracja animacji
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# scatters = [ax.scatter(planet.pos[0], planet.pos[1], planet.pos[2], color=planet.color, s=planet.mass*10) for planet in planets]
# lines = [ax.plot([], [], [], color=planet.color, linewidth=1)[0] for planet in planets]
#
# ani = FuncAnimation(fig, animate, fargs=(planets, scatters, lines), frames=range(1000), interval=50)
#
# # Ustawianie ograniczeń osi
# min_limit = -L
# max_limit = L
# ax.set_xlim(min_limit, max_limit)
# ax.set_ylim(min_limit, max_limit)
# ax.set_zlim(min_limit, max_limit)
#
# # Konfiguracja interaktywności
# ax.view_init(elev=20, azim=30)
#
# # Wyświetlanie animacji
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# Stała grawitacyjna
G = 10.0

class Planet:
    def __init__(self, x, y, z, vx, vy, vz, mass, color):
        self.pos = np.array([x, y, z], dtype=float)
        self.vel = np.array([vx, vy, vz], dtype=float)
        self.mass = mass
        self.color = color
        self.orbit = [self.pos.copy()]

def compute_accelerations(positions, masses):
    n = len(masses)
    accelerations = np.zeros_like(positions)
    for i in range(n):
        for j in range(n):
            if i != j:
                diff = positions[j] - positions[i]
                dist = np.linalg.norm(diff)
                accelerations[i] += G * masses[j] * diff / dist**3
    return accelerations

def equations(state, masses, dt):
    n = len(masses)
    positions = state[:3*n].reshape((n, 3))
    velocities = state[3*n:].reshape((n, 3))

    acc = compute_accelerations(positions, masses)
    new_positions = positions + velocities * dt
    new_velocities = velocities + acc * dt

    return np.hstack((new_positions.flatten(), new_velocities.flatten())) - state

def solve_3body(planets, dt, steps):
    n = len(planets)
    masses = np.array([p.mass for p in planets])
    state = np.hstack((np.array([p.pos for p in planets]).flatten(),
                       np.array([p.vel for p in planets]).flatten()))

    trajectories = [np.array([p.pos for p in planets])]

    for _ in range(steps):
        state = fsolve(equations, state, args=(masses, dt))
        positions = state[:3*n].reshape((n, 3))
        for i, planet in enumerate(planets):
            planet.pos = positions[i]
            planet.orbit.append(positions[i].copy())
        trajectories.append(positions)

    return trajectories

# Tworzenie instancji Planet z początkowymi współrzędnymi i prędkościami
v = 3
L = 5

planet_A = Planet(1, 1, 2, 0, 0, 0, 10, 'red')
planet_B = Planet(L * 2, 1, 3, -v / 2, v * np.sqrt(3) / 2, 0, 3, 'green')
planet_C = Planet(0, 0, -2, v, -v * np.sqrt(3) / 2, 0, 3, 'blue')

# Tworzenie listy planet
planets = [planet_A, planet_B, planet_C]

# Parametry symulacji
dt = 0.01
steps = 1000

# Rozwiązywanie problemu trzech ciał
trajectories = solve_3body(planets, dt, steps)

# Konfiguracja wykresu
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for planet in planets:
    orbit = np.array(planet.orbit)
    ax.plot(orbit[:, 0], orbit[:, 1], orbit[:, 2], color=planet.color)

ax.set_xlim(-L, L)
ax.set_ylim(-L, L)
ax.set_zlim(-L, L)

plt.show()
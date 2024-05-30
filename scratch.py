import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import numpy as np

class Planet:
    def __init__(self, x, y, z, radius, color, mass, vx=0, vy=0, vz=0):
        self.x = x
        self.y = y
        self.z = z
        self.radius = radius
        self.color = color
        self.mass = mass
        self.vx = vx
        self.vy = vy
        self.vz = vz
        self.orbit = [(x, y, z)]

def update_position(planet, dt):
    planet.x += planet.vx * dt
    planet.y += planet.vy * dt
    planet.z += planet.vz * dt

def update_velocity(planet, force, dt):
    ax = force[0] / planet.mass
    ay = force[1] / planet.mass
    az = force[2] / planet.mass
    planet.vx += ax * dt
    planet.vy += ay * dt
    planet.vz += az * dt

def gravitational_force(planet1, planet2):
    G = 1.0
    dx = planet2.x - planet1.x
    dy = planet2.y - planet1.y
    dz = planet2.z - planet1.z
    distance_squared = dx**2 + dy**2 + dz**2
    distance = math.sqrt(distance_squared)
    force_magnitude = G * planet1.mass * planet2.mass / distance_squared
    force_x = force_magnitude * dx / distance
    force_y = force_magnitude * dy / distance
    force_z = force_magnitude * dz / distance
    return (force_x, force_y, force_z)

def simulate(planets, dt):
    for planet in planets:
        net_force = [0, 0, 0]
        for other_planet in planets:
            if planet != other_planet:
                force = gravitational_force(planet, other_planet)
                net_force[0] += force[0]
                net_force[1] += force[1]
                net_force[2] += force[2]
        update_velocity(planet, net_force, dt)
    for planet in planets:
        update_position(planet, dt)
        planet.orbit.append((planet.x, planet.y, planet.z))

def animate(frame, planets, scatters, lines, time_text):
    dt = 0.01  # Small time step for stable simulation
    simulate(planets, dt)

    for i, planet in enumerate(planets):
        updated_points = np.array(planet.orbit).T
        lines[i].set_data(updated_points[0], updated_points[1])
        lines[i].set_3d_properties(updated_points[2])
        scatters[i]._offsets3d = (np.array([planet.x]), np.array([planet.y]), np.array([planet.z]))

    current_time = frame * dt * time_conversion
    time_text.set_text(f'Time: {current_time:.2f} years')

# Initial positions and velocities for stable orbits
L = 5
time_conversion = 1 / 3.154e+7

planet_A = Planet(1, 0, 0, 0.1, 'red', 10, 0, 0.5, 0)
planet_B = Planet(-1, 0, 0, 0.1, 'green', 3, 0, -0.5, 0)
planet_C = Planet(0, 1, 0, 0.1, 'blue', 3, -0.5, 0, 0)

planets = [planet_A, planet_B, planet_C]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

scatters = [ax.scatter(planet.x, planet.y, planet.z, color=planet.color, s=100) for planet in planets]
lines = [ax.plot([], [], [], color=planet.color, linewidth=0.5)[0] for planet in planets]

time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes)

ani = FuncAnimation(fig, animate, fargs=(planets, scatters, lines, time_text), frames=range(1000), interval=20)

min_limit = -L
max_limit = L
ax.set_xlim(min_limit, max_limit)
ax.set_ylim(min_limit, max_limit)
ax.set_zlim(min_limit, max_limit)

ax.view_init(elev=20, azim=30)

plt.show()

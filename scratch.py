import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Stałe
G = 6.67430e-11  # Stała grawitacyjna [m^3/kg/s^2]
m1 = 5.972e24     # Masa ciała 1 (Ziemia) [kg]
m2 = 7.34767309e22  # Masa ciała 2 (Księżyc) [kg]

# Początkowe warunki
x1_0 = 0.0    # Początkowa pozycja ciała 1 na osi x [m]
y1_0 = 0.0    # Początkowa pozycja ciała 1 na osi y [m]
x2_0 = 3.844e8  # Początkowa pozycja ciała 2 na osi x [m] (odległość Ziemia-Księżyc)
y2_0 = 0.0    # Początkowa pozycja ciała 2 na osi y [m]
vx1_0 = 0.0   # Początkowa prędkość ciała 1 na osi x [m/s]
vy1_0 = 1.0  # Początkowa prędkość ciała 1 na osi y [m/s] (prędkość orbitalna Ziemi)
vx2_0 = 0.0   # Początkowa prędkość ciała 2 na osi x [m/s]
vy2_0 = 0.0   # Początkowa prędkość ciała 2 na osi y [m/s]

# Funkcja obliczająca przyspieszenie
def acceleration(x1, y1, x2, y2):
    r = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)  # Odległość między ciałami
    ax1 = G * m2 * (x2 - x1) / r**3  # Przyspieszenie ciała 1 na osi x
    ay1 = G * m2 * (y2 - y1) / r**3  # Przyspieszenie ciała 1 na osi y
    ax2 = G * m1 * (x1 - x2) / r**3  # Przyspieszenie ciała 2 na osi x
    ay2 = G * m1 * (y1 - y2) / r**3  # Przyspieszenie ciała 2 na osi y
    return ax1, ay1, ax2, ay2

# Symulacja ruchu
t_max = 200000  # Czas symulacji [s]
dt = 100        # Krok czasowy [s]

# Listy do przechowywania wyników
x1_list, y1_list, x2_list, y2_list = [], [], [], []

# Warunki początkowe
x1, y1, x2, y2 = x1_0, y1_0, x2_0, y2_0
vx1, vy1, vx2, vy2 = vx1_0, vy1_0, vx2_0, vy2_0

# Symulacja
for t in range(0, t_max, dt):
    ax1, ay1, ax2, ay2 = acceleration(x1, y1, x2, y2)
    vx1 += ax1 * dt
    vy1 += ay1 * dt
    vx2 += ax2 * dt
    vy2 += ay2 * dt
    x1 += vx1 * dt
    y1 += vy1 * dt
    x2 += vx2 * dt
    y2 += vy2 * dt
    x1_list.append(x1)
    y1_list.append(y1)
    x2_list.append(x2)
    y2_list.append(y2)

# Funkcja inicjująca animację
def init():
    line1.set_data([], [])
    line2.set_data([], [])
    return line1, line2

# Funkcja animacji
def animate(i):
    line1.set_data(x1_list[:i], y1_list[:i])
    line2.set_data(x2_list[:i], y2_list[:i])
    return line1, line2
# Wykres
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(-1.5e8, 5e8)  # Zmieniona skala osi x
ax.set_ylim(-1.5e8, 1.5e8)  # Zmieniona skala osi y
ax.set_xlabel('Pozycja na osi x [m]')
ax.set_ylabel('Pozycja na osi y [m]')
ax.set_title('Ruch grawitacyjny dwóch ciał')

# Linie trajektorii
line1, = ax.plot([], [], lw=2, label='Ziemia')
line2, = ax.plot([], [], lw=2, label='Księżyc')
ax.legend()
ax.grid(True)

# Inicjalizacja animacji
ani = FuncAnimation(fig, animate, init_func=init, frames=len(x1_list), interval=10, blit=True)

# Wyświetlenie animacji
plt.show()

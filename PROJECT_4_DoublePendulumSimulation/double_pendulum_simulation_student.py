import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import matplotlib.animation as animation

# Constants
G_CONST = 9.81  # Acceleration due to gravity (m/s^2)
L_CONST = 0.4   # Length of each pendulum arm (m)
M_CONST = 1.0   # Mass of each pendulum bob (kg)

def derivatives(y, t, L1, L2, m1, m2, g_param):
    """
    Returns the time derivatives of the state vector y for a double pendulum.

    Args:
        y (list or np.array): Current state vector [theta1, omega1, theta2, omega2].
        t (float): Current time (not used directly in these autonomous equations, but required by odeint).
        L1 (float): Length of the first pendulum arm.
        L2 (float): Length of the second pendulum arm.
        m1 (float): Mass of the first bob.
        m2 (float): Mass of the second bob.
        g_param (float): Acceleration due to gravity.

    Returns:
        list: Time derivatives [dtheta1_dt, domega1_dt, dtheta2_dt, domega2_dt].
    """
    theta1, omega1, theta2, omega2 = y

    # Equations of motion (simplified for L1=L2=L, m1=m2=m)
    dtheta1_dt = omega1
    dtheta2_dt = omega2

    domega1_dt = (-omega1**2 * np.sin(2*theta1 - 2*theta2) - 2 * omega2**2 * np.sin(theta1 - theta2) - 
                  (g_param / L1) * (np.sin(theta1 - 2*theta2) + 3 * np.sin(theta1))) / (3 - np.cos(2*theta1 - 2*theta2))
    domega2_dt = (4 * omega1**2 * np.sin(theta1 - theta2) + omega2**2 * np.sin(2*theta1 - 2*theta2) + 
                  2 * (g_param / L1) * (np.sin(2*theta1 - theta2) - np.sin(theta2))) / (3 - np.cos(2*theta1 - 2*theta2))
    
    return [dtheta1_dt, domega1_dt, dtheta2_dt, domega2_dt]

def solve_double_pendulum(initial_conditions, t_span, t_points, L_param=L_CONST, g_param=G_CONST):
    """
    Solves the double pendulum ODEs.

    Args:
        initial_conditions (dict): {'theta1': val, 'omega1': val, 'theta2': val, 'omega2': val} in radians and rad/s.
        t_span (tuple): (t_start, t_end) for the simulation.
        t_points (int): Number of time points to generate.
        L_param (float): Pendulum arm length.
        g_param (float): Acceleration due to gravity.

    Returns:
        tuple: (t_arr, sol_arr)
               t_arr: 1D numpy array of time points.
               sol_arr: 2D numpy array with states [theta1, omega1, theta2, omega2] at each time point.
    """
    y0 = [initial_conditions['theta1'], initial_conditions['omega1'], 
          initial_conditions['theta2'], initial_conditions['omega2']]
    t_arr = np.linspace(t_span[0], t_span[1], t_points)
    
    sol_arr = odeint(derivatives, y0, t_arr, args=(L_param, L_param, M_CONST, M_CONST, g_param), rtol=1e-9, atol=1e-9)
    return t_arr, sol_arr

def calculate_energy(sol_arr, L_param=L_CONST, m_param=M_CONST, g_param=G_CONST):
    """
    Calculates the total energy of the double pendulum system.

    Args:
        sol_arr (np.array): Solution array from odeint (rows are time points, columns are [theta1, omega1, theta2, omega2]).
        L_param (float): Pendulum arm length.
        m_param (float): Bob mass.
        g_param (float): Acceleration due to gravity.

    Returns:
        np.array: 1D array of total energy at each time point.
    """
    theta1 = sol_arr[:, 0]
    omega1 = sol_arr[:, 1]
    theta2 = sol_arr[:, 2]
    omega2 = sol_arr[:, 3]

    V = -m_param * g_param * L_param * (2 * np.cos(theta1) + np.cos(theta2))
    T = m_param * L_param**2 * (omega1**2 + 0.5 * omega2**2 + omega1 * omega2 * np.cos(theta1 - theta2))
    
    return T + V

def animate_double_pendulum(t_arr, sol_arr, L_param=L_CONST, skip_frames=10):
    """
    Creates an animation of the double pendulum.

    Args:
        t_arr (np.array): Time array.
        sol_arr (np.array): Solution array from odeint.
        L_param (float): Pendulum arm length.
        skip_frames (int): Number of solution steps to skip for each animation frame.

    Returns:
        matplotlib.animation.FuncAnimation: The animation object.
    """
    theta1_all = sol_arr[:, 0]
    theta2_all = sol_arr[:, 2]

    theta1_anim = theta1_all[::skip_frames]
    theta2_anim = theta2_all[::skip_frames]
    t_anim = t_arr[::skip_frames]

    x1 = L_param * np.sin(theta1_anim)
    y1 = -L_param * np.cos(theta1_anim)
    x2 = x1 + L_param * np.sin(theta2_anim)
    y2 = y1 - L_param * np.cos(theta2_anim)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2*L_param - 0.1, 2*L_param + 0.1), ylim=(-2*L_param - 0.1, 0.1))
    ax.set_aspect('equal')
    ax.grid()
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('Double Pendulum Animation')

    line, = ax.plot([], [], 'o-', lw=2, markersize=8, color='blue') 
    time_template = 'Time = %.1fs'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    def init():
        line.set_data([], [])
        time_text.set_text('')
        return line, time_text

    def animate(i):
        thisx = [0, x1[i], x2[i]]
        thisy = [0, y1[i], y2[i]]
        line.set_data(thisx, thisy)
        time_text.set_text(time_template % t_anim[i])
        return line, time_text

    ani = animation.FuncAnimation(fig, animate, frames=len(t_anim),
                                  interval=25, blit=True, init_func=init)
    return ani

if __name__ == "__main__":
    # Initial conditions
    initial_conditions_rad = {
        'theta1': np.pi/2,  # 90 degrees
        'omega1': 0.0,
        'theta2': np.pi/2,  # 90 degrees
        'omega2': 0.0
    }
    t_start = 0
    t_end = 100
    t_points_sim = 2000

    # 1. Solve ODEs
    print(f"Solving ODEs for t = {t_start}s to {t_end}s...")
    t_solution, sol_solution = solve_double_pendulum(initial_conditions_rad, (t_start, t_end), t_points_sim, L_param=L_CONST, g_param=G_CONST)
    print("ODE solving complete.")

    # 2. Calculate Energy
    print("Calculating energy...")
    energy_solution = calculate_energy(sol_solution, L_param=L_CONST, m_param=M_CONST, g_param=G_CONST)
    print("Energy calculation complete.")

    # 3. Plot Energy vs Time
    plt.figure(figsize=(10, 5))
    plt.plot(t_solution, energy_solution, label='Total Energy')
    plt.xlabel('Time (s)')
    plt.ylabel('Energy (Joules)')
    plt.title('Total Energy of the Double Pendulum vs. Time')
    plt.grid(True)
    plt.legend()
    initial_energy = energy_solution[0]
    final_energy = energy_solution[-1]
    energy_variation = np.max(energy_solution) - np.min(energy_solution)
    print(f"Initial Energy: {initial_energy:.7f} J")
    print(f"Final Energy:   {final_energy:.7f} J")
    print(f"Max Energy Variation: {energy_variation:.7e} J")
    if energy_variation < 1e-5:
        print("Energy conservation target (< 1e-5 J) met.")
    else:
        print(f"Energy conservation target (< 1e-5 J) NOT met. Variation: {energy_variation:.2e} J. Try increasing t_points or tightening odeint tolerances.")
    plt.ylim(initial_energy - 5*energy_variation if energy_variation > 1e-7 else initial_energy - 1e-5, 
             initial_energy + 5*energy_variation if energy_variation > 1e-7 else initial_energy + 1e-5)
    plt.show()

    # 4. (Optional) Animate
    run_animation = True 
    if run_animation:
        print("Creating animation... This might take a moment.")
        anim_object = animate_double_pendulum(t_solution, sol_solution, L_param=L_CONST, skip_frames=max(1, t_points_sim // 1000) * 5) 
        plt.show() 
        print("Animation display complete.")
    else:
        print("Animation skipped.")

    print("Double Pendulum Simulation finished.")

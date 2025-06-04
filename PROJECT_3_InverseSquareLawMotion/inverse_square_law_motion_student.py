"""
学生模板：平方反比引力场中的运动
文件：inverse_square_law_motion_student.py
作者：[你的名字]
日期：[完成日期]

重要：函数名称、参数名称和返回值的结构必须与参考答案保持一致！
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# 常量 (如果需要，学生可以自行定义或从参数传入)
# 例如：GM = 1.0 # 引力常数 * 中心天体质量

def derivatives(t, state_vector, gm_val):
    """
    计算状态向量 [x, y, vx, vy] 的导数。

    运动方程（直角坐标系）:
    dx/dt = vx
    dy/dt = vy
    dvx/dt = -GM * x / r^3
    dvy/dt = -GM * y / r^3
    其中 r = sqrt(x^2 + y^2)。

    参数:
        t (float): 当前时间 (solve_ivp 需要，但在此自治系统中不直接使用)。
        state_vector (np.ndarray): 一维数组 [x, y, vx, vy]，表示当前状态。
        gm_val (float): 引力常数 G 与中心天体质量 M 的乘积。

    返回:
        np.ndarray: 一维数组，包含导数 [dx/dt, dy/dt, dvx/dt, dvy/dt]。
    
    实现提示:
    1. 从 state_vector 中解包出 x, y, vx, vy。
    2. 计算 r_cubed = (x**2 + y**2)**1.5。
    3. 注意处理 r_cubed 接近零的特殊情况（例如，如果 r 非常小，可以设定一个阈值避免除以零）。
    4. 计算加速度 ax 和 ay。
    5. 返回 [vx, vy, ax, ay]。
    """
    # TODO: 学生在此处实现代码
    x,y,vx,vy=state_vector
    r_cubed=(x**2+y**2)**1.5
    if r_cubed < 1e-12:
        ax = -gm_val * x / (1e-12) if x != 0 else 0
        ay = -gm_val * y / (1e-12) if y != 0 else 0
        return [vx, vy, ax, ay]
    ax=-gm_val*x/r_cubed
    ay=-gm_val*y/r_cubed
    return np.array([vx,vy,ax,ay])


def solve_orbit(initial_conditions, t_span, t_eval, gm_val):
    """
    使用 scipy.integrate.solve_ivp 求解轨道运动问题。

    参数:
        initial_conditions (list or np.ndarray): 初始状态 [x0, y0, vx0, vy0]。
        t_span (tuple): 积分时间区间 (t_start, t_end)。
        t_eval (np.ndarray): 需要存储解的时间点数组。
        gm_val (float): GM 值 (引力常数 * 中心天体质量)。

    返回:
        scipy.integrate.OdeSolution: solve_ivp 返回的解对象。
                                     可以通过 sol.y 访问解的数组，sol.t 访问时间点。
    
    实现提示:
    1. 调用 solve_ivp 函数。
    2. `fun` 参数应为你的 `derivatives` 函数。
    3. `args` 参数应为一个元组，包含传递给 `derivatives` 函数的额外参数 (gm_val,)。
    4. 可以选择合适的数值方法 (method)，如 'RK45' (默认) 或 'DOP853'。
    5. 设置合理的相对容差 (rtol) 和绝对容差 (atol) 以保证精度，例如 rtol=1e-7, atol=1e-9。
    """
    # TODO: 学生在此处实现代码
    sol=solve_ivp(
        fun=derivatives,
        t_span=t_span,
        y0=initial_conditions,
        t_eval=t_eval,
        args=(gm_val,),
        method='RK45',
        rtol=1e-7,atol=1e-9
    )
    return sol

def calculate_energy(state_vector, gm_val, m=1.0):
    """
    计算质点的（比）机械能。
    （比）能量 E/m = 0.5 * v^2 - GM/r

    参数:
        state_vector (np.ndarray): 二维数组，每行是 [x, y, vx, vy]，或单个状态的一维数组。
        gm_val (float): GM 值。
        m (float, optional): 运动质点的质量。默认为 1.0，此时计算的是比能 (E/m)。

    返回:
        np.ndarray or float: （比）机械能。

    实现提示:
    1. 处理 state_vector 可能是一维（单个状态）或二维（多个状态的时间序列）的情况。
    2. 从 state_vector 中提取 x, y, vx, vy。
    3. 计算距离 r = np.sqrt(x**2 + y**2)。注意避免 r=0 导致除以零的错误。
    4. 计算速度的平方 v_squared = vx**2 + vy**2。
    5. 计算比动能 kinetic_energy_per_m = 0.5 * v_squared。
    6. 计算比势能 potential_energy_per_m = -gm_val / r (注意处理 r=0 的情况)。
    7. 比机械能 specific_energy = kinetic_energy_per_m + potential_energy_per_m。
    8. 如果需要总能量，则乘以质量 m。
    """
    # TODO: 学生在此处实现代码
    is_single_state = state_vector.ndim == 1
    if is_single_state:
        state_vector = state_vector.reshape(1, -1)

    x = state_vector[:, 0]
    y = state_vector[:, 1]
    vx = state_vector[:, 2]
    vy = state_vector[:, 3]
    r=np.sqrt(x**2+y**2)
    v_squared = vx**2 + vy**2
    potential_energy_per_m = np.zeros_like(r)
    non_zero_r_mask = r > 1e-12
    potential_energy_per_m[non_zero_r_mask] = -gm_val / r[non_zero_r_mask]

    if np.any(~non_zero_r_mask):
        print("Warning: r=0 encountered in energy calculation. Potential energy is singular.")
        potential_energy_per_m[~non_zero_r_mask] = -np.inf # Or some other indicator

    kinetic_energy_per_m = 0.5 * v_squared
    specific_energy = kinetic_energy_per_m + potential_energy_per_m
    
    total_energy = m * specific_energy

    return total_energy[0] if is_single_state else total_energy

def calculate_angular_momentum(state_vector, m=1.0):
    """
    计算质点的（比）角动量 (z分量)。
    （比）角动量 Lz/m = x*vy - y*vx

    参数:
        state_vector (np.ndarray): 二维数组，每行是 [x, y, vx, vy]，或单个状态的一维数组。
        m (float, optional): 运动质点的质量。默认为 1.0，此时计算的是比角动量 (Lz/m)。

    返回:
        np.ndarray or float: （比）角动量。

    实现提示:
    1. 处理 state_vector 可能是一维或二维的情况。
    2. 从 state_vector 中提取 x, y, vx, vy。
    3. 计算比角动量 specific_Lz = x * vy - y * vx。
    4. 如果需要总角动量，则乘以质量 m。
    """
    # TODO: 学生在此处实现代码
    is_single_state = state_vector.ndim == 1
    if is_single_state:
        state_vector = state_vector.reshape(1, -1)
        
    x = state_vector[:, 0]
    y = state_vector[:, 1]
    vx = state_vector[:, 2]
    vy = state_vector[:, 3]

    specific_Lz = x * vy - y * vx
    total_Lz = m * specific_Lz
    
    return total_Lz[0] if is_single_state else total_Lz


if __name__ == "__main__":
    # --- 学生可以在此区域编写测试代码或进行实验 ---
    print("平方反比引力场中的运动 - 学生模板")

    # 任务1：实现函数并通过基础测试 (此处不设测试，依赖 tests 文件)

    # 任务2：不同总能量下的轨道绘制
    # 示例：设置椭圆轨道初始条件 (学生需要根据物理意义自行调整或计算得到)
    # GM_val_demo = 1.0
    # ic_ellipse_demo = [1.0, 0.0, 0.0, 0.8]
    # t_start_demo = 0
    # t_end_demo = 20
    # t_eval_demo = np.linspace(t_start_demo, t_end_demo, 500)

    # try:
    #     sol_ellipse = solve_orbit(ic_ellipse_demo, (t_start_demo, t_end_demo), t_eval_demo, gm_val=GM_val_demo)
    #     x_ellipse, y_ellipse = sol_ellipse.y[0], sol_ellipse.y[1]
        
    #     # 计算能量和角动量 (假设学生已实现)
    #     # energy = calculate_energy(sol_ellipse.y.T, GM_val_demo)
    #     # angular_momentum = calculate_angular_momentum(sol_ellipse.y.T)
    #     # print(f"Ellipse Demo: Initial Energy approx {energy[0]:.3f}")

    #     plt.figure(figsize=(8, 6))
    #     plt.plot(x_ellipse, y_ellipse, label='椭圆轨道 (示例)')
    #     plt.plot(0, 0, 'ko', markersize=8, label='中心天体')
    #     plt.title('轨道运动示例')
    #     plt.xlabel('x 坐标')
    #     plt.ylabel('y 坐标')
    #     plt.legend()
    #     plt.grid(True)
    #     plt.axis('equal')
    #     plt.show()
    # except NotImplementedError:
    #     print("请先实现 solve_orbit, calculate_energy, calculate_angular_momentum 函数。")
    # except Exception as e:
    #     print(f"运行示例时发生错误: {e}")

    # 学生需要根据“项目说明.md”完成以下任务：
    # 1. 实现 `derivatives`, `solve_orbit`, `calculate_energy`, `calculate_angular_momentum` 函数。
    # 2. 针对 E > 0, E = 0, E < 0 三种情况设置初始条件，求解并绘制轨道。
    # 3. 针对 E < 0 且固定时，改变角动量，求解并绘制轨道。
    # 4. (可选) 进行坐标转换和对称性分析。

    print("\n请参照 '项目说明.md' 完成各项任务。")
    print("使用 'tests/test_inverse_square_law_motion.py' 文件来测试你的代码实现。")
    
    t_start = 0
    t_end_ellipse = 20  # Enough time for a few orbits for typical elliptical case
    t_end_hyperbola = 5 # Hyperbola moves away quickly
    t_end_parabola = 10 # Parabola also moves away
    n_points = 1000
    mass_particle = 1.0 # Assume m=1 for simplicity in E and L calculations
    
    # Case 1: Elliptical Orbit (E < 0)
    # Initial conditions: x0=1, y0=0, vx0=0, vy0=0.8 (adjust vy0 for different eccentricities)
    ic_ellipse = [1.0, 0.0, 0.0, 0.8]
    t_eval_ellipse = np.linspace(t_start, t_end_ellipse, n_points)
    sol_ellipse = solve_orbit(ic_ellipse, (t_start, t_end_ellipse), t_eval_ellipse, gm_val=GM)
    x_ellipse, y_ellipse = sol_ellipse.y[0], sol_ellipse.y[1]
    energy_ellipse = calculate_energy(sol_ellipse.y.T, GM, mass_particle)
    Lz_ellipse = calculate_angular_momentum(sol_ellipse.y.T, mass_particle)
    print(f"Ellipse: Initial E = {energy_ellipse[0]:.3f}, Initial Lz = {Lz_ellipse[0]:.3f}")
    print(f"Ellipse: Final E = {energy_ellipse[-1]:.3f}, Final Lz = {Lz_ellipse[-1]:.3f} (Energy/Ang. Mom. Conservation Check)")

    # Case 2: Parabolic Orbit (E = 0)
    # For E=0, v_escape = sqrt(2*GM/r). If x0=1, y0=0, then vy0 = sqrt(2*GM/1) = sqrt(2)
    ic_parabola = [1.0, 0.0, 0.0, np.sqrt(2*GM)]
    t_eval_parabola = np.linspace(t_start, t_end_parabola, n_points)
    sol_parabola = solve_orbit(ic_parabola, (t_start, t_end_parabola), t_eval_parabola, gm_val=GM)
    x_parabola, y_parabola = sol_parabola.y[0], sol_parabola.y[1]
    energy_parabola = calculate_energy(sol_parabola.y.T, GM, mass_particle)
    print(f"Parabola: Initial E = {energy_parabola[0]:.3f}")

    # Case 3: Hyperbolic Orbit (E > 0)
    # If vy0 > v_escape, e.g., vy0 = 1.5 * sqrt(2*GM)
    ic_hyperbola = [1.0, 0.0, 0.0, 1.2 * np.sqrt(2*GM)] # Speed greater than escape velocity
    t_eval_hyperbola = np.linspace(t_start, t_end_hyperbola, n_points)
    sol_hyperbola = solve_orbit(ic_hyperbola, (t_start, t_end_hyperbola), t_eval_hyperbola, gm_val=GM)
    x_hyperbola, y_hyperbola = sol_hyperbola.y[0], sol_hyperbola.y[1]
    energy_hyperbola = calculate_energy(sol_hyperbola.y.T, GM, mass_particle)
    print(f"Hyperbola: Initial E = {energy_hyperbola[0]:.3f}")

    # Plotting the orbits
    plt.figure(figsize=(10, 8))
    plt.plot(x_ellipse, y_ellipse, label=f'Elliptical (E={energy_ellipse[0]:.2f})')
    plt.plot(x_parabola, y_parabola, label=f'Parabolic (E={energy_parabola[0]:.2f})')
    plt.plot(x_hyperbola, y_hyperbola, label=f'Hyperbolic (E={energy_hyperbola[0]:.2f})')
    plt.plot(0, 0, 'ko', markersize=10, label='Central Body (Sun)') # Central body
    plt.title('Orbits in an Inverse-Square Law Gravitational Field')
    plt.xlabel('x (arbitrary units)')
    plt.ylabel('y (arbitrary units)')
    plt.axhline(0, color='gray', lw=0.5)
    plt.axvline(0, color='gray', lw=0.5)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axis('equal') # Crucial for correct aspect ratio of orbits
    plt.show()

    # --- Demonstration for Task 3: Varying Angular Momentum for E < 0 ---
    print("\nDemonstrating varying angular momentum for E < 0...")
    E_target = -0.2 # Target negative energy (must be < 0 for ellipse)
    r0 = 1.5       # Initial distance from center (on x-axis)
    # E = 0.5*m*v0y^2 - GM*m/r0  => v0y = sqrt(2/m * (E_target + GM*m/r0))
    # Ensure (E_target + GM*m/r0) is positive for real v0y
    if E_target + GM * mass_particle / r0 < 0:
        print(f"Error: Cannot achieve E_target={E_target} at r0={r0}. E_target must be > -GM*m/r0.")
        print(f"Required E_target > {-GM*mass_particle/r0}")
    else:
        vy_base = np.sqrt(2/mass_particle * (E_target + GM * mass_particle / r0))
        
        initial_conditions_L = []
        # Lz = m * r0 * vy0. We vary vy0 slightly around vy_base to change Lz while trying to keep E close to E_target.
        # Note: Strictly keeping E constant while varying L means r0 or speed direction must change.
        # Here, we fix r0 and initial velocity direction (along y), so varying vy0 changes both E and L.
        # A more precise way for Task 3 would be to fix E and r_periapsis, then find v_periapsis for different L, 
        # or fix E and vary the launch angle from a fixed r0.
        # For simplicity in this demo, we'll vary vy0, which will slightly alter E too.
        # The project description implies fixing E and varying L. Let's try to achieve that more directly.
        # For a fixed E (<0) and r0, the speed v0 is fixed: v0 = sqrt(2/m * (E + GMm/r0)).
        # We can then vary the angle of v0 to change L = m*r0*v0*sin(alpha), where alpha is angle between r0_vec and v0_vec.
        # Let initial position be (r0, 0). Initial velocity (v0*cos(theta), v0*sin(theta)).
        # Lz = m * (x0*vy0 - y0*vx0) = m * r0 * v0*sin(theta). Energy E = 0.5*m*v0^2 - GMm/r0.
        
        v0_for_E_target = np.sqrt(2/mass_particle * (E_target + GM*mass_particle/r0))
        print(f"For E_target={E_target} at r0={r0}, required speed v0={v0_for_E_target:.3f}")

        plt.figure(figsize=(10, 8))
        plt.plot(0, 0, 'ko', markersize=10, label='Central Body')

        # Launch angles (theta) to vary Lz, keeping v0 (and thus E) constant
        launch_angles_deg = [90, 60, 45] # Degrees from positive x-axis for velocity vector
        
        for i, angle_deg in enumerate(launch_angles_deg):
            angle_rad = np.deg2rad(angle_deg)
            vx0 = v0_for_E_target * np.cos(angle_rad)
            vy0 = v0_for_E_target * np.sin(angle_rad)
            ic = [r0, 0, vx0, vy0]
            
            current_E = calculate_energy(np.array(ic), GM, mass_particle)
            current_Lz = calculate_angular_momentum(np.array(ic), mass_particle)
            print(f"  Angle {angle_deg}deg: Calculated E={current_E:.3f} (Target E={E_target:.3f}), Lz={current_Lz:.3f}")

            sol = solve_orbit(ic, (t_start, t_end_ellipse*1.5), np.linspace(t_start, t_end_ellipse*1.5, n_points), gm_val=GM)
            plt.plot(sol.y[0], sol.y[1], label=f'Lz={current_Lz:.2f} (Launch Angle {angle_deg}°)')

        plt.title(f'Elliptical Orbits with Fixed Energy (E ≈ {E_target:.2f}) and Varying Angular Momentum')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.axhline(0, color='gray', lw=0.5)
        plt.axvline(0, color='gray', lw=0.5)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.axis('equal')
        plt.show()





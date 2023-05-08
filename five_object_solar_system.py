def f(t,y):
    '''This function creates the ODE function for the scipy integrator without rocket burn.'''
    
    G = 6.67e-11
    
    # Defining the masses 
    m = np.array([0])
    for o in range(len(objects)):
        m = np.append(m, objects[o].m)
    
    ###################### 
    # Variable definitions 
    ###################### 
    
    # Sun variable definitions 
    x1 = y[0]
    y1 = y[1]
    z1 = y[2]
    px1 = y[3]
    py1 = y[4]
    pz1 = y[5]    
    
    # Earth variable definitions 
    x2 = y[6]
    y2 = y[7]
    z2 = y[8]
    px2 = y[9]
    py2 = y[10]
    pz2 = y[11]    
    
    # Mars variable definitions 
    x3 = y[12]
    y3 = y[13]
    z3 = y[14]
    px3 = y[15]
    py3 = y[16]
    pz3 = y[17]    
    
    # Jupiter variable definitions 
    x4 = y[18]
    y4 = y[19]
    z4 = y[20]
    px4 = y[21]
    py4 = y[22]
    pz4 = y[23]    
    
    # Spaceship variable definitions 
    x5 = y[24]
    y5 = y[25]
    z5 = y[26]
    px5 = y[27]
    py5 = y[28]
    pz5 = y[29]    

    ######################## 
    # Derivative definitions 
    ######################## 
    
    # Sun derivative definitions 
    dx1 = px1 / m[1]
    dy1 = py1 / m[1]
    dz1 = pz1 / m[1]
    dpx1 = -G * m[1] * ( m[2] * (x1 - x2) / (((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2) ** (3/2)) + m[3] * (x1 - x3) / (((x1 - x3)**2 + (y1 - y3)**2 + (z1 - z3)**2) ** (3/2)) + m[4] * (x1 - x4) / (((x1 - x4)**2 + (y1 - y4)**2 + (z1 - z4)**2) ** (3/2)) + m[5] * (x1 - x5) / (((x1 - x5)**2 + (y1 - y5)**2 + (z1 - z5)**2) ** (3/2)) )
    dpy1 = -G * m[1] * ( m[2] * (y1 - y2) / (((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2) ** (3/2)) + m[3] * (y1 - y3) / (((x1 - x3)**2 + (y1 - y3)**2 + (z1 - z3)**2) ** (3/2)) + m[4] * (y1 - y4) / (((x1 - x4)**2 + (y1 - y4)**2 + (z1 - z4)**2) ** (3/2)) + m[5] * (y1 - y5) / (((x1 - x5)**2 + (y1 - y5)**2 + (z1 - z5)**2) ** (3/2)) )
    dpz1 = -G * m[1] * ( m[2] * (z1 - z2) / (((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2) ** (3/2)) + m[3] * (z1 - z3) / (((x1 - x3)**2 + (y1 - y3)**2 + (z1 - z3)**2) ** (3/2)) + m[4] * (z1 - z4) / (((x1 - x4)**2 + (y1 - y4)**2 + (z1 - z4)**2) ** (3/2)) + m[5] * (z1 - z5) / (((x1 - x5)**2 + (y1 - y5)**2 + (z1 - z5)**2) ** (3/2)) )
    
    # Earth derivative definitions 
    dx2 = px2 / m[2]
    dy2 = py2 / m[2]
    dz2 = pz2 / m[2]
    dpx2 = -G * m[2] * ( m[1] * (x2 - x1) / (((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2) ** (3/2)) + m[3] * (x2 - x3) / (((x2 - x3)**2 + (y2 - y3)**2 + (z2 - z3)**2) ** (3/2)) + m[4] * (x2 - x4) / (((x2 - x4)**2 + (y2 - y4)**2 + (z2 - z4)**2) ** (3/2)) + m[5] * (x2 - x5) / (((x2 - x5)**2 + (y2 - y5)**2 + (z2 - z5)**2) ** (3/2)) )
    dpy2 = -G * m[2] * ( m[1] * (y2 - y1) / (((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2) ** (3/2)) + m[3] * (y2 - y3) / (((x2 - x3)**2 + (y2 - y3)**2 + (z2 - z3)**2) ** (3/2)) + m[4] * (y2 - y4) / (((x2 - x4)**2 + (y2 - y4)**2 + (z2 - z4)**2) ** (3/2)) + m[5] * (y2 - y5) / (((x2 - x5)**2 + (y2 - y5)**2 + (z2 - z5)**2) ** (3/2)) )
    dpz2 = -G * m[2] * ( m[1] * (z2 - z1) / (((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2) ** (3/2)) + m[3] * (z2 - z3) / (((x2 - x3)**2 + (y2 - y3)**2 + (z2 - z3)**2) ** (3/2)) + m[4] * (z2 - z4) / (((x2 - x4)**2 + (y2 - y4)**2 + (z2 - z4)**2) ** (3/2)) + m[5] * (z2 - z5) / (((x2 - x5)**2 + (y2 - y5)**2 + (z2 - z5)**2) ** (3/2)) )
    
    # Mars derivative definitions 
    dx3 = px3 / m[3]
    dy3 = py3 / m[3]
    dz3 = pz3 / m[3]
    dpx3 = -G * m[3] * ( m[1] * (x3 - x1) / (((x3 - x1)**2 + (y3 - y1)**2 + (z3 - z1)**2) ** (3/2)) + m[2] * (x3 - x2) / (((x3 - x2)**2 + (y3 - y2)**2 + (z3 - z2)**2) ** (3/2)) + m[4] * (x3 - x4) / (((x3 - x4)**2 + (y3 - y4)**2 + (z3 - z4)**2) ** (3/2)) + m[5] * (x3 - x5) / (((x3 - x5)**2 + (y3 - y5)**2 + (z3 - z5)**2) ** (3/2)) )
    dpy3 = -G * m[3] * ( m[1] * (y3 - y1) / (((x3 - x1)**2 + (y3 - y1)**2 + (z3 - z1)**2) ** (3/2)) + m[2] * (y3 - y2) / (((x3 - x2)**2 + (y3 - y2)**2 + (z3 - z2)**2) ** (3/2)) + m[4] * (y3 - y4) / (((x3 - x4)**2 + (y3 - y4)**2 + (z3 - z4)**2) ** (3/2)) + m[5] * (y3 - y5) / (((x3 - x5)**2 + (y3 - y5)**2 + (z3 - z5)**2) ** (3/2)) )
    dpz3 = -G * m[3] * ( m[1] * (z3 - z1) / (((x3 - x1)**2 + (y3 - y1)**2 + (z3 - z1)**2) ** (3/2)) + m[2] * (z3 - z2) / (((x3 - x2)**2 + (y3 - y2)**2 + (z3 - z2)**2) ** (3/2)) + m[4] * (z3 - z4) / (((x3 - x4)**2 + (y3 - y4)**2 + (z3 - z4)**2) ** (3/2)) + m[5] * (z3 - z5) / (((x3 - x5)**2 + (y3 - y5)**2 + (z3 - z5)**2) ** (3/2)) )
    
    # Jupiter derivative definitions 
    dx4 = px4 / m[4]
    dy4 = py4 / m[4]
    dz4 = pz4 / m[4]
    dpx4 = -G * m[4] * ( m[1] * (x4 - x1) / (((x4 - x1)**2 + (y4 - y1)**2 + (z4 - z1)**2) ** (3/2)) + m[2] * (x4 - x2) / (((x4 - x2)**2 + (y4 - y2)**2 + (z4 - z2)**2) ** (3/2)) + m[3] * (x4 - x3) / (((x4 - x3)**2 + (y4 - y3)**2 + (z4 - z3)**2) ** (3/2)) + m[5] * (x4 - x5) / (((x4 - x5)**2 + (y4 - y5)**2 + (z4 - z5)**2) ** (3/2)) )
    dpy4 = -G * m[4] * ( m[1] * (y4 - y1) / (((x4 - x1)**2 + (y4 - y1)**2 + (z4 - z1)**2) ** (3/2)) + m[2] * (y4 - y2) / (((x4 - x2)**2 + (y4 - y2)**2 + (z4 - z2)**2) ** (3/2)) + m[3] * (y4 - y3) / (((x4 - x3)**2 + (y4 - y3)**2 + (z4 - z3)**2) ** (3/2)) + m[5] * (y4 - y5) / (((x4 - x5)**2 + (y4 - y5)**2 + (z4 - z5)**2) ** (3/2)) )
    dpz4 = -G * m[4] * ( m[1] * (z4 - z1) / (((x4 - x1)**2 + (y4 - y1)**2 + (z4 - z1)**2) ** (3/2)) + m[2] * (z4 - z2) / (((x4 - x2)**2 + (y4 - y2)**2 + (z4 - z2)**2) ** (3/2)) + m[3] * (z4 - z3) / (((x4 - x3)**2 + (y4 - y3)**2 + (z4 - z3)**2) ** (3/2)) + m[5] * (z4 - z5) / (((x4 - x5)**2 + (y4 - y5)**2 + (z4 - z5)**2) ** (3/2)) )
    
    # Spaceship derivative definitions 
    dx5 = px5 / m[5]
    dy5 = py5 / m[5]
    dz5 = pz5 / m[5]
    dpx5 = -G * m[5] * ( m[1] * (x5 - x1) / (((x5 - x1)**2 + (y5 - y1)**2 + (z5 - z1)**2) ** (3/2)) + m[2] * (x5 - x2) / (((x5 - x2)**2 + (y5 - y2)**2 + (z5 - z2)**2) ** (3/2)) + m[3] * (x5 - x3) / (((x5 - x3)**2 + (y5 - y3)**2 + (z5 - z3)**2) ** (3/2)) + m[4] * (x5 - x4) / (((x5 - x4)**2 + (y5 - y4)**2 + (z5 - z4)**2) ** (3/2)) )
    dpy5 = -G * m[5] * ( m[1] * (y5 - y1) / (((x5 - x1)**2 + (y5 - y1)**2 + (z5 - z1)**2) ** (3/2)) + m[2] * (y5 - y2) / (((x5 - x2)**2 + (y5 - y2)**2 + (z5 - z2)**2) ** (3/2)) + m[3] * (y5 - y3) / (((x5 - x3)**2 + (y5 - y3)**2 + (z5 - z3)**2) ** (3/2)) + m[4] * (y5 - y4) / (((x5 - x4)**2 + (y5 - y4)**2 + (z5 - z4)**2) ** (3/2)) )
    dpz5 = -G * m[5] * ( m[1] * (z5 - z1) / (((x5 - x1)**2 + (y5 - y1)**2 + (z5 - z1)**2) ** (3/2)) + m[2] * (z5 - z2) / (((x5 - x2)**2 + (y5 - y2)**2 + (z5 - z2)**2) ** (3/2)) + m[3] * (z5 - z3) / (((x5 - x3)**2 + (y5 - y3)**2 + (z5 - z3)**2) ** (3/2)) + m[4] * (z5 - z4) / (((x5 - x4)**2 + (y5 - y4)**2 + (z5 - z4)**2) ** (3/2)) )
        
    return np.array([dx1, dy1, dz1, dpx1, dpy1, dpz1, dx2, dy2, dz2, dpx2, dpy2, dpz2, dx3, dy3, dz3, dpx3, dpy3, dpz3, dx4, dy4, dz4, dpx4, dpy4, dpz4, dx5, dy5, dz5, dpx5, dpy5, dpz5])



# Defining in initial conditions
y0 = np.array([0, 0, 0, 0.0, 0.0, 0.0, 0, 150000000000.0, 0, -1.7868e+29, 0.0, 0.0, -225000000000.0, 0, 0, 0.0, -1.44e+28, 0.0, 0, 780000000000.0, 0, -2.6e+31, 0.0, 0.0, 0, 146438000000.0, 0, 18306000000.0, 0.0, 0.0])
t_span = [0,10000000.0]
t_eval = np.linspace(0, 10000000.0, 10000000)

# Running the ODE solver
sol = solve_ivp(f, t_span, y0, method='LSODA', t_eval=t_eval, min_step=1)

# Saving the solutions
t = sol.t
# Assigning solutions for Sun
objects[0].sol_rx = sol.y[0]
objects[0].sol_ry = sol.y[1]
objects[0].sol_rz = sol.y[2]
objects[0].sol_px = sol.y[3]
objects[0].sol_py = sol.y[4]
objects[0].sol_pz = sol.y[5]
objects[0].sol_t = sol.t

# Assigning solutions for Earth
objects[1].sol_rx = sol.y[6]
objects[1].sol_ry = sol.y[7]
objects[1].sol_rz = sol.y[8]
objects[1].sol_px = sol.y[9]
objects[1].sol_py = sol.y[10]
objects[1].sol_pz = sol.y[11]
objects[1].sol_t = sol.t

# Assigning solutions for Mars
objects[2].sol_rx = sol.y[12]
objects[2].sol_ry = sol.y[13]
objects[2].sol_rz = sol.y[14]
objects[2].sol_px = sol.y[15]
objects[2].sol_py = sol.y[16]
objects[2].sol_pz = sol.y[17]
objects[2].sol_t = sol.t

# Assigning solutions for Jupiter
objects[3].sol_rx = sol.y[18]
objects[3].sol_ry = sol.y[19]
objects[3].sol_rz = sol.y[20]
objects[3].sol_px = sol.y[21]
objects[3].sol_py = sol.y[22]
objects[3].sol_pz = sol.y[23]
objects[3].sol_t = sol.t

# Assigning solutions for Spaceship
objects[4].sol_rx = sol.y[24]
objects[4].sol_ry = sol.y[25]
objects[4].sol_rz = sol.y[26]
objects[4].sol_px = sol.y[27]
objects[4].sol_py = sol.y[28]
objects[4].sol_pz = sol.y[29]
objects[4].sol_t = sol.t


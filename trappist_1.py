import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
import parameters as par

#============================================================
# Constants and Conversions
#------------------------------------------------------------

dist_conv   =   par.astronomical_unit                                       # m  -> mAU
mass_conv   =   par.solar_system['sun']['mass'] * par.star['mass_solar']    # kg -> trappist_mass
time_conv   =   (60*60*24)                                                  # s  -> days

G_SI        =   6.67408e-11                                                 # m^3 / kg / s^2
G           =   G_SI * mass_conv * time_conv**2 / dist_conv**3              # mAU^3 / trappist_1 mass / days^2

O           =   np.array([0,0]) # Origin - dist

#============================================================
# Find Initial Conditions
#------------------------------------------------------------

def r_ap(a,e):
    """ finds max distance
    Parameters
    ----------
    a:  semi-major axis
    e:  eccentricity
    Returns
    -------
    r_max
    """
    return a * (1 + e)

def r_ph(a,e):
    """ finds min distance
    Parameters
    ----------
    a:  semi-major axis
    e:  eccentricity
    Returns
    -------
    r_min
    """
    return a * (1 - e)

def v_ap(a,e):
    """ find velocity of at aphelion
    Parameters
    ----------
    a:  semi-major axis
    e:  eccentricity
    Returns
    -------
    v_ap
    """
    return np.sqrt( (G*par.star['mass']/a) * (1-e)/(1+e) )

def initial_conditions(N_b,N_t,dim=2):
    """
    1) creates R and V vectors for Trappist_1 system
    2) calculates initial Conditions
    3) initializes R and V
    4) Returns initialized R and V
    Parameters
    ----------
    N_t:    number of time steps
    N_b:    number of bodies
    dim:    ** number or coordinates
    """

    # create position and velocity arrays
    R   =   np.zeros((N_b,N_t,dim))
    V   =   np.zeros_like(R)

    # find inital distances (xhat), speed (yhat), momentum (yhat), and torque balance (about CM) for planets
    D0_planets  =   np.array([ r_ap(a,e) for a,e in zip(par.planets['semi-major'],par.planets['eccentricity']) ])
    S0_planets  =   np.array([ v_ap(a,e) for a,e in zip(par.planets['semi-major'],par.planets['eccentricity']) ])
    T_planets   =   np.dot(par.planets['mass'],D0_planets)
    P0_planets  =   np.dot(par.planets['mass'],S0_planets)

    # find initial distance (-xhat), and speed (yhat) of Trappist 1
    D0_star     =   -T_planets/par.star['mass']
    S0_star     =   -P0_planets/par.star['mass']

    # inital distances and speeds for system
    D0  =   np.hstack(( D0_star,D0_planets ))
    S0  =   np.hstack(( S0_star,S0_planets ))

    # initialize R
    for i_b,d0 in enumerate(D0):
        R[i_b,0,0] = d0

    # initialize V
    for i_b,s0 in enumerate(S0):
        V[i_b,0,1] = s0

    return R,V

#============================================================
# Parameters Arrays
#------------------------------------------------------------

# time
t_max,dt    =   1000, 1e-3                                                      # days
T           =   np.arange(0,t_max+dt,dt)                                        # days
N           =   len(T)                                                          # number

# bodies
names       =   np.array(['Trappist_1', 'b', 'c', 'd', 'e', 'f', 'g'])          # body names
bodies      =   np.arange(len(names))                                           # body indicies
M           =   np.hstack(( 1 , par.planets['mass'] ))                          # Trappist_1 mass

# initialized position and velocity arrays - dimentions = bodies, time, [x,y]
R,V         =   initial_conditions( len(bodies), len(T), 2)                     # mAU , mAU/day

# array tracking
per1    =   np.arange(11)
per2    =   per1 * (N-1)/10
per2    =   per2.astype(int)

def track(i):
    """ tracks progress of array creation"""
    if any(i == p for p in per2) == True:
        comp = int( i * 100/(N-1) )
        print("%s percent complete" % comp)

#============================================================
# Important Functions
#------------------------------------------------------------

def pos(r):
    """ finds position of object
    Arguments
    ---------
    r: position - np.array
    Returns
    -------
    rmag: magnitude of position vector - float
    rhat: unit direction vector - np.array
    """
    rmag = np.linalg.norm(r)
    assert rmag > 0, "why is |r| <= 0? -- r_hat"
    rhat = r/rmag
    return rmag, rhat

def acceleration(i_ob, i_ag, t_ob, t_ag):
    """ finds the acceleration due to gravity
    Arguments
    ---------
    i_ob:   object index
    i_ag:   agent index
    t_ob:   time index of object
    t_ag:   time index of agent
    Returns
    -------
    acceleration vector - np.array
    """
    r_ob    =   R[i_ob,t_ob]
    r_ag    =   R[i_ag,t_ag]
    if any(( np.array_equal(r_ob,r_ag) , i_ob == i_ag )):
        return O
    r           =   r_ag - r_ob
    rmag, rhat  =   pos(r)
    assert rmag > 0, "rmag can't be <= 0 -- acceleration(i_ob, i_ag, t_ob, t_ag)"
    assert len(rhat) == 2, "rhat should be 2 dimentions only"
    return (G * M[i_ag])/rmag**2 * rhat

#============================================================
# Integrators
#------------------------------------------------------------

def velocity_verlet(R,V,i_ob,i):
    """ velocity verlet integration
    Arguments
    ---------
    R:      position array
    V:      velocity array
    i_ob:   index of object in bodies
    i:      index in T
    Returns
    -------
    fills in all position and velocity arrays of planets
    saves arrays into 'npy' folder
    """
    acc1        =   np.array([ acceleration(i_ob,i_ag,i,i) for i_ag in bodies]).sum(axis=0)
    v_half      =   V[i_ob,i] + (dt/2)*acc1
    R[i_ob,i+1] =   R[i_ob,i] + dt*v_half
    acc2        =   np.array([ acceleration(i_ob,i_ag,i+1,i) for i_ag in bodies]).sum(axis=0)
    V[i_ob,i+1] =   v_half + (dt/2)*acc2

    return R,V

def integrate(T=T,R=R,V=V,integrator=velocity_verlet,saveA=True):
    """
    1) Fills in R and V, saves arrays, returns arrays
    2) Saves T, R, and V arrays into ../npy/ if saveA == True
    3) Returns T, R, V
    """

    print("filling in R and V")
    for i,time in enumerate(T[:-1]):
        track(i)
        for i_ob in bodies:
            R,V = velocity_verlet(R,V,i_ob,i)

    if saveA==True:
        print("saveing T, R, and V")
        np.save('npy/T.npy', T)
        np.save('npy/ignore/R.npy', R)
        np.save('npy/ignore/V.npy', V)

    return T,R,V

#============================================================
# Energy
#------------------------------------------------------------

def kinetic(V,saveA=True):
    """
    1) calculates kinetic energy
    2) saves as npy/KE.npy if saveA == True
    3) returns K with shape = (time,bodies)
    Parameters
    ----------
    V:      velocity array
    saveA:  ** default=True
    """

    N = len(V[0,:,0])
    print("creating KE")
    K = np.zeros((N,len(M)))
    for i in range(N):
        track(i)
        K[i] = (1/2) * M * ( V[:,i,0]**2 + V[:,i,1]**2 )

    if saveA == True:
        np.save('npy/KE.npy', K)

    return K

def potential(R,saveA=True):
    """
    1) calculates potential energy
    2) saves U as npy/UG.npy if saveA == True
    3) returns U with shape (time,bodies)
    Parameters
    ----------
    R:      position array
    saveA:  ** default = True
    """

    def U_grav(i_time,i_ob,i_ag):
        if i_ob == i_ag:
            return 0
        else:
            r = R[i_ag,i_time] - R[i_ob,i_time]
            rmag = np.linalg.norm(r)
            return G * M[i_ob] * M[i_ag] / rmag

    print("creating UG")
    N = len(R[0,:,0])
    U = np.zeros((N,len(M)))
    for i_time in range(N):
        track(i_time)
        for i_ob in bodies:
            U[i_time,i_ob] = np.array([ U_grav(i_time,i_ob,i_ag) for i_ag in bodies ]).sum()

    if saveA == True:
        np.save('npy/UG.npy', U)

    return U

def energy(K,U,saveA=True):
    """
    1) calculates E = K + U
    2) saves E as npy/E.npy if saveA == True
    3) returns E with shape (time)
    Parameters
    ----------
    R:      position array
    V:      velocity array
    saveA:  ** default = True
    """

    print("creating E")
    N = len(R[0,:,0])
    E = np.zeros(N)
    for i in range(N):
        track(i)
        E[i] = K[i].sum() + U[i].sum()

    if saveA == True:
        np.save('npy/E.npy', E)

    return E

def E_error(E,saveA=True):
    """ calculates energy err"""

    print("creating E_err")
    err = (E[1:] - E[0])/E[0]
    err_log = np.log10(err)

    if saveA == True:
        np.save('npy/E_err.npy', err_log)

    return err_log

def momentum(V,saveA=True):
    """ writes momentum array

    Parameters
    ----------
    V:  velocity array
    saveA:  ** saves array if == True
    """

    print("creating P")
    P = np.zeros(( N,2 ))
    for i,time in enumerate(T):
        track(i)
        P[i,0] = np.sum( M * V[:,i,0] )
        P[i,1] = np.sum( M * V[:,i,1] )

    if saveA == True:
        np.save('npy/P.npy', P)

    return P

#============================================================
# Nearby function
#------------------------------------------------------------

def nearby(R, res=1000, theta=1/60):
    """
    1) finds the average number of other planets that can be seen from a given planet at any given time.
    2) being able to see a planet is defined as the planet being close enough to resolve objects 1000km across on its surface

    Parameters
    ----------
    res:    ** size of smallest distinguishable features (km)
    theta:  ** smallest angle the human eye can see      (deg)
    """
    res_mAU = res*1000/dist_conv   # mAU
    theta_rad = np.deg2rad(theta)  # Radians
    d0 = res_mAU/theta_rad         # mAU
    vis = np.zeros((len(bodies),N))
    for i_bod in bodies:
        print("Starting %s" % names[i_bod])
        for i_time in range(N):
            count = 0
            for i_ag in bodies:
                if i_bod == i_ag:
                    continue
                d = R[i_ag, i_time] - R[i_bod, i_time]
                dmag = np.linalg.norm(d)
                if dmag <= d0:
                    count += 1
            vis[i_bod, i_time] = count
    vis_avg = np.array([ np.average([ vis[i_bod,:] ]) for i_bod in bodies ])
    np.save('npy/vis.npy', vis)
    np.save('npy/vis_avg.npy',vis_avg)
    return vis, vis_avg

def probability(vis):
    """ calculates average +/- 1 standard deviation"""
    prob = np.zeros(( len(bodies),2 ))
    for i_bod in bodies:
        print("starting %s" % names[i_bod] )
        prob[i_bod,0] = np.average(vis[i_bod])
        prob[i_bod,1] = np.std(vis[i_bod])
    np.save('npy/prob.npy',prob)
    return prob

#============================================================
# Write Arrays
#------------------------------------------------------------

def write(overwrite=False):
    """
    1) loads arrays
    2) creates and saves arrays if load fails
    3) saves R and V into smaller arrays to be compiled later

    Parameters
    ----------
    overwrite: will create arrays even if they exist if == True
    default = False
    """
    if overwrite == True:

        T,R,V       =   integrate()
        K           =   kinetic(V)
        U           =   potential(R)
        E           =   energy(K,U)
        E_err       =   E_error(E)
        P           =   momentum(V)
        vis,vis_avg =   nearby(R)
        for i in bodies:
            np.save('npy/compile/R_%s.npy' % i, R[i])
            np.save('npy/compile/V_%s.npy' % i, V[i])

    else:

        # compiled arrays
        try:
            T = np.load('npy/T.npy')
            R = np.load('npy/compile/R_0.npy')
            V = np.load('npy/compile/V_0.npy')
        except:
            T,R,V   = integrate()

        try:
            K = np.load('npy/KE.npy')
        except:
            K = kinetic(V)

        try:
            U = np.load('npy/UG.npy')
        except:
            U = potential(R)

        try:
            E = np.load('npy/E.npy')
        except:
            E = energy(K,U)

        try:
            E_err = np.load('npy/E_err.npy')
        except:
            E_err = E_error(E)

        try:
            P = np.load('npy/P.npy')
        except:
            P = momentum(V)
        try:
            vis = np.load('npy/vis.npy')
            vis_avg = np.load('npy/vis_avg.npy')
        except:
            vis, vis_avg = nearby(R)

        # uncompiled arrays
        try:
            np.load('npy/compile/R_0.npy')
        except:
            for i in bodies:
                np.save('npy/compile/R_%s.npy' % i, R[i])
                np.save('npy/compile/V_%s.npy' % i, V[i])

def compile():

    try:
        np.load('npy/R.npy')
        np.load('npy/V.npy')
    except:
        R = np.zeros(( len(bodies), N , 2 ))
        V = np.zeros_like(R)

        for i in bodies:
            R[i] = np.load('npy/compile/R_%s.npy' % i)
            V[i] = np.load('npy/compile/V_%s.npy' % i)

        np.save('npy/R.npy',R)
        np.save('npy/V.npy',V)

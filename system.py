from utils import *

def SLIP_walker_3D(x,u,gr=0):
    """
    The order of the states here is:
    x_m   xdot_m  y_m   ydot_m  z_m   zdot_m  x_t    y_t
    x[0]   x[1]   x[2]   x[3]   x[4]   x[5]   x[6]   x[7]
    
    The order of the controls here is:
    u_s    u_tx   u_ty
    u[0]   u[1]   u[2]
    """
    l_max = 1
    g = 9.81
    k = 100
    m = 1
    l = lambda x: np.sqrt(((x[0]-x[6])**2.0)+((x[2]-x[7])**2.0)+((x[4]-gr)**2.0))
    phi = lambda x: x[4]-gr-(x[4]-gr)*l_max/l(x)
    if phi(x) < 0: # stance
        xdot = np.array([x[1],
                        ((k*(l_max-l(x)))+u[0])*((x[0]-x[6])/(m*l(x))),
                        x[3],
                        ((k*(l_max-l(x)))+u[0])*((x[2]-x[7])/(m*l(x))),
                        x[5],
                        ((k*(l_max-l(x)))+u[0])*((x[4]-gr)/(m*l(x)))-g,
                        0.0,
                        0.0])
    else: # flight
        xdot = np.array([x[1],
                         0.0,
                         x[3],
                         0.0,
                         x[5],
                         -g,
                         x[1]+u[1],
                         x[3]+u[2]])
    return xdot.flatten()
    
def SLIP_walker(x,u):
    l_max = 1
    g = 9.81
    k = 100
    m = 1
    l = lambda x: np.sqrt(((x[0]-x[4])**2.0)+(x[2]**2.0))
    phi = lambda x: x[2]-x[2]*l_max/l(x)
    if phi(x) < 0: # stance
        xdot = np.array([x[1],
                        ((k*(l_max-l(x)))+u[1])*((x[0]-x[4])/(m*l(x))),
                        x[3],
                        ((k*(l_max-l(x)))+u[1])*(x[2]/(m*l(x)))-g,
                        0.0])
    else: # flight
        xdot = np.array([x[1],
                         0.0,
                         x[3],
                         -g,
                         x[1]+u[0]])
    return xdot.flatten()
    
def diff_drive(x,u):
    xvel = [u[0]*np.cos(x[2]),
            u[0]*np.sin(x[2]),
            u[1]]
    return np.array(xvel).flatten()

def single_int(x,u):
    xvel = [u[0],
            u[1]]
    return np.array(xvel).flatten()

def double_int(x,u):
    xvel = [x[2],
            x[3],
            u[0],
            u[1]]
    return np.array(xvel).flatten()

def double_int_1D(x,u):
    xvel = [x[1],
            u[0]]
    return np.array(xvel).flatten()

def quadratic_objective(xvec,uvec,xdes=None,Q=None,R=None):
    if Q is None:
        Q = np.eye(xvec.shape[0])
    if R is None:
        R = np.eye(uvec.shape[0])
    if xdes is None:
        xd = np.zeros(xvec.shape)
    elif len(xdes.shape) == 1:
        xd = np.repeat(xdes.reshape(-1,1),xvec.shape[1],axis=1)

    c = 0
    for i in range(xvec.shape[1]):
        c+=(xvec[:,i]-xd[:,i]).dot(Q).dot((xvec[:,i]-xd[:,i]).T) + uvec[:,i].dot(R).dot(uvec[:,i].T)
    return c

def quadratic_rattling_objective(xvec, uvec, dt=0.05, w1=1, w2=1, coord_fun=None, w_sz=20, ov=1, xdes=None, Q=None, R=None):
    c = w1*quadratic_objective(xvec,uvec,xdes,Q,R)
    if coord_fun is None:
        r = rattling_windows(xvec.T, dt, w_sz, ov)
        c += w2*np.mean(r)
    else:
        r = rattling_windows(coord_fun(xvec).T, dt, w_sz, ov)
        c += w2*np.mean(r)
    return c

def gauss_pdf(x,mean,cov):
    return np.exp(-0.5*(x-mean).dot(np.linalg.inv(cov)).dot((x-mean).T))
     
def bimodal_objective(x,mean1,cov1,mean2,cov2):
    return -(gauss_pdf(x,mean1,cov1) + gauss_pdf(x,mean2,cov2)+1.0)

def bimodal_pdf(x,mean1,cov1,mean2,cov2):
    return (gauss_pdf(x,mean1,cov1) + gauss_pdf(x,mean2,cov2))/2.0

def bimodal_rattling_objective(xvec,uvec,mean1,cov1,mean2,cov2,dt=0.05, w1=1, w2=1, w_sz=20, ov=1,R=None):
    c1 = 0
    c2 = 0
    for i in range(xvec.shape[1]):
        c1 += -w1*(gauss_pdf(xvec[:,i],mean1,cov1) + gauss_pdf(xvec[:,i],mean2,cov2)+1.0)
        if R is not None:
            c1 += uvec[:,i].dot(R).dot(uvec[:,i].T)
    c2 = w2*np.mean(rattling_windows(xvec.T, dt, w_sz, ov))
    return c1 + c2

def double_well_1D(x, a=1, b=1, xloc = 0.0, var_ind=0):
    return a*(x[var_ind]**4.0)-b*((x[var_ind]-xloc)**2.0)

def double_well_objective_1D(xvec, uvec, R=None, a=1, b=1,xloc = 0.0):
    c = 0
    for i in range(xvec.shape[1]):
        c += double_well_1D(xvec[:,i],a,b,xloc)
        if R is not None:
            c += uvec[:,i].dot(R).dot(uvec[:,i].T)
    return c

def double_well_rattling_objective_1D(xvec, uvec, a=1, b=1, xloc = 0.0, dt=0.05, w=1, w_sz=20, ov=1,R=None):
    c = 0
    for i in range(xvec.shape[1]):
        c += double_well_1D(xvec[:,i],a,b,xloc)
        if R is not None:
            c += uvec[:,i].dot(R).dot(uvec[:,i].T)
    c += w*np.mean(rattling_windows(xvec.T, dt, w_sz, ov))
    return c


def SLIP_objective(xvec, uvec, a=1, b=1, xloc = 0.0, var_ind=0, dt=0.05, w=1, w_sz=20, ov=1,xdes=None,Q=None,R=None):
    """
    This objective incorporates a regulator on control effort,
    a quadratic component to allow for maintaining a stable height (i.e., safety),
    and a diffusion component.
    """
    eps = 1e-10
    if xdes is None:
        xd = np.zeros(xvec.shape)
    elif len(xdes.shape) == 1:
        xd = np.repeat(xdes.reshape(-1,1),xvec.shape[1],axis=1)
        
    c = 0
    for i in range(xvec.shape[1]):
        c += double_well_1D(xvec[:,i],a,b,xloc,var_ind)
        if R is not None:
            c += uvec[:,i].dot(R).dot(uvec[:,i].T)
        if Q is not None:
            c += (xvec[:,i]-xd[:,i]).dot(Q).dot((xvec[:,i]-xd[:,i]).T)
            
    if np.abs(w) > eps:
        c += w*np.mean(rattling_windows(xvec[var_ind].T, dt, w_sz, ov))
    return c


def SLIP_objective_3D(xvec, uvec, mean1=np.zeros(2), cov1=np.eye(2), mean2=np.zeros(2), cov2=np.eye(2), var_ind=0, dt=0.05, w1=1, w2=1, w_sz=20, ov=1, xdes=None, Q=None, R=None):
    """
    This objective incorporates a regulator on control effort,
    a quadratic component to allow for maintaining a stable height (i.e., safety),
    and a diffusion component.
    """
    eps = 1e-10
    if xdes is None:
        xd = np.zeros(xvec.shape)
    elif len(xdes.shape) == 1:
        xd = np.repeat(xdes.reshape(-1,1),xvec.shape[1],axis=1)
        
    c1 = 0
    c2 = 0
    c3 = 0
    for i in range(xvec.shape[1]):
        # Cost due to potential
        c1 += -w1*(gauss_pdf(xvec[var_ind,i],mean1,cov1) + gauss_pdf(xvec[var_ind,i],mean2,cov2)+1.0)
        
        # Optional costs on state regulation
        if R is not None:
            c3 += uvec[:,i].dot(R).dot(uvec[:,i].T)
        if Q is not None:
            c3 += (xvec[:,i]-xd[:,i]).dot(Q).dot((xvec[:,i]-xd[:,i]).T)
            
    # Cost due to diffusion
    if np.abs(w2) > eps:
        c2 += w2*np.mean(rattling_windows(xvec[var_ind].T, dt, w_sz, ov))
    return c1+c2+c3

def NU(x):
    N = 10
    eps = 0.25
    corners = [[-1.0,-1.5],[-1.0,1.5],[1.0,1.5],[1.0,-1.5]]
    dist = lambda xv,v: np.sqrt((xv[0]-v[0])**2.0+(xv[1]-v[1])**2.0)
    seg1_x = np.linspace(corners[0][0],corners[1][0],N)
    seg1_y = np.linspace(corners[0][1],corners[1][1],N)
    seg2_x = np.linspace(corners[1][0],corners[3][0],N)
    seg2_y = np.linspace(corners[1][1],corners[3][1],N)
    seg3_x = np.linspace(corners[3][0],corners[2][0],N)
    seg3_y = np.linspace(corners[3][1],corners[2][1],N)
    N_x = np.hstack([seg1_x[:-1],seg2_x[:-1],seg3_x])
    N_y = np.hstack([seg1_y[:-1],seg2_y[:-1],seg3_y])
    NU = np.array(tuple(zip(N_x,N_y)))
    c = 0
    for ve in NU:
        d = dist(x,ve)
        if d < eps:
            c += d**2.0
        else:
            c += 3000
    return c

def NU_objective(xvec, uvec, var_ind=0, dt=0.05, w1=1, w2=1, w_sz=20, ov=1, xdes=None, Q=None, R=None):
    eps = 1e-10
    if xdes is None:
        xd = np.zeros(xvec.shape)
    elif len(xdes.shape) == 1:
        xd = np.repeat(xdes.reshape(-1,1),xvec.shape[1],axis=1)
        
    c1 = 0
    c2 = 0
    c3 = 0
    for i in range(xvec.shape[1]):
        # Cost due to potential
        c1 += w1*NU(xvec[var_ind,i])
    
        # Optional costs on state regulation
        if R is not None:
            c3 += uvec[:,i].dot(R).dot(uvec[:,i].T)
        if Q is not None:
            c3 += (xvec[:,i]-xd[:,i]).dot(Q).dot((xvec[:,i]-xd[:,i]).T)
            
    # Cost due to diffusion
    if np.abs(w2) > eps:
        c2 += w2*np.mean(rattling_windows(xvec[var_ind].T, dt, w_sz, ov))
    return c1+c2+c3

def image_objective(xvec, uvec, image, sample_pts, extent = [[-1,1],[-1,1]], var_ind=[0,1], dt=0.05, w1=1, w2=1, w_sz=20, ov=1, xdes=None, Q=None, R=None):
    eps = 1e-10
    if xdes is None:
        xd = np.zeros(xvec.shape)
    elif len(xdes.shape) == 1:
        xd = np.repeat(xdes.reshape(-1,1),xvec.shape[1],axis=1)
        
    c1 = 0
    c2 = 0
    c3 = 0
    for i in range(xvec.shape[1]):
        # Cost due to potential
        val = interp_img(xvec[var_ind,i],image,extent)
        if val == 1:
            dist,_ = closest_point(xvec[var_ind,i],sample_pts)
            c1 += w1*(1.0+dist)**10.0
            
        # Optional costs on state regulation
        if R is not None:
            c3 += uvec[:,i].dot(R).dot(uvec[:,i].T)
        if Q is not None:
            c3 += (xvec[:,i]-xd[:,i]).dot(Q).dot((xvec[:,i]-xd[:,i]).T)
            
    # Cost due to diffusion
    if np.abs(w2) > eps:
        c2 += w2*np.mean(rattling_windows(xvec[var_ind].T, dt, w_sz, ov))
    return c1+c2+c3
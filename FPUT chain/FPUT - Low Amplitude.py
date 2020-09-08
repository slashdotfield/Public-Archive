import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

## define constants

N = 32
alpha = 1./4.
h =1E-3
k_max = 5
totaltime = 12000
maximum = int(totaltime/h)
factor = np.sqrt(2.0/N+1)

def RK4(x,v,dx,dv,h):
    k1x = h*dx(x,v)
    k1v = h*dv(x,v)
    
    k2x = h*dx(x+k1x/2,v+k1v/2)
    k2v = h*dv(x+k1x/2,v+k1v/2)
    
    k3x = h*dx(x+k2x/2,v+k2v/2)
    k3v = h*dv(x+k2x/2,v+k2v/2)
    
    k4x = h*dx(x+k3x,v+k3v)
    k4v = h*dv(x+k3x,v+k3v)
    
    return x + (k1x + 2*k2x + 2*k3x + k4x)/6,v + (k1v + 2*k2v + 2*k3v + k4v)/6
    
def dxdt(x,v):
    return v
    
def dvdt(x,v):
    output = [0.0]
    for i in range(1,len(x)-1):
        x1 = x[i-1]
        x2 = x[i]
        x3 = x[i+1]
        output.append((x1 - 2*x2 + x3) + alpha*(x3 - x2)**2 - alpha*(x2-x1)**2)
    output.append(0.0)
    return np.array(output)
    
def get_frame(x,T,Q,Q_prime,counter):
    fig = plt.figure()
    fig.set_figheight(7)
    fig.set_figwidth(7)
    gridspec.GridSpec(8,8)
    
    ##plot x
    plt.subplot2grid((8,8), (0,0), colspan=8, rowspan=5)
    plt.title('$\\alpha$-FPUT Spring Experiment')
    plt.xlabel('$x_i$')
    plt.ylabel('Displacement')
    plt.plot(np.linspace(0,N,N+1),x,'--r')
    plt.plot(np.linspace(0,N,N+1),x,'.k')
    plt.xlim(0,N)
    plt.ylim(-11,11)
    
    ## plot energies    
    plt.subplot2grid((8,8), (6,0), colspan=8, rowspan=2)
    plt.title('Fourier mode energies')
    plt.xlabel('t')
    plt.ylabel('E')
    for k in range(1,k_max+1):
        plt.plot(T,( np.array(Q_prime[k-1])**2 + (2*np.sin(np.pi*k/2/(N+1))*np.array(Q[k-1]))**2 )/2,label=str(k),linewidth=1.0)
        plt.plot([T[-1]],[(Q_prime[k-1][-1]**2 + (2*np.sin(np.pi*k/2/(N+1))*Q[k-1][-1])**2)/2],'.k')
    plt.xlim(0,totaltime)
    plt.ylim(-0.025,130)
    plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
    
    plt.savefig(str(counter)+'.png')
    plt.close()
    
    
#high amplitude
#x = 10*np.sin(np.linspace(0,N,N+1)*np.pi/(N))

#low amplitude
x = np.sin(np.linspace(0,N,N+1)*np.pi/(N))

#k = 4
#x = np.sin(np.linspace(0,N,N+1)*4*np.pi/(N))

v = np.zeros(N+1)

Q = []
Q_prime = []
for k in range(1,k_max+1):
    output1 = 0.0
    output2 = 0.0
    for index in range(1,len(x)):
        output1 += x[index]*np.sin(np.pi*k*index/(N))
        output2 += v[index]*np.sin(np.pi*k*index/(N))
    Q.append([output1*factor])
    Q_prime.append([output2*factor])
T = [0.0]

counter = 0
get_frame(x,T,Q,Q_prime,counter)
for INDEX in range(maximum+1):
    x,v = RK4(x,v,dxdt,dvdt,h)
    if INDEX % 1000 == 0:
        T.append(h*INDEX)
        for k in range(1,k_max+1):
            output1 = 0.0
            output2 = 0.0
            for index in range(1,len(x)):
                output1 += x[index]*np.sin(np.pi*k*index/(N))
                output2 += v[index]*np.sin(np.pi*k*index/(N))
            Q[k-1].append(output1*factor)
            Q_prime[k-1].append(output2*factor)
        get_frame(x,T,Q,Q_prime,counter)
        counter += 1
    if INDEX % 1000 == 0:
        print(INDEX*h)


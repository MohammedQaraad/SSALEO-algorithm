# Comparing SSALEO as a Scalable Large Scale Global Optimization Algorithm to High-Performance algorithms for Real-world Constrained Optimization Benchmark
# More details about the algorithm are in [please cite the original paper ]
# Mohammed Qaraad , Souad Amjad, Nazar K. Hussein ,Seyedali Mirjalili ,Nadhir Ben Halima, and Mostafa A. Elhosseini.
# IEEE ACCESS,  2022, DOI : 10.1109/ACCESS.2022.3202894


import random
import numpy
import math
import time
import numpy as np
import matplotlib.pyplot as plt

def objective_Fun (x):
    return 20+x[0]**2-10.*np.cos(2*3.14159*x[0])+x[1]**2-10*np.cos(2*3.14159*x[1])

def SSALEO(objf, lb, ub, dim, N, Max_iteration):

    # Max_iteration=1000
    # lb=-100
    # ub=100
    # dim=30
    #N = 50  # Number of search agents
    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim
    Convergence_curve = numpy.zeros(Max_iteration)

    # Initialize the positions of salps
    SalpPositions = numpy.zeros((N, dim))
    for i in range(dim):
        SalpPositions[:, i] = numpy.random.uniform(0, 1, N) * (ub[i] - lb[i]) + lb[i]
    SalpFitness = numpy.full(N, float("inf"))

    FoodPosition = numpy.zeros(dim)
    FoodFitness = float("inf")
    # Moth_fitness=numpy.fell(float("inf"))

#     s = solution()

#     print('SSAELO is optimizing  "' + objf.__name__ + '"')

    timerStart = time.time()
    
    for i in range(0, N):
        # evaluate moths
        SalpFitness[i] = objf(SalpPositions[i, :])

    sorted_salps_fitness = numpy.sort(SalpFitness)
    I = numpy.argsort(SalpFitness)

    Sorted_salps = numpy.copy(SalpPositions[I, :])

    FoodPosition = numpy.copy(Sorted_salps[0, :])
    FoodFitness = sorted_salps_fitness[0]

    Iteration = 1

    # Main loop
    while Iteration < Max_iteration:

        # Number of flames Eq. (3.14) in the paper
        # Flame_no=round(N-Iteration*((N-1)/Max_iteration));

        c1 = 2 * math.exp(-((4 * Iteration / Max_iteration) ** 2))
        beta = 0.2+(1.2-0.2)*(1-(Iteration/Max_iteration)**3)**2    #                       % Eq.(14.2)
        alpha = abs(beta*math.sin((3*math.pi/2+math.sin(3*math.pi/2*beta))));  #            % Eq.(14.1)
        
        # Eq. (3.2) in the paper

        for i in range(0, N):

            SalpPositions = numpy.transpose(SalpPositions)

            if i<N/2:
                for j in range(0, dim):
                    c2 = random.random()
                    c3 = random.random()
                    # Eq. (3.1) in the paper
                    if c3 < 0.5:
                        SalpPositions[j, i] = FoodPosition[j] + c1 * (
                            (ub[j] - lb[j]) * c2 + lb[j]
                        )
                    else:
                        SalpPositions[j, i] = FoodPosition[j] - c1 * (
                            (ub[j] - lb[j]) * c2 + lb[j]
                        )

                    ####################

            elif i>=N/2 and i<N+1:
                point1 = SalpPositions[:, i - 1]
                point2 = SalpPositions[:, i]

                SalpPositions[:, i] = (point2 + point1) / 2
                # Eq. (3.4) in the paper


            # Local escaping operator(LEO)
            rand = random.sample(range(N - 1), 3)
            k =  int(rand[0])
            r1 = int(rand[1])
            r2 = int(rand[2])
            
            #k=int(k2[1])
            f1 = -1+(1-(-1))*random.random()
            f2 = -1+(1-(-1))*random.random();         
            ro = alpha*(2*random.random()-1);
            Xk = numpy.random.uniform(lb,ub,dim)    #;%lb+(ub-lb).*rand(1,nV);       % Eq.(28.8)
            X = numpy.zeros((2, dim))
            Xnew = numpy.zeros((1,dim))
            for m in range(dim):
                X[:, m] = numpy.random.uniform(0,1, 2) * (ub[m] - lb[m]) + lb[m]
                
            if random.random()<0.5:
                L1=1;
            else:
                L1=0    
            u1 = L1*2*random.random()+(1-L1)*1;
            u2 = L1*random.random()+(1-L1)*1;
            u3 = L1*random.random()+(1-L1)*1; 
            if random.random()<0.5:
                L2=1;
            else:
                L2=0  
                
            Xp = (1-L2)*SalpPositions[:,k]+(L2)*Xk;       
            if u1<0.5:
                Xnew = SalpPositions[:, i] + f1*(u1*FoodPosition-u2*Xp)+f2*ro*(u3*(X[0,:]-X[1,:])+u2*(SalpPositions[:,r1]-SalpPositions[:,r2]))/2
            else:
                Xnew = FoodPosition  + f1*(u1*FoodPosition-u2*Xp)+f2*ro*(u3*(X[0,:]-X[1,:])+u2*(SalpPositions[:,r1]-SalpPositions[:,r2]))/2
            Xnew=numpy.clip(Xnew, lb, ub)
            SalpPositions[:, i] = numpy.clip(SalpPositions[:, i], lb, ub)
            SalpPositions = numpy.transpose(SalpPositions)
            SalpFitness[i] = objf(SalpPositions[i, :])
            Xnew_Cost=objf(Xnew)
            if SalpFitness[i] > Xnew_Cost:
                SalpPositions[i,:]=Xnew 
                SalpFitness[i] = Xnew_Cost
            if SalpFitness[i] < FoodFitness:
                FoodPosition = numpy.copy(SalpPositions[i, :])
                FoodFitness = SalpFitness[i]            
            

#         for i in range(0, N):

#             # Check if salps go out of the search spaceand bring it back
#             for j in range(dim):
#                 SalpPositions[i, j] = numpy.clip(SalpPositions[i, j], lb[j], ub[j])

#             SalpFitness[i] = objf(SalpPositions[i, :])

#             if SalpFitness[i] < FoodFitness:
#                 FoodPosition = numpy.copy(SalpPositions[i, :])
#                 FoodFitness = SalpFitness[i]

        # Display best fitness along the iteration
#        if Iteration % 1 == 0:
#             print(
#                 [
#                     "At iteration "
#                     + str(Iteration)
#                     + " the best fitness is "
#                     + str(FoodFitness)
#                 ]
#             )

        Convergence_curve[Iteration] = FoodFitness

        Iteration = Iteration + 1

    return Convergence_curve


Max_iterations=50  # Maximum Number of Iterations
swarm_size = 30 # Number of salps
LB=-10  #lower bound of solution
UB=10   #upper bound of solution
Dim=2 #problem dimensions
NoRuns=100  # Number of runs
ConvergenceCurve=np.zeros((Max_iterations,NoRuns))
for r in range(NoRuns):
    result = SSALEO(objective_Fun, LB, UB, Dim, swarm_size, Max_iterations)
    ConvergenceCurve[:,r]=result
# Plot the convergence curves of all runs
idx=range(Max_iterations)
fig= plt.figure()

#3-plot
ax=fig.add_subplot(111)
for i in range(NoRuns):
    ax.plot(idx,ConvergenceCurve[:,i])
plt.title('Convergence Curve of the SSALEO Optimizer', fontsize=12)
plt.ylabel('Fitness')
plt.xlabel('Iterations')
plt.show()

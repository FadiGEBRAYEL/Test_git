
from numpy.core.records import array
import symengine
import numpy 
import math
import matplotlib.pyplot as plt
# initialisation
# q1=1
# q2=1
# q3=1
R1=5
R2=5
R3=5
#calcule de f(q)
def fonction(q):
    fct=([R1*math.cos(q[0])+R2*math.cos(q[0]+q[1])+R3*math.cos(q[0]+q[1]+q[2])],
         [R1*math.sin(q[0])+R2*math.sin(q[0]+q[1])+R3*math.sin(q[0]+q[1]+q[2])],
         [q[0]+q[1]+q[2]])

    fct=numpy.array(fct)
    # print('je suis dans fonction et voila',fct.shape)

    return(fct) 


# print('fct',fonction(1,1,1))  

#calcule analytique du jacobienne  
vars = symengine.symbols('q1 q2 q3') # Define x and y variables
f = symengine.sympify(['R1*cos(q1)+R2*cos(q1+q2)+R3*cos(q1+q2-q3)', 'R1*sin(q1)+R2*sin(q1+q2)+R3*sin(q1+q2-q3)','q1+q2-q3']) # Define function
J = symengine.zeros(len(f),len(vars)) # Initialise Jacobian matrix

 # Fill Jacobian matrix with entries
for i, fi in enumerate(f):
    for j, s in enumerate(vars):
        J[i,j] = symengine.diff(fi, s)
# print(J)


# entre la jacobienne deja calcule
def direct(q,X_d):
    fct=fonction(q)
    jac =([-R1*math.sin(q[0]) - R3*math.sin(q[0] + q[1] + q[2]) - math.sin(q[0] + q[1])*R2, -R3*math.sin(q[0] +q[1] + q[2]) - math.sin(q[0] + q[1])*R2, -R3*math.sin(q[0] + q[1] + q[2])],
         [R1*math.cos(q[0]) + R3*math.cos(q[0] + q[1] + q[2]) + math.cos(q[0] + q[1])*R2, R3*math.cos(q[0] + q[1] + q[2]) + math.cos(q[0] + q[1])*R2, R3*math.cos(+q[0] + q[1] + q[2])],
         [1, 1, 1])
    jac = numpy.array(jac)
    
    # print(jac)
    #calcule de l'inverse de la jacobienne
    J_inv=numpy.linalg.inv(jac)
   
    #calcule de la direction ,
    direction=numpy.dot(J_inv,(X_d-fct))

    # print('J_inv',J_inv.shape)
    # print('(X_d-fct)',(X_d-fct).shape) 
    # print('(direction',direction.shape) 
    return(direction)   


def calculQ(q,X_d):
    pas=0.8
    # X_d=numpy.array(
    #     ([5],
    #     [5],
    #     [20]))
    R1=5
    R2=5
    R3=5
    i=0

    Qk=q
    err_list=[]
    iter_list=[]
    Qk_1=Qk+pas*direct(Qk,X_d)
    # print(Qk_1)
    # print('je suis avant la boucle')
    # err=numpy.linalg.norm(X_d-fonction(Qk[0],Qk[1],Qk[2]))
    err=fonction(Qk_1).all()-fonction(Qk).all()
    print('la 1ere err est \t',err)
    
    while(err>0.0001 and i<75):
        
        Qk=Qk_1
        Qk_1=Qk+pas*direct(Qk,X_d)
        # print(Qk)
        # print(Qk_1)
        # err=numpy.linalg.norm(X_d-fonction(Qk_1[0],Qk_1[1],Qk_1[2]))
        err=fonction(Qk_1).all()-fonction(Qk).all()
        err_list.append(err)
        iter_list.append(i)
        i=i+1
    
    return(Qk_1)

# def echatillonage(X_A,X_B,nbr_echt):
#     # print(X_B[0]-X_A[0])
#     X_D=numpy.array(
#         ([5],
#         [5],
#         [20]))
#     X_D[0]=(X_B[0]-X_A[0])/nbr_echt #x
#     X_D[1]=(X_B[1]-X_A[1])/nbr_echt #y
#     X_D[2]=(X_B[2]-X_A[2])/nbr_echt #theta
#     return(X_D)

if __name__ == '__main__':
    nbr_echt=10
    i=1
    x_d_list=[]
    X_q_list=[]
    iter_list=[]
    X_b=numpy.array(
        ([25],
        [15],
        [50]))

    X_a=numpy.array(
        ([5],
        [5],
        [20]))
    x_d = X_a+i*((X_b-X_a)/nbr_echt)
    q_int=numpy.array(
        ([3.14/3],
         [3.14/4],
         [3.14/5]))

    q=calculQ(q_int,x_d)
    while(i <= nbr_echt):
        x_d = X_a+i*((X_b-X_a)/nbr_echt)
        q=calculQ(q,x_d)
        fct=fonction(q)
        x_d_list.append(x_d[1])
        X_q_list.append(fct[1])
        iter_list.append(i)
        
        i=i+1
    
    plt.plot(iter_list, x_d_list)
    
    # naming the x axis
    plt.xlabel('iterations')
    # naming the y axis
    plt.ylabel('val_segment')
 
    # giving a title to my graph
    plt.title('AB')
 
    # function to show the plot
    
    plt.plot(iter_list,X_q_list)
    plt.show()
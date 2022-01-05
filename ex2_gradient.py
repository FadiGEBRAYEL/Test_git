
from numpy import linalg
import symengine
import numpy 
import math
import matplotlib.pyplot as plt
# initialisation
# q1=1
# q2=1
# q3=1
R1=1
R2=1
R3=1
#calcule de f(q)
#comment
def fonction(q1, q2 ,q3):
    fct=([R1*math.cos(q1)+R2*math.cos(q1+q2)+R3*math.cos(q1+q2+q3)],
         [R1*math.sin(q1)+R2*math.sin(q1+q2)+R3*math.sin(q1+q2+q3)],
         [q1+q2+q3])

    fct=numpy.array(fct)
    # print('je suis dans fonction et voila',fct.shape)

    return(fct) 


# print('fct',fonction(1,1,1))  

#calcule analytique du jacobienne  
vars = symengine.symbols('q1 q2 q3') # Define x and y variables
f = symengine.sympify(['R1*cos(q1)+R2*cos(q1+q2)+R3*cos(q1+q2+q3)', 'R1*sin(q1)+R2*sin(q1+q2)+R3*sin(q1+q2+q3)','q1+q2+q3']) # Define function
J = symengine.zeros(len(f),len(vars)) # Initialise Jacobian matrix

 # Fill Jacobian matrix with entries
for i, fi in enumerate(f):
    for j, s in enumerate(vars):
        J[i,j] = symengine.diff(fi, s)
print(J)


# entre la jacobienne deja calcule
def direct(q1,q2,q3,X_d):
    fct=fonction(q1,q2,q3)
    jac =([-R1*math.sin(q1) - R3*math.sin(q1 + q2 + q3) - math.sin(q1 + q2)*R2, -R3*math.sin(q1 + q2 + q3) - math.sin(q1 + q2)*R2, -R3*math.sin(q1 + q2 + q3)],
         [R1*math.cos(q1) + R3*math.cos(q1 + q2 + q3) + math.cos(q1 + q2)*R2, R3*math.cos(q1 + q2 + q3) + math.cos(q1 + q2)*R2, R3*math.cos(q1 + q2 + q3)],
         [1, 1, 1])
    jac = numpy.array(jac)
    
    # print(jac)
    #calcule du transpose de la jacobienne
    # J_inv=numpy.linalg.inv(jac)
    # J_inv=numpy.transpose(jac)
    J_inv=numpy.transpose(jac)
    #calcule de la direction ,
    direction=numpy.dot(J_inv,(X_d-fct))


    # AJOUT LOUIS
    
    # print('J_inv',J_inv.shape)
    # print('(X_d-fct)',(X_d-fct).shape) 
    # print('(direction',direction.shape) 
    return(direction)   


if __name__ == '__main__':
    pas=0.9
    q1_int=3.14/3
    q2_int=3.14/4
    q3_int=3.14/5
    X_d=numpy.array(
        ([5],
        [5],
        [20]))
    R1=1
    R2=1
    R3=1
    i=0

    Qk=numpy.array(
        ([q1_int],
        [q2_int],
        [q3_int]))
    err_list=[]
    iter_list=[]
    Qk_1=Qk+pas*direct(q1_int,q2_int,q3_int,X_d)
    # print(Qk_1)
    # print('je suis avant la boucle')
    err=numpy.linalg.norm(X_d-fonction(Qk[0],Qk[1],Qk[2]))
    print('la 1ere err est \t',err)
    
    while(err > 0.0001 and i<75 ):
        # pas=pas0/(i+1)
        
        Qk=Qk_1
        Qk_1=Qk+pas*direct(Qk[0],Qk[1],Qk[2],X_d)# xk+1=xk+pas*direction 
        print(Qk)
        print(Qk_1)
        err=numpy.linalg.norm(X_d-fonction(Qk_1[0],Qk_1[1],Qk_1[2]))
        # err=abs(fonction(Qk_1[0],Qk_1[1],Qk_1[2]).all()-fonction(Qk[0],Qk[1],Qk[2]).all())
        err_list.append(err)
        iter_list.append(i)
        print('err dans la boucle',err)
        i=i+1
         
        
    
    print('je suis sortie de la boucle pour i =',i)
    # plotting the points
    plt.plot(iter_list, err_list)
 
    # naming the x axis
    plt.xlabel('iterations')
    # naming the y axis
    plt.ylabel('val_err')
 
    # giving a title to my graph
    plt.title('erreur')
 
    # function to show the plot
    plt.show()


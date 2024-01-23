import sympy as sp
import numpy as np
#define symbolic variables
x_1, x_2, S = sp.symbols('x_1 x_2 S')

#Define the Function
f=x_1**2+x_1*x_2+1/2*(x_2)**2

df_dx1=sp.diff(f,x_1)
df_dx2=sp.diff(f,x_2)
#Define the variables
x_1_value=1
x_2_value=1
X_0= np.array([[x_1_value, x_2_value]])#X_0 Matrix


q_0_Uvalue = df_dx1.subs({x_1: x_1_value, x_2: x_2_value})
q_0_Avalue = df_dx2.subs({x_1: x_1_value, x_2: x_2_value})
q_0 = np.array([[q_0_Uvalue, q_0_Avalue]])## Q0 MATRİX
q_0_transpose = np.transpose(q_0)## Q0 TRANSPOSE

p_0=-q_0
p_0_transpose = np.transpose(p_0) ## PO TRANSPOSE

X_1 = X_0 + S*p_0
X_1_Uvalue = X_1[0,0]
X_1_Avalue = X_1[0,1]

q_1_Uvalue =df_dx1.subs({x_1: X_1_Uvalue, x_2: X_1_Avalue})
q_1_Avalue =df_dx2.subs({x_1: X_1_Uvalue, x_2: X_1_Avalue})
q_1=np.array([[q_1_Uvalue,q_1_Avalue]]) ##Q1 MATRİX
q_1_transpose =np.transpose(q_1) ##Q1 TRANSPOSE
# Multiplication Operation
result_sum = np.dot(p_0, q_1_transpose).sum()

## Find S
if result_sum != 0:
    S_solve = sp.solve(result_sum, S)


## X_1  (without S)
reel_X_1 = X_0+ S_solve*p_0
reel_X_1_Uvalue = reel_X_1[0,0]
reel_X_1_Avalue = reel_X_1[0,1]

## Q_1  (without S)
reel_q_1_Uvalue =df_dx1.subs({x_1: reel_X_1_Uvalue, x_2: reel_X_1_Avalue})
reel_q_1_Avalue =df_dx2.subs({x_1: reel_X_1_Uvalue, x_2: reel_X_1_Avalue})
reel_q_1=np.array([[reel_q_1_Uvalue,reel_q_1_Avalue]])

#q1 transpose
reel_q_1_transpose = np.transpose(reel_q_1)


#find beta
beta = np.dot(reel_q_1, reel_q_1_transpose).sum()/np.dot(q_0, q_0_transpose).sum()

p_1 = -(reel_q_1_transpose)+(beta)*(p_0_transpose)
print(p_1)
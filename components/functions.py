import torch
import math

#quaternion are defined as a 2 element tesnor, first is the scalar the second is a 3 float tensor representing the vector part

def q_prod(q1,q2):
    q1_vector = torch.take(q1,torch.tensor([1,2,3]))
    q2_vector = torch.take(q2,torch.tensor([1,2,3]))
    scalar = q1[0]*q2[0]-torch.dot(q1_vector,q2_vector)
    vector = torch.cross(q1_vector,q2_vector)+torch.mul(q1[0],q2[1])+torch.mul(q2[0],q1[1])
    return torch.cat(scalar,vector) 


def qpu_power(q,w):
    if(q[1]==q[2]==q[3]==0):
        return {1,0,0,0}
    else:
        a_s = torch.acos(torch.clamp(q[0],-1+1e-12,1-1e-12))
        s = torch.cos((w*a_s))
        magnitude = quaternion_magnitude(q)
        mul = torch.sin(a_s)/magnitude
        v1 = q[1]*mul
        v2 = q[2]*mul
        v3 = q[3]*mul
        return [s,v1,v2,v3]

def bias_qpu_power(q,w,b):
    a_s = torch.acos(q[0])
    s = torch.cos((w*a_s)+b)
    magnitude = quaternion_magnitude(q)
    mul = torch.sin(a_s+b)/magnitude
    v1 = q[1]*mul
    v2 = q[2]*mul
    v3 = q[3]*mul
    return [s,v1,v2,v3]

def quaternion_magnitude(q):
    return torch.sqrt(q[0]**2+q[1]**2+q[2]**2+q[3]**2)

def qpu_forward(qpu,weight,bias):
    """"""
    
    

def noramlization(q):
    magnitude = quaternion_magnitude(q)
    q[0]/=magnitude
    q[3]/=magnitude
    q[1]/=magnitude
    q[2]/=magnitude
    return q


if __name__=="__main__":

    q1 = [0.99,2,3,4]
    tq = torch.tensor(q1)
    qpow = qpu_power(tq,2)
    print(qpow)
    print(quaternion_magnitude(qpow))


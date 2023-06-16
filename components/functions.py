import torch
import math

#quaternion are defined as a 4 element tesnor

def q_prod(q1:torch.Tensor,q2:torch.Tensor):
    q1_vector = torch.take(q1,torch.tensor([1,2,3]))
    q2_vector = torch.take(q2,torch.tensor([1,2,3]))
    q1_scalar = torch.take(q1,torch.tensor([0]))
    q2_scalar = torch.take(q2,torch.tensor([0]))
    scalar = q1_scalar*q2_scalar-torch.dot(q1_vector,q2_vector)
    vector = torch.cross(q1_vector,q2_vector)+torch.mul(q1[0],q2[1])+torch.mul(q2[0],q1[1])
    return torch.cat((scalar,vector)) 


def qpu_power(q,w):
    if(q[1]==q[2]==q[3]==0):
        return {1,0,0,0}
    else:
        a_s = torch.acos(torch.clamp(q[0],-1+1e-12,1-1e-12))
        s = torch.cos((w*a_s))
        magnitude = torch.linalg.norm(q)
        mul = torch.sin(a_s)/magnitude
        v1 = q[1]*mul
        v2 = q[2]*mul
        v3 = q[3]*mul
        return [s,v1,v2,v3]

def bias_qpu_power(q,w,b):
    wasb = w*(torch.acos(q[0])+b)
    s = torch.cos(wasb)
    magnitude = torch.linalg.norm(q)
    mul = torch.sin(wasb)/magnitude
    v1 = q[1]*mul
    v2 = q[2]*mul
    v3 = q[3]*mul
    return torch.tensor([s,v1,v2,v3])

def qpu_forward(inputs,weights,bias):
    """"""
    out = torch.tensor([])
    in_channels = weights.shape[-1]
    out_channels = weights.shape[0]

    inputs = torch.reshape(inputs,(in_channels,4))
    for i in range(out_channels):
        weight = weights[i]
        node = bias_qpu_power(inputs[0],weight[0],bias[0])
        for j,input in enumerate(inputs[1:]):
            node = q_prod(node,bias_qpu_power(input,weight[j+1],bias[j+1]))
        out = torch.cat((out,node))

    #out = torch.reshape(out,(out_channels,4))
    return out
    

def normalize(q):
    magnitude = torch.linalg.norm(q)
    # q[0]/=magnitude
    # q[1]/=magnitude
    # q[2]/=magnitude
    q/=magnitude
    return q

def angleAxisMap(q):
    scalar = torch.acos(q[0])
    o_vec = torch.cat(q[1],q[2],q[3])
    n_vec = normalize(o_vec)
    return torch.cat(scalar,n_vec)



if __name__=="__main__":

    q1 = torch.Tensor([])
    q2 = torch.Tensor([1,1,1,2])

    test = torch.cat((q1,q2))
    print(test)
    


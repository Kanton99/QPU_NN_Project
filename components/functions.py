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


def qpu_power(x,y,z,w,weights):
    wabs = torch.mul(weights,(torch.acos(x)))
    x = torch.cos(wabs)
    norms = torch.sqrt(y**2+z**2+w**2)
    mul = torch.sin(wabs)/norms
    y = y * mul
    z = z * mul
    w = w * mul
    return (x,y,z,w)

def bias_qpu_power(x,y,z,w,weights,b):
    theta = (torch.acos(x)+b.unsqueeze(-1))
    
    wabs = weights*theta
    x = torch.cos(wabs)
    norms = torch.sqrt(y**2+z**2+w**2+1e-12)
    mul = torch.sin(wabs)/norms
    y = y * mul
    z = z * mul
    w = w * mul
    return (x,y,z,w)


#Similar to original function due to being the most efficient
def qpu_forward(inputs,weights,bias):
    """"""
    in_channels = inputs.shape[-1]//4
    out_channels = weights.shape[0]

    x,y,z,w = inputs.unsqueeze(-2).split(in_channels,dim=-1)

    x,y,z,w = bias_qpu_power(x,y,z,w,weights[0],bias)
    x,y,z,w = QuaternionRemoveZeros.apply(x,y,z,w)
    x,y,z,w = quaterion_chain_prod(x,y,z,w)

    #inputs = torch.reshape(inputs,(in_channels,4))
    # for i in range(out_channels):
    #     weight = weights[i]
    #     node = bias_qpu_power(inputs[0],weight[0],bias[0])
    #     for j,input in enumerate(inputs[1:]):
    #         node = q_prod(node,bias_qpu_power(input,weight[j+1],bias[j+1]))
    #     out = torch.cat((out,node))

    #out = torch.reshape(out,(out_channels,4))
    ret = torch.cat((x,y,z,w),dim=1)
    return ret
    
def quaterion_chain_prod(x,y,z,w):

    return x,y,z,w
#copied from original code due to being the best implementation
class QuaternionRemoveZeros(torch.autograd.Function):
    """Replace [0, 0, 0, 0] with [1, 0, 0, 0]
    """
    @staticmethod
    def forward(ctx,r,i,j,k):
        norm = r**2+ i**2+ j**2+ k**2
        index = norm == 0
        ctx.save_for_backward(index)
        r[index] = 1
        return r,i,j,k

    @staticmethod
    def backward(ctx,gr,gi,gj,gk):
        index, = ctx.saved_tensors
        gr[index] = 0
        gi[index] = 0
        gj[index] = 0
        gk[index] = 0
        return gr, gi, gj, gk

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

    x = torch.Tensor([[[1,0,0,0],[1,2,3,4],[1,2,3,4],[1,2,3,4]],[[1,0,0,0],[1,2,3,4],[1,2,3,4],[1,2,3,4]]])
    weights = torch.ones(x.shape)
    batchSize = x.shape[0]
    x = x.permute(0,2,1)
    x = x.reshape(batchSize, -1)
  #  x = x[0]
    print(x)
    r,i,j,k = x.unsqueeze(-2).split(4, dim=-1)
    
    #r,i,j,k = qpu_power(r,i,j,k,weights)
    


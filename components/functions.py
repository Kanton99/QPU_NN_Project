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
    theta = torch.acos(x)+b.unsqueeze(-1)
    
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

    x,y,z,w = bias_qpu_power(x,y,z,w,weights,bias)
    x,y,z,w = QuaternionRemoveZeros.apply(x,y,z,w)
    x,y,z,w = quaternion_chained_prod(x,y,z,w,-1)
    ret = torch.cat((x,y,z,w),dim=1)
    return ret

def quaternion_chained_prod_m(r_input, i_input, j_input, k_input, dim, last=None):
    batch_num = r_input.shape[0]
    r_prod = r_input.clone()
    i_prod = i_input.clone()
    j_prod = j_input.clone()
    k_prod = k_input.clone()
    r_out = torch.tensor((32,32))
    i_out = torch.tensor((32,32))
    j_out = torch.tensor((32,32))
    k_out = torch.tensor((32,32))
    for i in range(batch_num):
        for j in range(r_prod.shape[1]):
            s = r_prod[i][j][0]
            x = i_prod[i][j][0]
            y = j_prod[i][j][0]
            z = k_prod[i][j][0]
            mat = torch.tensor([[s,-x,-y,-z],
                                [x,s,z,-y],
                                [y,-z,s,x],
                                [z,y,-x,s]],requires_grad=True)
            for k in range(1,r_prod.shape[2]-1):
                s = r_prod[i][j][k]
                x = i_prod[i][j][k]
                y = j_prod[i][j][k]
                z = k_prod[i][j][k]
                mat2 = torch.tensor([[s,-x,-y,-z],
                                [x,s,z,-y],
                                [y,-z,s,x], 
                                [z,y,-x,s]],requires_grad=True)
                mat = torch.matmul(mat,mat2)
            last = torch.tensor([r_prod[i][j][-1],i_prod[i][j][-1],j_prod[i][j][-1],k_prod[i][j][-1]])
            res = torch.matmul(mat,last)
            r_out[i][j] = res[0]
            i_out[i][j] = res[1]
            j_out[i][j] = res[2]
            k_out[i][j] = res[3]

    return r_out,i_out,j_out,k_out

# this function down to quaternion_chained_prod_grad are copied since they are the fastests implementation for chaining the hamiltonian product
# and our tried code was too slow    
def quaternion_chained_prod(r_input, i_input, j_input, k_input, dim, last=None):
    """
    Chained quaternion product along a dimension (recursive)
    Hamilton product:
    a1 a2 - b1 b2 - c1 c2 - d1 d2 
    + ( a1 b2 + b1 a2 + c1 d2 − d1 c2 ) i
    + ( a1 c2 − b1 d2 + c1 a2 + d1 b2 ) j 
    + ( a1 d2 + b1 c2 − c1 b2 + d1 a2 ) k 
    """
    channel = r_input.shape[dim]
    if channel == 1:
        return r_input.squeeze(dim), i_input.squeeze(dim), j_input.squeeze(dim), k_input.squeeze(dim)
    else:
        # Split into pair(0) and odd(1)
        r_out, i_out, j_out, k_out = r_input.unfold(dim, 2, 2), i_input.unfold(dim, 2, 2), j_input.unfold(dim, 2, 2), k_input.unfold(dim, 2, 2)
        r_pair, r_odd = r_out.select(-1, 0), r_out.select(-1, 1)
        i_pair, i_odd = i_out.select(-1, 0), i_out.select(-1, 1)
        j_pair, j_odd = j_out.select(-1, 0), j_out.select(-1, 1)
        k_pair, k_odd = k_out.select(-1, 0), k_out.select(-1, 1)
        # pair * odd
        r_out, i_out, j_out, k_out = hamilton_product_chunk(r_pair, i_pair, j_pair, k_pair, r_odd, i_odd, j_odd, k_odd)
        # Multiply last
        if channel % 2 == 1:
            last = (r_input.select(dim, -1), i_input.select(dim, -1), j_input.select(dim, -1), k_input.select(dim, -1))
        if r_out.shape[dim] % 2 == 1 and last is not None:
            r_out = torch.cat([r_out,last[0].unsqueeze(dim)],dim=dim)
            i_out = torch.cat([i_out,last[1].unsqueeze(dim)],dim=dim)
            j_out = torch.cat([j_out,last[2].unsqueeze(dim)],dim=dim)
            k_out = torch.cat([k_out,last[3].unsqueeze(dim)],dim=dim)
            last = None
        # Recursion
        r_out, i_out, j_out, k_out = quaternion_chained_prod(r_out, i_out, j_out, k_out, dim, last)
        return r_out, i_out, j_out, k_out

class QuaternionChainedProdFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_r, input_i, input_j, input_k, dim=-1):
        """
        Chained quaternion product along a dimension (for loop)
        Hamilton product:
        a1 a2 - b1 b2 - c1 c2 - d1 d2 
        + ( a1 b2 + b1 a2 + c1 d2 − d1 c2 ) i
        + ( a1 c2 − b1 d2 + c1 a2 + d1 b2 ) j 
        + ( a1 d2 + b1 c2 − c1 b2 + d1 a2 ) k 
        """
        input_r, input_i, input_j, input_k = input_r.clone(), input_i.clone(), input_j.clone(), input_k.clone()
        cumprod_r, cumprod_i, cumprod_j, cumprod_k = quaternion_cumprod_(input_r, input_i, input_j, input_k, dim)
        ctx.save_for_backward(cumprod_r, cumprod_i, cumprod_j, cumprod_k)
        ctx.dim = dim
        return cumprod_r.select(dim, -1), cumprod_i.select(dim, -1), cumprod_j.select(dim, -1), cumprod_k.select(dim, -1)

    @staticmethod
    def backward(ctx, grad_output_r, grad_output_i, grad_output_j, grad_output_k):
        cumprod_r, cumprod_i, cumprod_j, cumprod_k, = ctx.saved_tensors  # L, *
       
        # Compute cumprod of left and right seq for each input, grads are stored in cumprod on the fly to save memory
        grad_chain_r, grad_chain_i, grad_chain_j, grad_chain_k = quaternion_chained_prod_grad_cumprod(cumprod_r, cumprod_i, cumprod_j, cumprod_k, 
                                                                        grad_output_r, grad_output_i, grad_output_j, grad_output_k, dim=ctx.dim)
        
        return grad_chain_r, grad_chain_i, grad_chain_j, grad_chain_k, None

def hamilton_product_chunk(r1, i1, j1, k1, r2, i2, j2, k2):
    """
    Hamilton product
    a1 a2 - b1 b2 - c1 c2 - d1 d2 
    + ( a1 b2 + b1 a2 + c1 d2 − d1 c2 ) i
    + ( a1 c2 − b1 d2 + c1 a2 + d1 b2 ) j 
    + ( a1 d2 + b1 c2 − c1 b2 + d1 a2 ) k 
    """
    r_out, i_out, j_out, k_out = r1 * r2 - i1 * i2 - j1 * j2 - k1 * k2, \
                                 r1 * i2 + i1 * r2 + j1 * k2 - k1 * j2, \
                                 r1 * j2 - i1 * k2 + j1 * r2 + k1 * i2, \
                                 r1 * k2 + i1 * j2 - j1 * i2 + k1 * r2
    return r_out, i_out, j_out, k_out

def quaternion_cumprod_(r, i, j, k, dim):
    """Cumpute quaternion cumpord (inplace)
    """
    seq_len = r.shape[dim]
    cumprod_r = r.split(1, dim)
    cumprod_i = i.split(1, dim)
    cumprod_j = j.split(1, dim)
    cumprod_k = k.split(1, dim)
    for n in range(1, seq_len):
        cr, ci, cj, ck = hamilton_product_chunk(cumprod_r[n - 1], cumprod_i[n - 1], cumprod_j[n - 1], cumprod_k[n - 1], 
                                                cumprod_r[n], cumprod_i[n], cumprod_j[n], cumprod_k[n])
        cumprod_r[n].copy_(cr)
        cumprod_i[n].copy_(ci)
        cumprod_j[n].copy_(cj)
        cumprod_k[n].copy_(ck)
    return r, i, j, k

def quaternion_chained_prod_grad_cumprod(cumprod_r, cumprod_i, cumprod_j, cumprod_k, grad_output_r, grad_output_i, grad_output_j, grad_output_k, dim):
    """Compute grad of quaternion chained prod from cumprod
    Args:
        cumprod_*: *, N, *
        grad_output_*: *, *
    """
    seq_len = cumprod_r.shape[dim]
    # Split shares the origin memory
    grad_output_r = grad_output_r.unsqueeze(dim)
    grad_output_i = grad_output_i.unsqueeze(dim)
    grad_output_j = grad_output_j.unsqueeze(dim)
    grad_output_k = grad_output_k.unsqueeze(dim)

    rl = torch.ones_like(cumprod_r)
    rl.narrow(dim, 1, seq_len - 1).copy_(cumprod_r.narrow(dim, 0, seq_len - 1))
    il = torch.zeros_like(cumprod_i)
    il.narrow(dim, 1, seq_len - 1).copy_(cumprod_i.narrow(dim, 0, seq_len - 1))
    jl = torch.zeros_like(cumprod_j)
    jl.narrow(dim, 1, seq_len - 1).copy_(cumprod_j.narrow(dim, 0, seq_len - 1))
    kl = torch.zeros_like(cumprod_k)
    kl.narrow(dim, 1, seq_len - 1).copy_(cumprod_k.narrow(dim, 0, seq_len - 1))

    rr, ir, jr, kr =  hamilton_product_chunk(cumprod_r, -cumprod_i, -cumprod_j, -cumprod_k, 
                                             cumprod_r.narrow(dim, seq_len - 1, 1), cumprod_i.narrow(dim, seq_len - 1, 1), 
                                             cumprod_j.narrow(dim, seq_len - 1, 1), cumprod_k.narrow(dim, seq_len - 1, 1))

    grad_r, grad_i, grad_j, grad_k = quaternion_chained_prod_grad(rl, il, jl, kl, rr, ir, jr, kr, 
                                                grad_output_r, grad_output_i, grad_output_j, grad_output_k)
    return grad_r, grad_i, grad_j, grad_k

def quaternion_chained_prod_grad(rl, il, jl, kl, rr, ir, jr, kr, grad_output_r, grad_output_i, grad_output_j, grad_output_k):
    grad_input_r = (   rl * rr - il * ir - jl * jr - kl * kr) * grad_output_r + \
                    (- ir * jl + il * jr + rr * kl + rl * kr) * grad_output_k + \
                    (  rr * jl + rl * jr + ir * kl - il * kr) * grad_output_j + \
                    (  rr * il + rl * ir - jr * kl + jl * kr) * grad_output_i

    grad_input_i = ( - rr * il - rl * ir - jr * kl + jl * kr) * grad_output_r + \
                    (- rr * jl + rl * jr - ir * kl - il * kr) * grad_output_k + \
                    (- ir * jl - il * jr + rr * kl - rl * kr) * grad_output_j + \
                    (  rl * rr - il * ir + jl * jr + kl * kr) * grad_output_i

    grad_input_j = ( - rr * jl - rl * jr + ir * kl - il * kr) * grad_output_r + \
                    (  rr * il - rl * ir - jr * kl - jl * kr) * grad_output_k + \
                    (  rl * rr + il * ir - jl * jr + kl * kr) * grad_output_j + \
                    (- ir * jl - il * jr - rr * kl + rl * kr) * grad_output_i

    grad_input_k = ( - ir * jl + il * jr - rr * kl - rl * kr) * grad_output_r + \
                    (  rl * rr + il * ir + jl * jr - kl * kr) * grad_output_k + \
                    (- rr * il + rl * ir - jr * kl - jl * kr) * grad_output_j + \
                    (  rr * jl - rl * jr - ir * kl - il * kr) * grad_output_i
    
    return grad_input_r, grad_input_i, grad_input_j, grad_input_k
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

    x = torch.Tensor([[1,1,1,1],[2,2,2,2]])
    y = torch.Tensor([1,1])
    print(x+y.unsqueeze(-1))
    


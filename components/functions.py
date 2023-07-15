import torch

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

# this function and hamilton_product_chunk are copied since they are the fastests implementation for chaining the hamiltonian product
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

if __name__=="__main__":
    exit(0)
    


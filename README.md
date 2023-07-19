# Quaternion Unit Product
Quaternion Unit Product (QPU) is an innovative way to build Neural Networks when working with quaternion data.
They leverage properties of quaternions and hamiltonian products to provide a better and faster way to train networks with quaternion data as input, such as a skeleton model.

$$QPU(\{q_i\}^N_{i=1} ;\{w_i\}^N_{i=1};b) = \bigotimes^N_{i=1}qpow(q_i,w_i,b)$$

where:

$$q_i=[s_i,v_i]\\
qpow(q_i,w_i,b)=[cos(w_i(arccos(s_i)+b)),\frac{v_i}{\|{v_i}\|}sin(w_i(arccos(s_i)+b))]$$

# Installation
Linux needed to work

Create python environment

```python -m venv venv```

Activate the environment with

```source venv/bin/activate```

Install required dependencies

```pip install -r requirements.txt```

# Usage
To use an example of the results that the method provides just execute the notebook that comes with the code.

If you want to use it in your own code, make a model which flattens each batch of the input, or extend from the MLP_base class in CubeEdge.models.
Then make the model input be 4*"number of elements in a batch", each layer in between with any size, but multiples of 4 are preferred.
THe output should be of the size of the input batch. 

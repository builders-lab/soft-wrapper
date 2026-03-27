import ctypes
import os 

#===================Load C ++ Library=========================

#===path of the .dll file======

LIB_PATH= os.path.join(os.path.dirname(__file__),"cuda.so")

try:
    tensor_lib= ctypes.CDLL(LIB_PATH)
    print("Library uploaded")
except OSError:
    print("ERROR: Library file nahi mili. please put .dll or .so file ")


#========ENUMS AND CONSTRAINTS===============

#Maximum number of supported dimension(#defin TENSOR_MAX_DIMS 8)
TENSOR_MAX_DIMS=8

# enum class tensor_dtype_t
DTYPE_UINT32=0
DTYPE_INT32=1
DTYPE_FLOAT32=4
DTYPE_UINT64=2
DTYPE_INT64=3
DTYPE_FLOAT64=5

# enum class device_type
DEVICE_GPU=0
DEVICE_CPU=1


#======================MAPPING FUNCTIONS ====================


#-----TENSOR INFORMAATION----------
#unit32_t tensor_id(tensor_t* t); this will give yoe the id of the tensor
tensor_lib.tensor_id.argtypes=[ctypes.c_void_p]
tensor_lib.tensor_id.restype=ctypes.c_uint32

# #------MEMORY POOL----------------

# # tensor_pool_t* tensor_pool_create(size_t capacity_bytes);

tensor_lib.tensor_pool_create.argtypes=[ctypes.c_size_t,ctypes.c_bool]
tensor_lib.tensor_pool_create.restype=ctypes.c_void_p


#--------TENSOR CREATION & DATA ------------

##tensor_t* tensor_create(tensor_pool_t *pool, tensor_dtype_t dtype, uint32_t num_dims, uint32_t *dims, void* elems);
tensor_lib.tensor_create.argtypes=[
    ctypes.c_void_p,   #pool
    ctypes.c_int,      #dtype(enmus get converted to int)
    ctypes.c_uint32,   #num_dims
    ctypes.POINTER(ctypes.c_uint32),  #dims(array of uint32)
    ctypes.c_void_p    #elems (pointer to actual data /numpy array)

]
tensor_lib.tensor_create.restype=ctypes.c_void_p


#-------POOL OPERATIONS---------------
# void tensor_pool_zero(tensor_pool_t *pool);
tensor_lib.tensor_pool_zero.argtypes=[ctypes.c_void_p]
tensor_lib.tensor_pool_zero.restype=None

#void tensor_pool_destroy(tensor_pool_t *pool);
tensor_lib.tensor_pool_destroy.argtypes=[ctypes.c_void_p]
tensor_lib.tensor_pool_destroy.restype=None

#void *tensor_pool_alloc(tensor_pool_t *pool,size_t size,unit32_t *id);
tensor_lib.tensor_pool_alloc.argtypes=[
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_uint32)
]
tensor_lib.tensor_pool_alloc.restype=ctypes.c_void_p

# size_t tensor_pool_size(tensor_pool_t *pool)
tensor_lib.tensor_pool_size.argtypes=[ctypes.c_void_p]
tensor_lib.tensor_pool_size.restype=ctypes.c_size_t

# size_t tensor_pool_used(tensor_pool_t * pool);
tensor_lib.tensor_pool_used.argtypes=[ctypes.c_void_p]
tensor_lib.tensor_pool_used.restype=ctypes.c_size_t

# bool tensor_move_device(tensor_t*t,device_type target_device,tensor_pool_t*pool);
# tensor_lib.tensor_move_device.argtypes=[
#     ctypes.c_void_p,
#     ctypes.c_int,
#     ctypes.c_void_p
# ]
# tensor_lib.tensor_move_device.restype=ctypes.c_bool












#--------TENSOR DATA ADN UTILs----------

# void*tensor_get_data(tensor_t* t);
tensor_lib.tensor_get_data.argtypes=[ctypes.c_void_p]
tensor_lib.tensor_get_data.restype=ctypes.c_void_p

# unit8_t tensor_get_ndims(tensor_t*t);
tensor_lib.tensor_get_ndims.argtypes=[ctypes.c_void_p]
tensor_lib.tensor_get_ndims.restype=ctypes.c_uint8

#unit32_t* tensor_get_dims(tensor_t *t);
tensor_lib.tensor_get_dims.argtypes=[ctypes.c_void_p]
tensor_lib.tensor_get_dims.restype=ctypes.POINTER(ctypes.c_uint32)


# void tensor_print_data(tensor_t *t);
tensor_lib.tensor_print_data.argtypes=[ctypes.c_void_p]
tensor_lib.tensor_print_data.restype=None

# bool tensor_fill_random_normal(tensor_t*t,float mean,float std_dev);
tensor_lib.tensor_fill_random_normal.argtypes=[
    ctypes.c_void_p,
    ctypes.c_float,
    ctypes.c_float
]
tensor_lib.tensor_fill_random_normal.restype=ctypes.c_bool





#------MATH OPERATIONS----------------

#tensor_t* tensor_mul(tensor_pool_t*pool,tensor_t*x,tensor_t*y);
tensor_lib.tensor_mul.argtypes=[
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p
]
tensor_lib.tensor_mul.restype=ctypes.c_void_p

#tensor_t* tensor_mul_naive(tensor_pool_t*pool,tensor_t*x,tensor_t *y);
tensor_lib.tensor_mul_naive.argtypes=[
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p
]
tensor_lib.tensor_mul_naive.restype=ctypes.c_void_p



#tensor_t* tensor_transpose(tensor_pool_t*pool,tensor_t*a);
tensor_lib.tensor_transpose.argtypes=[
    ctypes.c_void_p,
    ctypes.c_void_p
]
tensor_lib.tensor_transpose.restype=ctypes.c_void_p

# tensor_t* tensor_add(tensor_pool_t*pool,tensor_t*x,tensor_t*y);
tensor_lib.tensor_add.argtypes=[
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p
]
tensor_lib.tensor_add.restype=ctypes.c_void_p

# tensor_t*tensor_add_bias(tensor_pool_t*pool,const tensor_t*xw,const tensor_t*bias);
# tensor_lib.tensor_add_bias.argtypes=[
#     ctypes.c_void_p,
#     ctypes.c_void_p,
#     ctypes.c_void_p
# ]
# tensor_lib.tensor_add_bias.restype=ctypes.c_void_p


#-----------ACTIVATIONS AND LOSSES---------------

# tensor_t* tensor_relu(tensor_pool_t*pool,tensor_t *a);
# tensor_lib.tensor_relu.argtypes=[
#     ctypes.c_void_p,
#     ctypes.c_void_p
# ]
# tensor_lib.tensor_relu.restype=ctypes.c_void_p


## tensor_t* tensor_mse_loss(tensor_pool_t *pool, tensor_t *predictions, tensor_t *target);
# tensor_lib.tensor_mse_loss.argtypes=[
#     ctypes.c_void_p,
#     ctypes.c_void_p,
#     ctypes.c_void_p
# ]
# tensor_lib.tensor_mse_loss.restype=ctypes.c_void_p


# tensor_t* tensor_cross_entropy_loss(tensor_pool_t*pool,const tensor_t *predictions,const tensor_t*targets);
# tensor_lib.tensor_cross_entropy_loss.argtypes=[
#     ctypes.c_void_p,
#     ctypes.c_void_p,
#     ctypes.c_void_p
# ]
# tensor_lib.tensor_cross_entropy_loss.restype=ctypes.c_void_p


# bool tensor_evaluate(tensor_pool_t*pool, tensor_t*t);
# tensor_lib.tensor_evaluate.argtypes=[
#     ctypes.c_void_p,
#     ctypes.c_void_p
# ]
# tensor_lib.tensor_evaluate.restype=ctypes.c_bool


# #--------UNSTABLE/ GRAPH  /  BAKARD OPEWRATIONS---------

# # bool tensor_backward(tensor_pool_t*pool,tensor_t *t);
# tensor_lib.tensor_backward.argtypes=[
#     ctypes.c_void_p,
#     ctypes.c_void_p
# ]
# tensor_lib.tensor_backward.restype=ctypes.c_bool

# # void tensor_sgd_template(tensor_pool_t*static_weights_pool, double learning_rate)
# tensor_lib.tensor_sgd_template.argtypes=[
#     ctypes.c_void_p,
#     ctypes.c_double
# ]
# tensor_lib.tensor_sgd_template.restype=None

# # tensor_graph_t* tensor_graph_create(tensor_pool_t*pool);
# tensor_lib.tensor_graph_create.argtypes=[
#     ctypes.c_void_p
# ]
# tensor_lib.tensor_graph_create.restype=ctypes.c_void_p

# #bool tensor_graph_build(tensor_graph_t*g,tensor_t*t);
# tensor_lib.tensor_graph_build.argtypes=[
#     ctypes.c_void_p,
#     ctypes.c_void_p
# ]
# tensor_lib.tensor_graph_build.restype=ctypes.c_bool


# # bool tensor_graph_forward_evaluate(tensor_graph_t*g);
# tensor_lib.tensor_graph_forward_evaluate.argtypes=[
#     ctypes.c_void_p
# ]
# tensor_lib.tensor_graph_forward_evaluate.restype=ctypes.c_bool


# # bool tensor_graph_backward_evaluate(tensor_graph_t*g);
# tensor_lib.tensor_graph_backward_evaluate.argtypes=[
#     ctypes.c_void_p
# ]
# tensor_lib.tensor_graph_backward_evaluate.restype=ctypes.c_bool

#--------GRAPH OPERATIONS (DONE)----------

# int32_t getPosOfNode(execution_node_t *et);
tensor_lib.getPosOfNode.argtypes=[ctypes.c_void_p]
tensor_lib.getPosOfNode.restype=ctypes.c_int32

# void printExecutionNode(execution_node_t *et);
tensor_lib.printExecutionNode.argtypes=[ctypes.c_void_p]
tensor_lib.printExecutionNode.restype=None
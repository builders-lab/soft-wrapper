import ctypes
import wrapper


#==========ANTI TRUNCATION SAFETY=======
wrapper.tensor_lib.tensor_pool_create.restype=ctypes.c_void_p
wrapper.tensor_lib.tensor_create.restype=ctypes.c_void_p
wrapper.tensor_lib.tensor_add.restype=ctypes.c_void_p
wrapper.tensor_lib.tensor_mul.restype=ctypes.c_void_p
wrapper.tensor_lib.tensor_mul_naive.restype=ctypes.c_void_p
wrapper.tensor_lib.tensor_transpose.restype=ctypes.c_void_p
wrapper.tensor_lib.tensor_get_data.restype=ctypes.c_void_p
wrapper.tensor_lib.tensor_get_dims.restype=ctypes.POINTER(ctypes.c_uint32)


class MemoryPool:
    """ ye class C++ ke memory ko handle karegi saftly"""
    def __init__(self,size_mb=10):
        pool_size=size_mb*1024*1024
        self.c_pool=wrapper.tensor_lib.tensor_pool_create(ctypes.c_size_t(pool_size))
        if not self.c_pool:
            raise MemoryError(" C++ memory pool is unable to create")
    
    def used_memory(self):
        return wrapper.tensor_lib.tensor_pool_used(ctypes.c_void_p(self.c_pool))
    
    def total_size(self):
        return wrapper.tensor_lib.tensor_pool_size(ctypes.c_void_p(self.c_pool))
    
    def clear(self):
        """Training loop me fast reset karne ke liye """
        wrapper.tensor_lib.tensor_pool_zero(ctypes.c_void_p(self.c_pool))
    
    def destroy(self):
        """ Memory ko completely free karne ke liye """
        wrapper.tensor_lib.tensor_pool_destroy(ctypes.c_void_p(self.c_pool))
    # left tensor alloc


class Tensor:
    """ ye class c++ ke opaque pointer ko ek python object me convert kar degi """
    def __init__(self,pool:MemoryPool, data=None , shape=None, c_pointer=None):
        self.pool = pool

        # agar c++ se direct pointer mil raha hai 
        if c_pointer is not None:
            self.c_tensor=ctypes.c_void_p(c_pointer)
        
        # agar hum python se nya matrix bhej raeeh hai 
        elif data is not None and shape is not None:
            ndims= len(shape)
            dims_array=(ctypes.c_uint32*ndims)(shape)
            
            # 2d python list ko flat 1d  c array me convert karna  hai to 
            flat_data=[val for row in data for val in row] if ndims==2 else data
            c_data=(ctypes.c_float*len(flat_data))(*flat_data)

            ptr= wrapper.tensor_lib.tensor_create(
                ctypes.c_void_p(self.pool.c_pool),
                wrapper.DTYPE_FLOAT32,
                ndims,
                dims_array,
                ctypes.cast(c_data,ctypes.c_void_p)
            )
            self.c_tensor= ctypes.c_void_p(ptr)
        else:
            raise ValueError("please give me (data+shape) or c_pointer")


    
    #========TENSOR PROPERTIES (INFO)============
    @property
    def ndim(self):
        return wrapper.tensor_lib.tensor_get_ndims(self.c_tensor)
    
    @property
    def shape(self):
        dims_ptr = wrapper.tensor_lib.tensor_get_dims(self.c_tensor)
        return [dims_ptr[i] for i in range(self.ndim)]

    @property
    def data(self):
        """ C++ ke c array  ko wapas python list me laane ke liye """
        total_elements=1
        current_shape=self.shape
        for d in current_shape:
            total_elements*=d
        
        data_ptr=wrapper.tensor_lib.tensor_get_data(self.c_tensor)
        float_array=ctypes.cast(data_ptr,ctypes.POINTER(ctypes.c_float))

        # agar 2d hai to wapas row column formate me karne ke liye 
        if self.ndim==2:
            cols=current_shape[1]
            return [flat_list[i:i+cols] for i in range(0,len(flat_list),cols)]
        return flat_list
    
    def show(self):
        """terminal me matrix print karne ke liye """
        wrapper.tensor_lib.tensor_print_data(self.c_tensor)
    


    #+===============maths operations ==========================
    def __add__(self,other):
        #A+B
        result_ptr=wrapper.tensor_lib.tensor_add(
            ctypes.c_void_p(self.pool.c_pool),
            self.c_tensor,
            other.c_tensor
        )
        return Tensor(self.pool,c_pointer=result_ptr)

    def __mul__(self,other):
        # A*B(fasst cache optimed )
        result_ptr=wrapper.tensor_lib.tensor_mul(
            ctypes.c_void_p(self.pool.c_pool),
            self.c_tensor,
            other.c_tensor
        )
        return Tensor(self.pool,c_pointer=result_ptr)
    
    def transpose(self):
        result_ptr=wrapper.tensor_lib.tensor_transpose(
            ctpyes.c_void_p(self.pool.c_pool),self.c_tensor
        )
        return Tensor(self.pool, c_pointer=result_ptr)
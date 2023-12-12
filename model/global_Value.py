# 路网的基本参数
N = 100
# 控制H_d中模拟连续函数的方差
epsilon_in_Di_c = 1
eta = 100

# 控制H_w中的参数
# 允许的状态
allowed_states = [
    [-1, -1, -1, -1],
    [-1, 1, -1, -1],
    [1, 1, 1, -1],
    [-1, 1, -1, 1],
    [1, -1, 1, -1],
    [-1, -1, -1, 1],
    [1, -1, 1, 1],
    [1, 1, 1, 1]
]

# Define the mapping from disallowed to allowed states
# The mapping should reflect the correct order as index 0 is the least significant bit
mapping = {
    (1, -1, -1, -1): (-1, -1, -1, -1),  
    (1, 1, -1, -1): (-1, 1, -1, 1),   
    (-1, 1, 1, -1): (1, 1, 1, -1),       
    (-1, -1, 1, -1): (1, -1, 1, -1),    
    (1, -1, -1, 1): (-1, 1, -1, -1),    
    (1, 1, -1, 1): (1, 1, 1, 1),   
    (-1, -1, 1, 1): (1, -1, 1, 1), 
    (-1, 1, 1, 1): (1, 1, 1, 1)        
}

# Function to check and map the input list to the allowed state
def Map_state(input_list):
    # Check if the input list is in the allowed states
    if input_list in allowed_states:
        # If it's in allowed states, return it as is with a False indicating no change
        return input_list, False
    else:
        # If it's not in allowed states, map it according to the table and return True indicating it was changed
        return list(mapping.get(tuple(input_list), input_list)), True

# 允许的状态
H_w_a = 0.001

# Bsb的初始化参数
a_0 = 1
k = 1
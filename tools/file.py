import pickle
import os

def Ensure_directory_exists(filename):
    """
    检查文件的目录是否存在，如果不存在，则创建。
    :param filename: 要检查的文件名或路径。
    """
    directory = os.path.dirname(filename)
    
    # 如果目录名不为空且目录不存在，则创建
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

def Save_data(temp_data_list, filename='temp_data_list.pkl'):
    """
    保存 temp_data_list 到一个文件中。
    :param temp_data_list: 要保存的数据列表。
    :param filename: 保存数据的文件名，默认为 'temp_data_list.pkl'。
    """
    Ensure_directory_exists(filename)
    with open(filename, 'wb') as file:
        pickle.dump(temp_data_list, file)

def Load_data(filename='temp_data_list.pkl'):
    """
    从文件中加载 temp_data_list。
    :param filename: 包含数据的文件名，默认为 'temp_data_list.pkl'。
    :return: 返回加载的数据列表。
    """
    with open(filename, 'rb') as file:
        return pickle.load(file)

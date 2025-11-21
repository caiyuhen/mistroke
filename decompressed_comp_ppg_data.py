from datetime import datetime, timedelta
import numpy as np
import time
import sys
import ctypes
import platform
import threading
import os

system_name = platform.system()

# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 候选库路径（按优先顺序尝试加载）
if system_name == "Windows":
    HASH_LIB_CANDIDATES = [
        os.path.join(current_dir, 'compression/win/hashCheck.dll'),
        os.path.join(current_dir, 'compression/win/hashCheck.so'),
    ]
    RESTORE_LIB_CANDIDATES = [
        os.path.join(current_dir, 'compression/win/restoreAndCheck.dll'),
        os.path.join(current_dir, 'compression/win/restoreAndCheck.so'),
    ]
elif system_name == "Linux":
    HASH_LIB_CANDIDATES = [os.path.join(current_dir, 'compression/linux/hashCheck.so')]
    RESTORE_LIB_CANDIDATES = [os.path.join(current_dir, 'compression/linux/restoreAndCheck.so')]
else:
    HASH_LIB_CANDIDATES = []
    RESTORE_LIB_CANDIDATES = []

thread_local = threading.local()

def _load_first_available(candidates):
    last_error = None  # pyright: ignore[reportUnusedVariable]
    for p in candidates:
        if os.path.exists(p):
            try:
                return ctypes.CDLL(p)
            except OSError as e:
                last_error = e
                continue
    return None

def get_restore_and_check_lib():
    if not hasattr(thread_local, 'restore_lib'):
        thread_local.restore_lib = _load_first_available(RESTORE_LIB_CANDIDATES)
    return thread_local.restore_lib

def get_hash_check_lib():
    if not hasattr(thread_local, 'hash_lib'):
        thread_local.hash_lib = _load_first_available(HASH_LIB_CANDIDATES)
    return thread_local.hash_lib

def release_restore_and_check_lib():
    if hasattr(thread_local, 'restore_lib'):
        del thread_local.restore_lib

def get_compression_byte_data(data):
    """
    从给定的数据列表中提取压缩字节数据。

    此函数遍历输入的数据列表，提取每个有效条目中的'collectData'字段，
    并计算这些压缩数据的总大小。

    参数:
    data (list): 包含字典的列表，每个字典应该有'collectTime'键，
                 可能包含'collectData'键。

    返回:
    list: 包含所有有效的压缩数据的列表。

    注意:
    - 函数会跳过无效的数据条目（非字典类型或缺少'collectTime'键）。
    - 函数会记录压缩数据的总大小（以字节为单位和格式化后的大小）。
    - 依赖外部的logger对象来记录信息。
    - 使用外部的format_bytes函数来格式化字节大小。
    """
    bytes_list = []  # 用于存储提取的压缩数据
    total_size = 0   # 用于累计压缩数据的总大小

    if isinstance(data, bytes):
        bytes_list.append(data)
    elif isinstance(data, list):
        for i, row in enumerate(data):
            # 检查数据的有效性
            if not isinstance(row, dict) or "collectTime" not in row:
                print(f"Error: Invalid data at index {i}")
                continue

            # 提取并存储压缩数据
            if "collectData" in row:
                compression_data = row['collectData']
                bytes_list.append(compression_data)
                total_size += sys.getsizeof(compression_data)
            
    # 记录压缩数据的总大小
    total_size_m = format_bytes(total_size)
    return bytes_list

def format_bytes(size_bytes):
    """
    将字节大小转换为人类可读的格式，带有适当的单位。

    参数:
    size_bytes (int): 需要转换的字节大小，应为非负整数。

    返回:
    str: 转换后的大小字符串，包含数值（保留两位小数）和单位。

    异常:
    ValueError: 如果输入的size_bytes为负数。
    TypeError: 如果输入的size_bytes不是整数。

    示例:
    >>> convert_size(1000)
    '0.98 KB'
    >>> convert_size(1000000)
    '0.95 MB'
    """
    # 检查输入参数的有效性
    if not isinstance(size_bytes, int):
        raise TypeError("输入必须是整数")
    if size_bytes < 0:
        raise ValueError("大小不能为负数")

    # 定义存储大小单位，从字节（B）到拍字节（PB）
    units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']

    # 初始化单位索引
    index = 0

    # 当大小大于等于1024且还有更大的单位可用时，进行转换
    while size_bytes >= 1024 and index < len(units) - 1:
        size_bytes /= 1024.0  # 除以1024进行单位转换
        index += 1  # 移动到下一个更大的单位

    # 格式化输出
    # 使用:.2f来保留两位小数
    # 使用format方法来确保即使是整数也会显示两位小数
    return "{:.2f} {}".format(size_bytes, units[index])

def check_bytes_compress_data(check_list):
    """
    检查并转换压缩数据。

    此函数接收一个压缩数据列表，并将每个压缩数据分块为4字节的块，
    然后将每块转换为一个整数。对于每个压缩数据，如果第一个整数大于15000，
    则认为数据异常并记录错误日志，同时将该数据的结果设置为None。否则，
    将转换后的整数列表添加到结果列表中。

    参数:
    check_list (list): 包含压缩数据的列表，每个元素是一个字节序列。

    返回:
    list: 包含转换后的整数列表或None的列表，表示每个压缩数据的检查结果。

    异常:
    无

    注意:
    - 函数假定输入的每个压缩数据的长度是4的倍数。
    - 如果压缩数据异常，会记录错误日志。
    """

    # 初始化结果列表
    mt_list = []

    # 遍历每个压缩数据
    for compression_data in check_list:
        # 初始化当前压缩数据的临时结果列表
        mt = []

        # 遍历当前压缩数据，按4字节一块进行处理
        chunk_count = len(compression_data) // 4
        if len(compression_data) % 4 != 0:
            print("Warning: 压缩数据长度不是4的倍数，将截断处理")
        for j in range(0, chunk_count):
            # 提取当前的4字节数据块
            byte_data = compression_data[4 * j:4 * (j + 1)]
            # 将4字节数据块转换为整数，并添加到临时结果列表中
            mt.append(int.from_bytes(byte_data, byteorder="little"))

        # 检查转换后的整数列表的第一个元素是否大于15000
        if len(mt) == 0 or mt[0] > 15000:
            # 如果异常，将结果设置为None，并记录错误日志
            mt_list.append(None)
            # logger.error("压缩数据存在异常")
        else:
            # 如果正常，将转换后的整数列表添加到结果列表中
            mt_list.append(mt)

    # 返回结果列表
    return mt_list

def check_compression_data(check_list, deviceId=None):
    """
    校验压缩数据的哈希值。

    此函数接收一个压缩数据列表，并对每个压缩数据进行哈希校验。哈希校验通过
    调用外部库函数 `hashCheck` 完成，校验结果会记录在日志中。函数返回每个压缩
    数据的校验结果，结果为1表示校验通过，0表示校验失败。

    参数:
    check_list (list): 包含压缩数据的列表，每个元素是一个字节序列。
    deviceId (str): 设备号，用于日志记录。

    返回:
    list: 包含每个压缩数据校验结果的列表，1表示通过，0表示失败。

    异常:
    无

    注意:
    - 函数假定输入的每个压缩数据是字节序列。
    - 函数依赖外部库 `hash_check_lib` 来进行哈希校验。
    - 函数执行过程中会记录校验结果和异常信息的日志。
    """
    
    # 初始化结果列表
    result_list = []

    # 遍历每个压缩数据
    lib = get_hash_check_lib()

    for compressData in check_list:
        try:
            if lib is None:
                # 库缺失时降级：返回0（校验失败），并继续
                result_list.append(0)
                continue
            # 获取压缩数据的长度
            compressDataLen = len(compressData)

            # 将压缩数据转换为 ctypes 的无符号整数数组
            compressMatrix = (ctypes.c_uint * compressDataLen)(*compressData)

            # 设置 hashCheck 函数的返回类型和参数类型
            lib.hashCheck.restype = ctypes.POINTER(ctypes.c_uint)
            lib.hashCheck.argtypes = (ctypes.POINTER(ctypes.c_uint), ctypes.c_uint, ctypes.c_int)

            # 调用 hashCheck 函数进行哈希校验
            compressHashValue = lib.hashCheck(compressMatrix, compressDataLen, 1)

            # 比较压缩数据和解密后的哈希值
            result = [compressData[1], compressData[2], compressData[3], compressData[4], compressData[5]] == [compressHashValue[0], compressHashValue[1], compressHashValue[2], compressHashValue[3], compressHashValue[4]]
        except Exception as e:
            result = 0
        # 将校验结果添加到结果列表中
        result_list.append(int(result))

    # 返回结果列表
    return result_list

def uncompression_and_check_data(check_list, deviceId=None, checkData=False, request_tag=""):
    """
    根据上传的压缩数据进行解压并获取验证结果。

    此函数接收一个压缩数据列表，对每个压缩数据进行解压，并验证解压结果。
    验证结果、解压数据的大小和一致性信息会记录在日志中。函数返回解压后的
    数据或一致性检查结果和数据大小。

    参数:
    check_list (list): 包含压缩数据的列表，每个元素是一个字节序列。
    deviceId (str): 设备号，用于日志记录。
    checkData (bool): 指示是否返回一致性检查结果和数据大小。默认为 False。

    返回:
    list: 如果 checkData 为 False，返回解压后的数据列表。如果 checkData 为 True，返回
          一致性检查结果和数据大小的列表。

    异常:
    无

    注意:
    - 函数假定输入的每个压缩数据是字节序列。
    - 函数依赖外部库 `restore_and_check_lib` 来进行数据解压和验证。
    - 函数执行过程中会记录解压结果和异常信息的日志。
    """
    # 初始化变量
    origin_data = []
    uncompress_problem_count = 0
    consistent_list = []
    data_size = []
    hz_list = []

    # 遍历每个压缩数据
    for i, compressData in enumerate(check_list):
        consistent = 0
        # 将压缩数据转换为 ctypes 的无符号整数数组
        compressMatrix = (ctypes.c_uint * len(compressData))(*compressData)
        def restore_operation():
            lib = get_restore_and_check_lib()
            if lib is None:
                raise RuntimeError("未找到恢复库，无法进行解压。请确保 compression 目录下的库文件存在。")
            lib.checkAndRestore.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_uint))
            lib.checkAndRestore.restype = ctypes.POINTER(ctypes.c_int)
            return lib.checkAndRestore(compressData[39], compressMatrix)
        restoreValue = retry_operation(restore_operation, compressData[23])
        hz_list.append(compressData[39])

        # 检查解压数据是否存在问题
        if restoreValue[compressData[23]] == 1:
            uncompress_problem_count += 1
        else:
            consistent = 1
            origin_data.append(restoreValue[:compressData[23]])
            # 记录解压数据的大小
            data_size.append(sys.getsizeof(restoreValue[:compressData[23]]))
        # 记录一致性结果
        consistent_list.append(consistent)

    # 根据 checkData 参数返回相应的结果
    if checkData:
        return consistent_list, data_size
    else:
        # 返回还原数据和采样率列表，便于后续上采样使用
        return origin_data, hz_list

def retry_operation(func, to_be_checked, max_retries=5):
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                pass

            func_result = func()
            if func_result[to_be_checked] == 1:
                pass
                if attempt < max_retries - 1:
                    continue
                else:
                    return func_result
            return func_result
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(0.1 * (2 ** attempt))  # 指数退避

def uncompress_data_upsample(downsampled_datas, target_fs, rates=None, default_original_fs=125, deviceId=None):
    """
    根据上传的下采样数据进行上采样并获取还原数据。

    此函数接收一个下采样数据列表，对每个下采样数据进行上采样，并记录还原数据。
    如果指定，函数会展示下采样和上采样的数据。上采样数据会进行四舍五入并转换为整数列表。

    参数:
    downsampled_datas (list): 包含下采样数据的列表，每个元素是一个数值序列。
    deviceId (str): 设备号，用于日志记录。

    返回:
    list: 包含每个上采样数据的列表。如果上采样失败，返回 None。

    异常:
    无

    注意:
    - 函数假定输入的每个下采样数据是数值序列。
    - 函数依赖 `linear_interpolation` 函数进行上采样。
    - 函数执行过程中会记录还原数据的大小和异常信息的日志。
    """
    # 初始化变量
    upsample_datas = []
    count_empty = 0

    # 遍历每个下采样数据
    # 若未提供采样率列表，则使用默认采样率
    if rates is None:
        rates = [default_original_fs] * len(downsampled_datas)

    for downsampled_data, original_rate in zip(downsampled_datas, rates):
        # 使用线性插值进行上采样
        upsampled_data = linear_interpolation(downsampled_data, original_rate, target_fs)

        if upsampled_data is None:
            # 如果上采样失败，记录 None 并增加异常计数
            upsample_datas.append(None)
            count_empty += 1
        else:
            # 将上采样数据四舍五入并转换为整数列表
            upsampled_data_int = np.round(upsampled_data).astype(int).tolist()
            upsample_datas.append(upsampled_data_int)

    # 返回上采样数据列表
    return upsample_datas

def linear_interpolation(original_data, original_rate, new_rate):
    """
    使用线性插值对数据进行重采样。

    此函数接收一个原始数据列表，使用线性插值方法将数据从原始采样率
    重采样到新的采样率。

    参数:
    original_data (list): 包含原始数据的列表。
    original_rate (int): 原始数据的采样率。
    new_rate (int): 新数据的采样率。

    返回:
    list: 重采样后的数据列表。如果原始数据为空或为 None，返回 None。

    异常:
    无

    注意:
    - 函数假定输入的原始数据是数值序列。
    - 函数依赖 NumPy 库进行线性插值计算。
    """
    # 检查原始数据是否为空或为 None
    if original_data is None or len(original_data) == 0:
        return None

    # 计算原始数据的长度
    original_length = len(original_data)

    # 计算新的数据长度
    new_length = int(original_length * new_rate / original_rate)

    # 生成原始和新的时间轴
    original_time = np.arange(original_length) / original_rate
    new_time = np.arange(new_length) / new_rate

    # 使用线性插值计算新的数据点
    new_data = np.interp(new_time, original_time, original_data)

    return new_data

def decompressed_and_upsampled_ppg_data(res_list):
    
    compression_data_bytes_list = get_compression_byte_data(res_list)
    compression_data_int_list = check_bytes_compress_data(compression_data_bytes_list)
    # 校验哈希（库缺失时会返回失败列表）
    check_result_list = check_compression_data(compression_data_int_list)
    # 解压并拿到采样率列表
    origin_ppg_data_list, hz_list = uncompression_and_check_data(compression_data_int_list, request_tag='')
    # 使用真实采样率进行上采样
    upsampled_data_list = uncompress_data_upsample(origin_ppg_data_list, target_fs=250, rates=hz_list)
    
    return upsampled_data_list
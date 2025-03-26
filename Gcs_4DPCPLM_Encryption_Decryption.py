import hashlib

import numpy as np
import math

N = np.float32(4.0)

a0 = np.float32(0.0)
a1 = np.float32(1.0)
a2 = np.float32(2.0)
a3 = np.float32(3.0)
a4 = np.float32(4.0)
one_hundred = np.float32(100.0)

decimal_24_sequence = (np.ones(24) * 2 ** np.array(range(24))).astype(np.uint32)

decimal_30_sequence = (np.ones(30) * 2 ** np.array(range(30))).astype(np.uint32)

# 8位二进制转十进制数组
bin2dec = np.array([128, 64, 32, 16, 8, 4, 2, 1])

shift_arr = np.array([7, 6, 5, 4, 3, 2, 1, 0])


def PLM(a_n, b_n, u, N):
    """
    混沌系统单次迭代 -> PCPLM(at, bt)
    :param a_n: 对应at
    :param b_n: 对应bt
    :param u: U
    :param N: N
    :return: 单次计算值
    """
    if a_n > 1.0:
        return a_n
    else:
        if a_n == 1.0:
            return a_n - a1 / (one_hundred * N)
        elif 0 < a_n < 1 / N:
            return N ** a2 * (a4 - u + u * a4 * b_n * (a1 - b_n)) * (a_n - a0) * (a1 / N - a_n)
        elif 1 / N < a_n < 2 / N:
            return a1 - N ** a2 * (a4 - u + u * a4 * b_n * (a1 - b_n)) * (a_n - a1 / N) * (a2 / N - a_n)
        elif 2 / N < a_n < 3 / N:
            return N ** a2 * (a4 - u + u * a4 * b_n * (a1 - b_n)) * (a_n - a2 / N) * (a3 / N - a_n)
        elif 3 / N < a_n < 1:
            return a1 - N ** a2 * (a4 - u + u * a4 * b_n * (a1 - b_n)) * (a_n - a3 / N) * (a4 / N - a_n)
        else:
            return a_n + a1 / (one_hundred * N)


def SHA_256(image_arr):
    """
    获取图像hash值
    :param image_arr: 图像数组
    :return: 图像256位二进制hash值
    """
    hex_bin = bin(int(hashlib.sha256(np.ascontiguousarray(image_arr)).hexdigest(), 16))[2:]
    if len(hex_bin) != 256:
        hex_bin = hex_bin.zfill(256)
    hex_binary = np.array(list(map(int, hex_bin)), dtype=np.uint8)
    return hex_binary


def BLAKE2b_256(image_arr):
    """
    使用blake2b获取图像hash值
    :param image_arr: 图像数组
    :return: 256位二进制hash值
    """
    hex_bin = bin(int(hashlib.blake2b(np.ascontiguousarray(image_arr)).hexdigest(), 16))[2:]
    if len(hex_bin) != 512:
        hex_bin = hex_bin.zfill(512)
    hex_binary = np.array(list(map(int, hex_bin)), dtype=np.uint8).reshape((-1, 256))
    hex_binary = hex_binary[0] ^ hex_binary[1]
    return hex_binary


def GCS_parameter_generator(binary_array):
    """
    使用256位二进制序列生成混沌系统的初始值
    :param binary_array: 256位二进制序列
    :return: 混沌系统初始值
    """
    paras1 = binary_array[:24 * 4]
    u1 = binary_array[24 * 4:24 * 4 + 30]
    paras2 = binary_array[24 * 4 + 30:24 * 8 + 30]
    u2 = binary_array[24 * 8 + 30:24 * 8 + 60]
    paras1 = np.sum(paras1.reshape((-1, 24)) * decimal_24_sequence, 1, dtype=np.float32) / np.float32(2 ** 25)
    u1 = np.sum(u1 * decimal_30_sequence, dtype=np.float32) / np.float32(2 ** 31)
    paras2 = np.sum(paras2.reshape((-1, 24)) * decimal_24_sequence, 1, dtype=np.float32) / np.float32(2 ** 25)
    u2 = np.sum(u2 * decimal_30_sequence, dtype=np.float32) / np.float32(2 ** 31)
    tuple_1 = tuple(np.around(np.append(paras1, u1), 7))
    tuple_2 = tuple(np.around(np.append(paras2, u2), 7))
    return tuple_1, tuple_2

def GCS_4DPCPLM_generator(row, col, para_args):
    """
    生成混沌序列
    :param row: 图像行大小
    :param col: 图像列大小
    :param para_args: 混沌系统的初始值
    :return: 混沌系统生成序列
    """
    (x, y, z, w, u) = para_args
    tmp = [0] * 4
    for i in range(math.ceil(60)):  # 4DPCPLM迭代50次,防止瞬态效应
        tmp[0] = PLM(x, y, u, N)
        tmp[1] = PLM(y, z, u, N)
        tmp[2] = PLM(z, w, u, N)
        tmp[3] = PLM(w, x, u, N)
        x = tmp[0]
        y = tmp[1]
        z = tmp[2]
        w = tmp[3]
    chaotic_sequence = []
    for i in range(math.ceil((row + col) / 4)):  # 4DPCPLM迭代一次生成四个参数，所以需要迭代(row + col) / 4 次
        chaotic_sequence.append(PLM(x, y, u, N))
        chaotic_sequence.append(PLM(y, z, u, N))
        chaotic_sequence.append(PLM(z, w, u, N))
        chaotic_sequence.append(PLM(w, x, u, N))
        x = chaotic_sequence[i * 4 + 0]
        y = chaotic_sequence[i * 4 + 1]
        z = chaotic_sequence[i * 4 + 2]
        w = chaotic_sequence[i * 4 + 3]
    return chaotic_sequence

def DiffusionAndPermutation(original_img, *args):
    """
    加密图像：扩散+置乱
    :param original_img: 原始图像
    :param args: 用于进行加密的序列数组
    :return: 加密图像
    """
    row, column, diffusion_array, permutation_array = args
    per_arr1, per_arr2 = permutation_array
    cipher_img = original_img ^ diffusion_array

    # 此处原来为升序排列，改为降序在np.argsort返回数组之后添加[::-1]即可
    # cipher_img = cipher_img[np.argsort(per_arr1)[::-1]]
    cipher_img = cipher_img[np.argsort(per_arr1)]
    for c in range(column):
        cipher_img[:, c] = np.roll(cipher_img[:, c], -(per_arr2[c]))

    return cipher_img.astype(np.uint8)

def GCS4DPCPLM_Image_Encryption(image=None, **kwargs):
    """
    主函数：生成混沌系统参数、生成混沌序列、加密图像；用C实现此函数功能即可
    :param image: 原始图像
    :return: 加密图像
    """
    private_key = kwargs['privateKey']
    row, column = image.shape
    # 公钥为图像hash值，私钥人为指定随机数组，异或后为密钥用于生成混沌系统初始值
    # 公钥获取（替换为blake2b）
    # public_key = SHA_256(image)
    public_key = BLAKE2b_256(image)
    secret_key = SHA_256(public_key ^ private_key)
    para_tuple_1, para_tuple_2 = GCS_parameter_generator(secret_key)
    # 填入混沌系统初值并生成混沌序列
    chaotic_sequence_1 = np.abs(GCS_4DPCPLM_generator(row, column, para_tuple_1))
    chaotic_sequence_2 = np.abs(GCS_4DPCPLM_generator(row, column, para_tuple_2))

    sub1 = (np.floor(np.array(chaotic_sequence_1) * (10 ** 5))).astype(np.uint32)
    sub2 = (np.floor(np.array(chaotic_sequence_2) * (10 ** 5))).astype(np.uint32)
    # 使用混沌序列进行加密
    diffusion_arr = (sub1[:row].reshape((-1, 1)) % 256) ^ (sub2[:column].reshape((1, -1)) % 256)
    permutation_arr = (((sub1[row:row + column] % row) + np.array(range(row))) % row,
                       ((sub2[column:row + column] % column) + np.array(range(column))) % column)
    result_img = DiffusionAndPermutation(image, row, column, diffusion_arr, permutation_arr)

    return result_img


def GCS4DPCPLM_Image_Dencryption(original_img, cipher_img, private_key):
    """
     该方法用于解密密文图像，常用于获取密钥、明文敏感性测试、裁剪攻击和噪音攻击解密后的图像
    :param original_img: 原图像（主要用于获取根据其生成的二进制hash数组）
    :param cipher_img: 密文图像
    :param private_key: 私钥
    :return: 解密后图像
    """
    cipher_ = cipher_img.copy()
    row, column = original_img.shape
    # 公钥为图像hash值，私钥人为指定随机数组，异或后为密钥用于生成混沌系统初始值
    # 同理公钥获取方式改为 blake2b
    # public_key = SHA_256(original_img)
    public_key = BLAKE2b_256(original_img)
    secret_key = SHA_256(public_key ^ private_key)
    para_tuple_1, para_tuple_2 = GCS_parameter_generator(secret_key)
    # 填入混沌系统初值并生成混沌序列
    chaotic_sequence_1 = np.abs(GCS_4DPCPLM_generator(row, column, para_tuple_1))
    chaotic_sequence_2 = np.abs(GCS_4DPCPLM_generator(row, column, para_tuple_2))
    sub1 = (np.floor(np.array(chaotic_sequence_1) * (10 ** 5))).astype(np.uint32)
    sub2 = (np.floor(np.array(chaotic_sequence_2) * (10 ** 5))).astype(np.uint32)

    diffusion_arr = (sub1[:row].reshape((-1, 1)) % 256) ^ (sub2[:column].reshape((1, -1)) % 256)
    row_permutation_arr = ((sub1[row:] % row) + np.array(range(row))) % row
    col_permutation_arr = ((sub2[column:] % column) + np.array(range(column))) % column

    # 使用混沌序列进行解密，解密为加密的逆过程，解密步骤这里不再整合到一个方法内，如下：
    for c in range(column):
        cipher_[:, c] = np.roll(cipher_[:, c], col_permutation_arr[c])
    result_img = cipher_[np.argsort(np.argsort(row_permutation_arr))]
    result_img = result_img ^ diffusion_arr

    return result_img.astype(np.uint8)
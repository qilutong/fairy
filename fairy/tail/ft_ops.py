import os
import glob


def check_sep(path):
    """判断路径字符串末尾是否有分隔符，没有则添加"""
    if path[-1] != os.path.sep:
        path = path + os.path.sep
    return path


def check_dir(directory):
    """判断目录是否存在，不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)


def get_cwd():
    """获取fairy绝对路径"""
    fairy_dir = os.path.dirname(os.path.abspath(__file__))
    return check_sep(os.path.dirname(fairy_dir))


def read_file_list(path, suffix):
    """
    从给定路径中读取符合后缀的文件名，并将路径存入列表
    :param path:
    :param suffix:要匹配的文件后缀，可传入字符串、列表或元组
    :return:
    """
    # 判断要读取的文件后缀
    suffixes = []
    if isinstance(suffix, str):
        suffixes.append(suffix)
    elif isinstance(suffix, list) or isinstance(suffix, tuple):
        suffixes = list(suffix)
    else:
        raise ValueError("suffix should be string, list or tuple")

    # 判断末尾文件是否有分隔符，没有则添加
    path = check_sep(path)

    file_list = []
    for suffix in suffixes:
        # 将目录下符合条件的文件名存入列表中
        file_list.extend(glob.glob("{}*.{}".format(path, suffix)))
    return file_list

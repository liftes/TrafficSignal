def Wrap_title(title, max_length=50):
    """
    将标题文本分割为多行，使每行的长度不超过max_length。

    :param title: 原始标题字符串。
    :param max_length: 每行的最大字符数。
    :return: 处理后的多行标题字符串。
    """
    words = title.split()
    wrapped_title = ''
    current_line = ''

    for word in words:
        if len(current_line + ' ' + word) <= max_length:
            current_line += ' ' + word
        else:
            wrapped_title += current_line + '\n'
            current_line = word

    wrapped_title += current_line
    return wrapped_title
from urllib.parse import urlparse

EMPTY_STRING = 'EMPTY_STRING'


def get_netloc(x):
    p = urlparse(x)
    val = p.netloc
    if len(val) == 0:
        val = EMPTY_STRING
    return val


def get_path(x):
    p = urlparse(x)
    val = p.path
    if len(val) == 0:
        val = EMPTY_STRING
    return val


def get_end_of_path(x):
    splits = x.split('/')
    val = ''
    if len(splits) > 0:
        val = splits[-1]
    else:
        val = x
    if len(val) == 0:
        val = EMPTY_STRING
    return val


def get_clean_script(x):
    p = urlparse(x)
    return f'{p.netloc}{p.path}'

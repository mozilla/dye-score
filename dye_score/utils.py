from urllib.parse import urlparse

EMPTY_STRING = 'EMPTY_STRING'
PARSE_ERROR = 'PARSE_ERROR'


def get_netloc(x):
    try:
        p = urlparse(x)
        val = p.netloc
    except ValueError as e:
        val = PARSE_ERROR
    if len(val) == 0:
        val = EMPTY_STRING
    return val


def get_path(x):
    try:
        p = urlparse(x)
        val = p.path
    except ValueError as e:
        val = PARSE_ERROR
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
    try:
        p = urlparse(x)
        val = f'{p.netloc}{p.path}'
    except ValueError as e:
        val = PARSE_ERROR
    return val

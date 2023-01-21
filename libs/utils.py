import os, re
import sys
import tempfile
from io import StringIO
from typing import Callable, Any
from datetime import datetime
import unicodedata
import logging

os.environ['TEMP'] = 'D:/temp'


def create_logger(*args, **kwargs):
    ch = logging.StreamHandler(sys.stdout)
    logging.basicConfig(format='[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s {%(filename)s:%(lineno)d}',
                        level=logging.INFO, handlers=[ch])
    return logging.getLogger(*args, **kwargs)


def slugify(value, allow_unicode=False):
    """
    Convert to ASCII if 'allow_unicode' is False. Convert spaces to hyphens.
    Remove characters that aren't alphanumerics, underscores, or hyphens.
    Convert to lowercase. Also strip leading and trailing whitespace.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value).strip().lower()
    return re.sub(r'[-\s]+', '-', value)


def get_stamp():
    return datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


def create_temp_dir(*args):
    tempdir = tempfile.gettempdir()
    path = os.path.join(tempdir, *args)
    os.makedirs(path, exist_ok=True)
    return path


def create_stamped_temp(*args):
    return create_temp_dir(*args, get_stamp())


def capture_output(f: Callable[[], Any]):
    sys_stdout_orig = sys.stdout
    sys.stdout = buffer = StringIO()
    f()
    sys.stdout = sys_stdout_orig
    return buffer.getvalue()

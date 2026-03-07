import os
import shutil
import tempfile
from contextlib import contextmanager

@contextmanager
def save_temp_file(upload_file):
    """
    Context manager to save an uploaded file to a temporary location 
    and ensure it's deleted after use.
    """
    suffix = os.path.splitext(upload_file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(upload_file.file, tmp)
        tmp_path = tmp.name
    
    try:
        yield tmp_path
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

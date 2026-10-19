"""Paths for the public reference pipeline; all generated files live in its output folder."""
from pathlib import Path
import os

CODE_ROOT=Path(__file__).resolve().parent
REPOSITORY=CODE_ROOT.parent
DATA_ROOT=Path(os.environ.get('PM10_OUTPUT_DIR',str(REPOSITORY/'outputs'))).resolve()

def material_source(source):
    return DATA_ROOT/Path(source).resolve().relative_to(CODE_ROOT)

def code_directory(data_directory):
    return CODE_ROOT/Path(data_directory).resolve().relative_to(DATA_ROOT)

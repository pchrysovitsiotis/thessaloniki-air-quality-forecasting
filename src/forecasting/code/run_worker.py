"""One isolated horizon. Completed model fits can be resumed without retuning."""
# The source tree is separate from input data and generated artifacts.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.dont_write_bytecode = True
_layout_sys.path.insert(0, str(next(p for p in _LayoutPath(__file__).resolve().parents if p.name == 'src')))
from _pipeline_paths import material_source, code_directory
import os, ctypes, traceback
from training.common import ROOT, write, source_inventory
from training.pipeline import run
from training.evaluation import evaluate

def normal_execution_speed():
    # Applies only to this worker; does not change the system power plan.
    if os.name != 'nt':return
    class Power(ctypes.Structure):
        _fields_=[('version',ctypes.c_ulong),('control',ctypes.c_ulong),('state',ctypes.c_ulong)]
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.GetCurrentProcess.restype=ctypes.c_void_p
    kernel.SetProcessInformation.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p,ctypes.c_ulong]
    state=Power(1,1,0)
    ok=kernel.SetProcessInformation(kernel.GetCurrentProcess(),4,ctypes.byref(state),ctypes.sizeof(state))
    write(ROOT/'runtime_power_state.json',{'automatic_speed_throttling_opt_out':bool(ok),'error':0 if ok else ctypes.get_last_error()})

if __name__=='__main__':
    try:
        normal_execution_speed()
        write(ROOT/'execution.json',{'stage':'running'})
        if not (ROOT/'source_integrity_before.json').exists():write(ROOT/'source_integrity_before.json',source_inventory())
        run();evaluate()
        write(ROOT/'execution.json',{'stage':'completed'})
    except BaseException as exc:
        write(ROOT/'execution.json',{'stage':'failed','error':str(exc),'traceback':traceback.format_exc()})
        raise

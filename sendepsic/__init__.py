# NumPy 2.x Compatibility Patch for legacy libraries (like py4DSTEM)
import sys
import builtins
import numpy as np
for alias, target_type in [
    ("float_", np.float64),
    ("int_", np.int_ or np.int64),
    ("bool_", np.bool_ or bool),
    ("complex_", np.complex128),
    ("unicode_", np.str_),
    ("string_", np.bytes_),
]:
    if alias not in np.sctypeDict:
        np.sctypeDict[alias] = target_type
    if not hasattr(np, alias):
        setattr(np, alias, target_type)

# Intercept imports to patch np.integer in py4DSTEM submodules and exspy quantification
_orig_import = builtins.__import__
def patched_import(name, globals=None, locals=None, fromlist=(), level=0):
    module = _orig_import(name, globals, locals, fromlist, level)
    
    # Patch py4DSTEM
    target_module_name = 'py4DSTEM.process.diffraction.crystal_ACOM'
    if target_module_name in sys.modules:
        mod = sys.modules[target_module_name]
        if hasattr(mod, 'np') and not hasattr(mod, '_patched_np'):
            class NumpyProxy:
                def __init__(self, orig_np):
                    self._orig_np = orig_np
                    self.integer = orig_np.int_
                    self.signedinteger = orig_np.int_
                def __getattr__(self, getattr_name):
                    return getattr(self._orig_np, getattr_name)
            mod.np = NumpyProxy(mod.np)
            mod._patched_np = True
            
    # Patch exspy eds_tem quantification
    exspy_module_name = 'exspy.signals.eds_tem'
    if exspy_module_name in sys.modules:
        exspy_mod = sys.modules[exspy_module_name]
        if hasattr(exspy_mod, 'EDSTEMSpectrum') and not hasattr(exspy_mod.EDSTEMSpectrum, '_patched_quant'):
            _orig_quantification = exspy_mod.EDSTEMSpectrum.quantification
            
            def patched_quantification(self, *args, **kwargs):
                plot_result = kwargs.get('plot_result', False)
                if plot_result:
                    kwargs['plot_result'] = False
                    res = _orig_quantification(self, *args, **kwargs)
                    
                    if isinstance(res, tuple):
                        comp_list = res[0]
                    else:
                        comp_list = res
                        
                    for sig in (comp_list if isinstance(comp_list, list) else [comp_list]):
                        try:
                            if sig.axes_manager.navigation_size == 1:
                                element = sig.metadata.Sample.elements[0]
                                try:
                                    xray_line = sig.metadata.Sample.xray_lines[0]
                                    line_str = f" ({xray_line})"
                                except Exception:
                                    line_str = ""
                                data_val = sig.data
                                if hasattr(data_val, 'item'):
                                    c = float(data_val.item())
                                elif hasattr(data_val, '__len__'):
                                    c = float(data_val[0])
                                else:
                                    c = float(data_val)
                                print(f"{element}{line_str}: Composition = {c:.2f} percent")
                        except Exception:
                            pass
                            
                    try:
                        first_sig = comp_list[0] if isinstance(comp_list, list) else comp_list
                        if first_sig.axes_manager.navigation_size != 1:
                            from hyperspy.utils.plot import plot_signals
                            plot_signals(comp_list)
                    except Exception:
                        pass
                        
                    return res
                else:
                    return _orig_quantification(self, *args, **kwargs)
            
            exspy_mod.EDSTEMSpectrum.quantification = patched_quantification
            exspy_mod.EDSTEMSpectrum._patched_quant = True
            
    return module

builtins.__import__ = patched_import

from .radial_profile import radial_profile_analysis
from .feature_extract import feature_extract
from .phase import phase_analysis
from .synthesis import comprehensive_scientific_synthesis

__all__ = [
    "radial_profile_analysis",
    "feature_extract",
    "phase_analysis",
    "comprehensive_scientific_synthesis",
]

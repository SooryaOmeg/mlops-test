import sys, os, types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class _Stub:
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _Stub()
    def __call__(self, *a, **kw):
        return _Stub()
    def __iter__(self):
        return iter([_Stub(), _Stub()])
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False

class _StreamlitModule(types.ModuleType):
    cache_resource = staticmethod(lambda f: f)
    cache_data = staticmethod(lambda f: f)
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _Stub()

sys.modules["streamlit"] = _StreamlitModule("streamlit")
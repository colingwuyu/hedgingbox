class Handler:
    def __init__(self, obj):
        self._obj = obj

    def __getattr__(self, attr: str):
        # Delegates attribute calls to the wrapped environment.
        return getattr(self._obj, attr)

    def get_obj(self):
        return self._obj

    def set_obj(self, obj):
        self._obj = obj


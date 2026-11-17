# certifi as the KiCad plugin's launcher uses it, for pyright: certifi is in
# KiCad's Python, not in this project's virtualenv.

def where() -> str: ...

# The parts of wxPython the KiCad plugin's launcher uses, for pyright: wxPython
# runs only in KiCad's Python, not in this project's virtualenv.

class App:
    def __init__(self, redirect: bool = ...) -> None: ...

class Window:
    def Destroy(self) -> bool: ...

class Sizer:
    def Add(
        self,
        item: Window | Sizer,
        proportion: int = ...,
        flag: int = ...,
        border: int = ...,
    ) -> object: ...

class BoxSizer(Sizer):
    def __init__(self, orient: int = ...) -> None: ...

class StdDialogButtonSizer(BoxSizer): ...

class Dialog(Window):
    def __init__(
        self, parent: Window | None, id: int = ..., title: str = ..., style: int = ...
    ) -> None: ...
    def CreateStdDialogButtonSizer(self, flags: int) -> StdDialogButtonSizer: ...
    def SetSizerAndFit(self, sizer: Sizer) -> None: ...
    def ShowModal(self) -> int: ...

class MessageDialog(Dialog):
    def __init__(
        self, parent: Window | None, message: str, caption: str = ..., style: int = ...
    ) -> None: ...
    def SetExtendedMessage(self, extendedMessage: str) -> None: ...
    def SetYesNoCancelLabels(self, yes: str, no: str, cancel: str) -> bool: ...

class ProgressDialog(Dialog):
    def __init__(
        self,
        title: str,
        message: str,
        maximum: int = ...,
        parent: Window | None = ...,
        style: int = ...,
    ) -> None: ...
    def Pulse(self, newmsg: str = ...) -> tuple[bool, bool]: ...

class StaticText(Window):
    def __init__(self, parent: Window, id: int = ..., label: str = ...) -> None: ...

class TextCtrl(Window):
    def __init__(
        self,
        parent: Window,
        id: int = ...,
        value: str = ...,
        size: tuple[int, int] = ...,
        style: int = ...,
    ) -> None: ...

def MessageBox(message: str, caption: str = ..., style: int = ...) -> int: ...

ALL: int
CANCEL: int
DEFAULT_DIALOG_STYLE: int
EXPAND: int
HSCROLL: int
ICON_INFORMATION: int
ICON_QUESTION: int
ID_NO: int
ID_YES: int
LEFT: int
OK: int
PD_APP_MODAL: int
PD_ELAPSED_TIME: int
RESIZE_BORDER: int
RIGHT: int
TE_MULTILINE: int
TE_READONLY: int
VERTICAL: int
YES_NO: int

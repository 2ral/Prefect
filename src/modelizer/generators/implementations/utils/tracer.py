import os
from typing import Any
from types import FrameType
from modelizer.dependencies.debuggingbook import Tracer
from pathlib import Path, PureWindowsPath


class CustomTracer(Tracer):
    def __init__(self, buffer:list, entry_str:str=None, exit_str:str=None) -> None:
        super().__init__()
        self.buffer = buffer
        self.traceit = self.trace_custom
        self.log = self.log_simple
        if entry_str != None and exit_str != None:
            self.state = False
            self.entry_str = entry_str
            self.exit_str = exit_str
            self.log = self.log_entry_exit

    def trace_custom(self, frame: FrameType, event: str, arg: Any):
        # Get the filename/path from the current frame
        filename = frame.f_code.co_filename

        # Normalize the path to make it OS-independent - convert to a Path object (which handles cross-platform logic)
        path_obj = Path(filename)

        # On Windows, ensure consistent POSIX-like format (e.g., for logs or diffs)
        if os.name == 'nt':
            filename = PureWindowsPath(path_obj).as_posix()
        else:
            filename = path_obj.as_posix()

        # strip everything before 'lib' and add 'python3.10' if needed
        parts = path_obj.parts
        try:
            lib_index = parts.index('lib')
            # Rebuild path from 'lib' onward, and inject 'python3.10' if not present
            lib_path = list(parts[lib_index:])
            if 'python3.10' not in lib_path:
                lib_path.insert(1, 'python3.10')
            filename = '/'.join(lib_path)
        except ValueError:
            # 'lib' not found in path; use normalized filename as fallback
            pass

        # add function name and line number to generated file path
        fmt_localization = filename + " " + frame.f_code.co_name + " " + str(frame.f_lineno)

        self.log(fmt_localization)

    def log_entry_exit(self, msg):
        if self.state:
            if self.exit_str in msg:
                self.state = False
            self.buffer.append(msg)
        elif self.entry_str in msg:
            self.state = True
            self.buffer.append(msg)
    
    def log_simple(self, msg):
        self.buffer.append(msg)

from __future__ import annotations

import os
import signal
import subprocess


def bind_parent_lifetime(expected_parent: int) -> None:
    """Bind this Linux worker to its actual direct Popen owner, not a tool PID."""
    import ctypes
    import sys
    if type(expected_parent) is not int or expected_parent<=1 or not sys.platform.startswith('linux'):
        raise RuntimeError('native worker requires an explicit supported parent lifetime')
    libc=ctypes.CDLL(None,use_errno=True)
    libc.prctl.argtypes=[ctypes.c_int,ctypes.c_ulong,ctypes.c_ulong,ctypes.c_ulong,ctypes.c_ulong]
    libc.prctl.restype=ctypes.c_int
    if libc.prctl(1,signal.SIGKILL,0,0,0)!=0:
        raise RuntimeError('native worker could not bind its parent lifetime')
    if os.getppid()!=expected_parent:
        raise RuntimeError('native worker owner exited before initialization')


def terminate_process_tree(process: subprocess.Popen[str], *, own_process_group: bool = False) -> None:
    if os.name == "nt":
        descendants = _windows_descendant_pids(process.pid)
        try:
            completed = subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
                check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except (OSError, subprocess.TimeoutExpired):
            completed = None
        for process_id in reversed(descendants):
            _terminate_windows_pid(process_id)
        if (completed is None or completed.returncode != 0) and process.poll() is None:
            process.kill()
        return

    process_ids = _posix_descendant_pids(process.pid) + [process.pid]
    process_groups: set[int] = set()
    if own_process_group:
        # A start_new_session child owns this process group even after its
        # leader exits and grandchildren are reparented away from its PID.
        process_groups.add(process.pid)
    for process_id in process_ids:
        try:
            process_groups.add(os.getpgid(process_id))
        except ProcessLookupError:
            continue
    for process_group in process_groups:
        if process_group == os.getpgrp():
            continue  # Never kill the caller's group for a non-isolated child.
        try:
            os.killpg(process_group, signal.SIGKILL)
        except ProcessLookupError:
            continue
    for process_id in reversed(process_ids):
        try:
            os.kill(process_id, signal.SIGKILL)
        except ProcessLookupError:
            continue


def _windows_descendant_pids(root_pid: int) -> list[int]:
    import ctypes
    from ctypes import wintypes

    class ProcessEntry32(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", ctypes.c_size_t),
            ("th32ModuleID", wintypes.DWORD),
            ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD),
            ("pcPriClassBase", wintypes.LONG),
            ("dwFlags", wintypes.DWORD),
            ("szExeFile", wintypes.WCHAR * 260),
        ]

    kernel32 = ctypes.windll.kernel32
    snapshot = kernel32.CreateToolhelp32Snapshot(0x00000002, 0)
    invalid_handle = ctypes.c_void_p(-1).value
    if snapshot == invalid_handle:
        return []
    children: dict[int, list[int]] = {}
    entry = ProcessEntry32()
    entry.dwSize = ctypes.sizeof(ProcessEntry32)
    try:
        ok = kernel32.Process32FirstW(snapshot, ctypes.byref(entry))
        while ok:
            children.setdefault(int(entry.th32ParentProcessID), []).append(
                int(entry.th32ProcessID)
            )
            ok = kernel32.Process32NextW(snapshot, ctypes.byref(entry))
    finally:
        kernel32.CloseHandle(snapshot)

    descendants: list[int] = []
    stack = list(children.get(root_pid, []))
    while stack:
        process_id = stack.pop()
        descendants.append(process_id)
        stack.extend(children.get(process_id, []))
    return descendants


def _terminate_windows_pid(process_id: int) -> None:
    import ctypes

    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(0x0001 | 0x00100000, False, process_id)
    if not handle:
        return
    try:
        kernel32.TerminateProcess(handle, 1)
        kernel32.WaitForSingleObject(handle, 2000)
    finally:
        kernel32.CloseHandle(handle)


def _posix_descendant_pids(root_pid: int) -> list[int]:
    try:
        completed = subprocess.run(
            ["ps", "-eo", "pid=,ppid="],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    children: dict[int, list[int]] = {}
    for line in completed.stdout.splitlines():
        fields = line.split()
        if len(fields) != 2:
            continue
        try:
            process_id, parent_id = (int(field) for field in fields)
        except ValueError:
            continue
        children.setdefault(parent_id, []).append(process_id)
    descendants: list[int] = []
    stack = list(children.get(root_pid, []))
    while stack:
        process_id = stack.pop()
        descendants.append(process_id)
        stack.extend(children.get(process_id, []))
    return descendants

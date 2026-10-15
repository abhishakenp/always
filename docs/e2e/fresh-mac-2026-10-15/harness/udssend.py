# udssend.py <sock> '<json>' : send one daemon command, then print events for 0.6 s
import socket, sys, time
s = socket.socket(socket.AF_UNIX); s.connect(sys.argv[1]); s.settimeout(0.1)
s.sendall((sys.argv[2] + "\n").encode())
end = time.time() + 0.6; buf = b""
while time.time() < end:
    try:
        d = s.recv(65536)
        if not d: break
        buf += d
    except socket.timeout:
        pass
for l in buf.decode(errors="replace").splitlines():
    if "Paused" in l or "Resumed" in l or "AppPaused" in l or "Error" in l: print(" ", l[:200])

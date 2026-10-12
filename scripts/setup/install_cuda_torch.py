import socket
import sys

# Force IPv4 to prevent Windows IPv6 CloudFront MTU stalling
_orig_getaddrinfo = socket.getaddrinfo
def _ipv4_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    return _orig_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)
socket.getaddrinfo = _ipv4_getaddrinfo

from pip._internal.cli.main import main

print("=" * 60)
print("INSTALLING PYTORCH WITH CUDA 12.4 (FOR GTX 1650)")
print("=" * 60)
args = [
    "install",
    "torch",
    "torchvision",
    "torchaudio",
    "--index-url",
    "https://download.pytorch.org/whl/cu124",
    "--upgrade"
]
ret = main(args)
if ret != 0:
    print(f"Pip exited with error code {ret}")
    sys.exit(ret)

print("\n" + "=" * 60)
print("INSTALLING FASTER-WHISPER & AUDIO UTILITIES")
print("=" * 60)
main(["install", "faster-whisper", "soundfile", "librosa", "opensmile", "--upgrade"])
print("\nInstallation routine complete!")

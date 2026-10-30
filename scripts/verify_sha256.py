"""Check SHA256SUMS manifest for reproducibility repository assets."""
import hashlib, sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
lines = (root / 'SHA256SUMS').read_text(encoding='utf-8').splitlines()

for line in lines:
    if not line.strip() or line.startswith('#'):
        continue
    sha, path = line.split('  ', 1)
    f = root / path
    if not f.exists():
        print('MISMATCH (missing):', path)
        sys.exit(1)
    b = f.read_bytes()
    h = hashlib.sha256(b).hexdigest()
    if h != sha:
        # Cross-platform checkout tolerance: Windows CRLF vs Linux LF for text files
        h_norm = hashlib.sha256(b.replace(b'\r\n', b'\n')).hexdigest()
        if h_norm != sha:
            print('MISMATCH:', path)
            sys.exit(1)

print('PASS:', len(lines), 'files verified')

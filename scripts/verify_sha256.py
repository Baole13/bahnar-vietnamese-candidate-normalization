"""Check SHA256SUMS manifest for reproducibility repository assets."""
import hashlib,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
lines=(root/'SHA256SUMS').read_text().splitlines()
for line in lines:
    sha,path=line.split('  ',1)
    f=root/path
    if not f.exists() or hashlib.sha256(f.read_bytes()).hexdigest()!=sha:
        print('MISMATCH:',path);sys.exit(1)
print('PASS:',len(lines),'files verified')

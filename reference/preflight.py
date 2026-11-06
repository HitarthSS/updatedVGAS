"""Verify released payload hashes and capture the isolated evaluation runtime."""
from pathlib import Path
import hashlib,json,subprocess
from huggingface_hub import snapshot_download
ROOT=Path(__file__).resolve().parent
meta=json.loads((ROOT/'assets-metadata.json').read_text());verified=[]
for f in meta['files']:
 p=ROOT/'checkpoints'/f['path']
 if not p.exists():continue
 if f.get('sha256'):
  actual=hashlib.sha256(p.read_bytes()).hexdigest()
  assert actual==f['sha256'],f['path']
 verified.append(f['path'])
assert len([f for f in verified if f.endswith('last.ckpt')])==4
base=snapshot_download('HuggingFaceTB/SmolVLM2-500M-Video-Instruct',local_files_only=True,token=False)
(ROOT/'pip-freeze.txt').write_text(subprocess.check_output([str(ROOT/'.venv/bin/python'),'-m','pip','freeze'],text=True))
(ROOT/'preflight.json').write_text(json.dumps(dict(verified_checkpoint_files=verified,base_vlm_cached_revision=Path(base).name,base_vlm_cache_path=base,tokenizer='same cached SmolVLM2 processor as policy',api_inference=False),indent=2)+'\n')
print('Verified',len(verified),'checkpoint files; cached base VLM',Path(base).name,flush=True)

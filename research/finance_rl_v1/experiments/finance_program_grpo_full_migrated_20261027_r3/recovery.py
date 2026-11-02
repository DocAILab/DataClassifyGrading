"""Preserve every interrupted attempt; never silently delete or reuse stale rows."""
import hashlib,json,time
from pathlib import Path

def require_hash(path,expected):
 if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:raise ValueError('recovery SHA identity changed')

def preserve_incomplete(folder,archive,attempt):
 folder=Path(folder)
 if not folder.exists() or (folder/'result.json').exists():return None
 archive=Path(archive);archive.mkdir(parents=True,exist_ok=True);target=archive/(folder.name+f'-before-attempt-{attempt:03d}')
 if target.exists():raise ValueError('preserved interruption destination exists')
 if not target.resolve().is_relative_to(archive.resolve()):raise ValueError('outside intended archive')
 folder.rename(target)
 return target

def next_action(*,running,exit_code,completed,recoverable,attempts,limit):
 if running:return 'WAIT'
 if completed and exit_code==0:return 'ACCEPT'
 if exit_code is not None and not recoverable:return 'DIAGNOSE'
 return 'BUDGET_STOP' if attempts>=limit else 'ADMIT'

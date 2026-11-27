"""Keep one immutable recovery point per completed optimizer update."""
import hashlib
import json
import os
from pathlib import Path
import time
import uuid


def hash_file(path):
    result = hashlib.sha256()
    with path.open("rb") as source:
        for buf in iter(lambda: source.read(1024 * 1024), b""):
            result.update(buf)
    return result.hexdigest()


def load_state(path, identity):
    import torch
    state = torch.load(path, map_location="cpu", weights_only=True)
    if tuple(state["identity"]) != tuple(identity):
        raise ValueError("Checkpoint belongs to a different model, cohort or seed")
    step = state["step"]
    opt = state["optimizer"]["state"]
    if (type(step) is not int or step < 1 or len(opt) != 80 or len(state["adapter"]) != 80
            or {int(v["step"]) for v in opt.values()} != {step}
            or not all(torch.isfinite(x).all() for x in state["adapter"].values())
            or not all(torch.isfinite(v[k]).all() for v in opt.values() for k in ("exp_avg","exp_avg_sq"))
            or not all(isinstance(state[k], torch.Tensor) and state[k].numel() for k in ("cpu_rng","cuda_rng"))):
        raise ValueError("Incomplete optimizer, weights or random state")
    return state


def atomic_json(path, data):
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".partial")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


def capture(source, identity):
    """Pin the inode before loading so an update can publish a new resume.pt concurrently."""
    folder = source.parent / "checkpoints"
    folder.mkdir(exist_ok=True)
    snapshot = folder / ("capture-" + uuid.uuid4().hex + ".partial")
    os.link(source, snapshot)
    try:
        state = load_state(snapshot, identity)
        step = state["step"]
        archive = folder / f"step-{step:03d}.pt"
        checksum = hash_file(snapshot)
        try:
            os.link(snapshot, archive)
        except FileExistsError:
            if hash_file(archive) != checksum:
                raise RuntimeError(f"Conflicting weights at update {step}")
        receipt = dict(step=step, sha256=checksum, bytes=archive.stat().st_size, identity=list(identity),
                       model_tensors=len(state["adapter"]), optimizer_slots=len(state["optimizer"]["state"]),
                       saved_at=archive.stat().st_mtime, checked_at=time.time())
        atomic_json(archive.with_suffix(".json"), receipt)
        return receipt
    finally:
        snapshot.unlink(missing_ok=True)


def save(state, latest):
    import torch
    folder = latest.parent / "checkpoints"
    folder.mkdir(exist_ok=True)
    temp = latest.parent / "resume.partial"
    with temp.open("wb") as dest:
        torch.save(state, dest)
        dest.flush()
        os.fsync(dest.fileno())
    try:
        capture(temp, state["identity"])
        os.replace(temp, latest)
    finally:
        temp.unlink(missing_ok=True)


def resume(latest, identity):
    """The archive can lead the latest pointer if publication was interrupted."""
    folder = latest.parent / "checkpoints"
    archives = sorted(folder.glob("step-*.pt"), reverse=True)
    if latest.exists():
        # A legacy writer may have published a newer step than its mirror.
        candidates = [latest, *archives]
    else:
        candidates = archives
    best = None
    errors = []
    for path in candidates:
        if path != latest and best and int(path.stem.split("-")[-1]) <= best["step"]:
            break
        try:
            receipt = path.with_suffix(".json")
            if path != latest and receipt.is_file() and hash_file(path) != json.loads(receipt.read_bytes())["sha256"]:
                raise ValueError("Checkpoint checksum mismatch")
            state = load_state(path, identity)
            if path == latest:
                archived = folder / ("step-%03d.json" % state["step"])
                if archived.is_file() and hash_file(path) != json.loads(archived.read_bytes())["sha256"]:
                    raise ValueError("Latest checkpoint differs from its archived checksum")
            if path != latest and state["step"] != int(path.stem.split("-")[-1]):
                raise ValueError("Checkpoint step number differs from its filename")
            if best is None or state["step"] > best["step"]:
                best = state
        except Exception as error:
            errors.append(f"{path.name}: {error}")
    if errors:
        print("Recovery check: " + " | ".join(errors), flush=True)
    if candidates and best is None:
        raise ValueError("No verified recovery point available")
    return best

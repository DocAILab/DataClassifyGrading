"""Find idle CUDA devices while respecting the caller's visible-device list."""
import csv
import os
import subprocess


def idle():
    def query(kind, fields):
        output = subprocess.check_output(
            ["nvidia-smi", "--query-" + kind + "=" + fields, "--format=csv,noheader,nounits"], text=True)
        return [[cell.strip() for cell in row] for row in csv.reader(output.splitlines()) if row]

    visible = os.environ.get("FINANCE_ALLOWED_GPUS", os.environ.get("CUDA_VISIBLE_DEVICES", "all"))
    allowed = {item.strip() for item in visible.split(",")} if visible != "all" else None
    occupied = {row[0] for row in query("compute-apps", "gpu_uuid")}
    return [uuid for idx,uuid,total,free,use in query("gpu","index,uuid,memory.total,memory.free,utilization.gpu")
            if (allowed is None or idx in allowed or uuid in allowed) and uuid not in occupied
            and int(total)-int(free)<1024 and int(use)<=5]

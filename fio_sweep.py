#!/usr/bin/env python3
"""
fio parameter sweep: sweeps rw, bs, iodepth, numjobs, and HIPFILE_FORCE_COMPAT_MODE,
writing IOPS, BW, latency, and CPU utilization to a CSV.
"""
import subprocess
import csv
import re
import itertools
import os
import sys
import argparse

FIO_BIN = "./fio"

# --directory is required via FIO_DIRECTORY env var; LD_LIBRARY_PATH is inherited as-is.
_fio_dir = os.environ.get("FIO_DIRECTORY")
if not _fio_dir:
    sys.exit("error: FIO_DIRECTORY env var must be set (e.g. /mnt/ais/ext4/zbyrne/fiotest)")

FIXED_ARGS = [
    "--warnings-fatal",
    "--name=hipfile",
    f"--directory={_fio_dir}",
    "--time_based",
    "--ramp_time=5",
    "--runtime=10",
    "--direct=1",
    "--size=256M",
    "--ioengine=libhipfile",
    "--rocm_io=hipfile",
    "--gpu_dev_ids=0",
    "--group_reporting",  # aggregate all jobs into one output section
]

# Sweep axes
RW_MODES        = ["read", "write"]
BLOCK_SIZES     = ["4K", "64K", "256K", "1M", "4M"]
IO_DEPTHS       = [1, 4, 8, 16]
NUM_JOBS        = [1, 8]
COMPAT_MODES    = ["true", "false"]   # HIPFILE_FORCE_COMPAT_MODE
HIPFILE_MODES   = ["stream", "sync"]

CSV_FIELDS = [
    "compat_mode", "hipfile_mode", "rw", "bs", "iodepth", "numjobs",
    "iops_avg", "bw_mib_s", "lat_avg_usec",
    "cpu_usr_pct", "cpu_sys_pct",
]


def parse_bw(val_str, unit):
    v = float(val_str)
    if unit == "KiB":
        return v / 1024
    if unit == "GiB":
        return v * 1024
    return v  # MiB


def parse_iops(val_str):
    val_str = val_str.strip()
    if val_str.endswith("k"):
        return float(val_str[:-1]) * 1000
    return float(val_str)


def parse_output(output):
    m = {}

    # iops        : min= 2142, max= 2432, avg=2377.10, ...
    hit = re.search(r'\biops\s*:\s*min=\s*[\d.k]+,\s*max=\s*[\d.k]+,\s*avg=([\d.k]+)', output)
    m["iops_avg"] = parse_iops(hit.group(1)) if hit else None

    # write: IOPS=2375, BW=2377MiB/s (2493MB/s)(...)
    hit = re.search(
        r'(?:read|write|trim):\s+IOPS=[\d.k]+,\s+BW=([\d.]+)(KiB|MiB|GiB)/s',
        output,
    )
    m["bw_mib_s"] = parse_bw(hit.group(1), hit.group(2)) if hit else None

    # Use total lat (not clat) so sync and async engines are comparable.
    # clat for sync is near-zero (just kernel accounting); lat includes slat.
    # Negative lookbehind on 'c' distinguishes " lat (" from "clat (".
    hit = re.search(r'(?<!c)lat \((nsec|usec|msec)\):.*?avg=([\d.]+)', output)
    if hit:
        unit, val = hit.group(1), float(hit.group(2))
        if unit == "nsec":
            m["lat_avg_usec"] = val / 1000
        elif unit == "msec":
            m["lat_avg_usec"] = val * 1000
        else:
            m["lat_avg_usec"] = val
    else:
        m["lat_avg_usec"] = None

    # cpu          : usr=15.27%, sys=15.84%, ...
    hit = re.search(r'cpu\s*:\s*usr=([\d.]+)%,\s*sys=([\d.]+)%', output)
    if hit:
        m["cpu_usr_pct"] = float(hit.group(1))
        m["cpu_sys_pct"] = float(hit.group(2))
    else:
        m["cpu_usr_pct"] = None
        m["cpu_sys_pct"] = None

    return m


def run_fio(rw, bs, iodepth, numjobs, compat_mode, hipfile_mode):
    cmd = [FIO_BIN] + FIXED_ARGS + [
        f"--rw={rw}",
        f"--bs={bs}",
        f"--iodepth={iodepth}",
        f"--numjobs={numjobs}",
        f"--hipfile_mode={hipfile_mode}",
    ]

    env = os.environ.copy()
    env["HIPFILE_FORCE_COMPAT_MODE"] = compat_mode

    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        print(f"\n  ERROR rc={result.returncode}", file=sys.stderr)
        if result.stderr:
            print(f"  stderr: {result.stderr[:400]}", file=sys.stderr)
        return None
    return result.stdout


def fmt(v, fmt_str):
    return format(v, fmt_str) if v is not None else "ERR"


def main():
    parser = argparse.ArgumentParser(description="Sweep fio parameters and write results to CSV.")
    parser.add_argument("--output", default="fio_sweep_results.csv", help="Output CSV path")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")
    args = parser.parse_args()

    combos = list(itertools.product(COMPAT_MODES, HIPFILE_MODES, RW_MODES, BLOCK_SIZES, IO_DEPTHS, NUM_JOBS))
    total = len(combos)
    print(f"{'DRY RUN: ' if args.dry_run else ''}Sweeping {total} combinations -> {args.output}")

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for i, (compat, hipfile_mode, rw, bs, iodepth, numjobs) in enumerate(combos, 1):
            tag = f"compat={compat:<5} mode={hipfile_mode:<7} rw={rw:<6} bs={bs:<5} iodepth={iodepth:<3} numjobs={numjobs}"
            print(f"[{i:4d}/{total}] {tag}", end="", flush=True)

            if args.dry_run:
                print()
                continue

            output = run_fio(rw, bs, iodepth, numjobs, compat, hipfile_mode)

            if output is None:
                metrics = {k: None for k in CSV_FIELDS[6:]}
                print("  FAILED")
            else:
                metrics = parse_output(output)
                iops = fmt(metrics["iops_avg"], ".0f")
                bw   = fmt(metrics["bw_mib_s"], ".1f")
                lat  = fmt(metrics["lat_avg_usec"], ".1f")
                usr  = fmt(metrics["cpu_usr_pct"], ".2f")
                sys_ = fmt(metrics["cpu_sys_pct"], ".2f")
                print(f"  IOPS={iops:>8}  BW={bw:>10} MiB/s  lat={lat:>10} us  cpu={usr}/{sys_}%")

            row = {
                "compat_mode": compat,
                "hipfile_mode": hipfile_mode,
                "rw": rw, "bs": bs, "iodepth": iodepth, "numjobs": numjobs,
                **metrics,
            }
            writer.writerow(row)
            f.flush()

    if not args.dry_run:
        print(f"\nDone. Results in {args.output}")


if __name__ == "__main__":
    main()

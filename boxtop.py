#!/usr/bin/env python3
"""Boxtop: btop-style system data for the Omarchy bar.

Usage:
  boxtop.py [COUNT]                   print one JSON sample and exit
  boxtop.py --watch INTERVAL [COUNT]  print one JSON line every INTERVAL seconds

Watch mode keeps the previous sample in memory, so CPU usage (overall,
per-core and per-process) is a delta between ticks with no sleeping inside a
sample. The shell starts it when the card opens and stops it when it closes.
"""
import ctypes
import ctypes.util
import json
import os
import pwd
import shutil
import sys
import time

CLK_TCK = os.sysconf("SC_CLK_TCK")
PAGE_KB = os.sysconf("SC_PAGE_SIZE") // 1024


def read(path):
    with open(path, "rb") as f:
        return f.read()


# ── CPU ──
def cpu_times():
    """Return {name: (idle, total)} for 'cpu' and every 'cpuN' line."""
    out = {}
    for line in read("/proc/stat").split(b"\n"):
        if not line.startswith(b"cpu"):
            break
        p = line.split()
        v = [int(x) for x in p[1:]]
        idle = v[3] + (v[4] if len(v) > 4 else 0)
        out[p[0].decode()] = (idle, sum(v))
    return out


def pct(prev, cur):
    di = cur[0] - prev[0]
    dt = cur[1] - prev[1]
    if dt <= 0:
        return 0.0
    return max(0.0, min(100.0, 100.0 * (1.0 - di / dt)))


def cpu_freq():
    try:
        base = "/sys/devices/system/cpu"
        freqs = []
        for d in os.listdir(base):
            if d.startswith("cpu") and d[3:].isdigit():
                try:
                    freqs.append(int(read(f"{base}/{d}/cpufreq/scaling_cur_freq")) / 1000)
                except OSError:
                    pass
        return round(sum(freqs) / len(freqs)) if freqs else None
    except OSError:
        return None


def find_temp_path():
    base = "/sys/class/thermal"
    try:
        zones = sorted(z for z in os.listdir(base) if z.startswith("thermal_zone"))
    except OSError:
        return None
    types = {}
    for z in zones:
        try:
            types[read(f"{base}/{z}/type").decode().strip()] = f"{base}/{z}/temp"
        except OSError:
            pass
    for name in ("x86_pkg_temp", "TCPU", "INT3400 Thermal", "SEN1", "acpitz"):
        if name in types:
            return types[name]
    return f"{base}/{zones[0]}/temp" if zones else None


def cpu_temp(path):
    if not path:
        return None
    try:
        return int(read(path)) // 1000
    except (OSError, ValueError):
        return None


def cpu_model():
    try:
        for ln in read("/proc/cpuinfo").decode(errors="replace").splitlines():
            if ln.startswith("model name"):
                name = ln.split(":", 1)[1]
                for junk in ("(R)", "(TM)", "Intel ", "AMD ", "CPU ", " Processor", "Core "):
                    name = name.replace(junk, " ")
                return " ".join(name.split("@")[0].split())[:22]
    except OSError:
        pass
    return "cpu"


# ── MEM / DISK / NET / UPTIME ──
def meminfo():
    d = {}
    for line in read("/proc/meminfo").split(b"\n"):
        k, _, v = line.partition(b":")
        if v:
            d[k] = int(v.split()[0])
    total = d[b"MemTotal"]
    avail = d.get(b"MemAvailable", d[b"MemFree"])
    cached = max(0, d.get(b"Cached", 0) + d.get(b"SReclaimable", 0) - d.get(b"Shmem", 0))
    st = d.get(b"SwapTotal", 0)
    sf = d.get(b"SwapFree", 0)
    return {"used": total - avail, "total": total, "cached": cached,
            "swap_used": st - sf, "swap_total": st}


def net_bytes():
    rx = tx = 0
    try:
        for line in read("/proc/net/dev").split(b"\n")[2:]:
            p = line.split()
            if len(p) < 10 or p[0] == b"lo:":
                continue
            rx += int(p[1])
            tx += int(p[9])
    except OSError:
        pass
    return rx, tx


def human(b):
    if b < 1024:
        return f"{b}B"
    if b < 1048576:
        return f"{b / 1024:.0f}K"
    if b < 1073741824:
        return f"{b / 1048576:.1f}M"
    return f"{b / 1073741824:.2f}G"


def uptime_human():
    s = float(read("/proc/uptime").split()[0])
    d, r = divmod(int(s), 86400)
    h, r = divmod(r, 3600)
    m, _ = divmod(r, 60)
    return f"{d}d {h}h {m}m" if d else (f"{h}h {m}m" if h else f"{m}m")


# ── GPU ──
# NVIDIA is read through NVML (libnvidia-ml, which ships with the driver and
# nvidia-smi) via ctypes, so there is no subprocess per tick. AMD and Intel
# come from /sys/class/drm and hwmon. A GPU in runtime suspend (a laptop dGPU
# that is switched off) is reported as sleeping and never queried, because
# querying it would wake it up.
DRM = "/sys/class/drm"
PCI = "/sys/bus/pci/devices"
PCI_IDS = ("/usr/share/hwdata/pci.ids", "/usr/share/misc/pci.ids")
VENDORS = {"0x10de": "nvidia", "0x1002": "amd", "0x8086": "intel"}


def read_int(path):
    try:
        return int(read(path).split()[0])
    except (OSError, ValueError, IndexError):
        return None


def read_str(path):
    try:
        return read(path).decode(errors="replace").strip()
    except OSError:
        return None


def pci_name(vendor, device):
    """'AD102 [GeForce RTX 4090]' -> 'GeForce RTX 4090' from pci.ids.

    A bracket listing several cards ('Navi 21 [Radeon RX 6800/6800 XT / 6900 XT]')
    can't say which one this is, so the chip name outside it is used instead.
    """
    v, d = vendor[2:].lower(), device[2:].lower()
    for path in PCI_IDS:
        try:
            with open(path, "rb") as f:
                in_vendor = False
                for raw in f:
                    line = raw.decode(errors="replace").rstrip("\n")
                    if not line or line[0] == "#":
                        continue
                    if line[0] != "\t":
                        if in_vendor:
                            break
                        in_vendor = line.startswith(v + "  ")
                    elif in_vendor and line[1] != "\t" and line[1:].startswith(d + "  "):
                        name = line[1 + len(d) + 2:].strip()
                        if "[" in name and name.endswith("]"):
                            i = name.rindex("[")
                            inner = name[i + 1:-1]
                            name = name[:i].strip() if "/" in inner and name[:i].strip() else inner
                        return name
        except OSError:
            continue
    return None


def short_gpu_name(name, vendor):
    name = name or vendor.upper()
    for junk in ("NVIDIA ", "GeForce ", "AMD ", "Intel(R) ", "Intel ", "(R)", "(TM)", " Graphics"):
        name = name.replace(junk, " ")
    name = " ".join(name.split())
    return name[:22] or vendor


class Nvml:
    """Just enough NVML for the gpu box. Every call returns None on error."""

    class Memory(ctypes.Structure):
        _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong),
                    ("used", ctypes.c_ulonglong)]

    class MemoryV2(ctypes.Structure):
        _fields_ = [("version", ctypes.c_uint), ("total", ctypes.c_ulonglong),
                    ("reserved", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong),
                    ("used", ctypes.c_ulonglong)]

    class Utilization(ctypes.Structure):
        _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]

    def __init__(self):
        lib = ctypes.CDLL(ctypes.util.find_library("nvidia-ml") or "libnvidia-ml.so.1")
        init = getattr(lib, "nvmlInit_v2", None) or lib.nvmlInit
        if init() != 0:
            raise OSError("nvmlInit failed")
        self.lib = lib
        self.has_mem_v2 = hasattr(lib, "nvmlDeviceGetMemoryInfo_v2")

    def handles(self):
        count = ctypes.c_uint(0)
        fn = getattr(self.lib, "nvmlDeviceGetCount_v2", None) or self.lib.nvmlDeviceGetCount
        if fn(ctypes.byref(count)) != 0:
            return []
        out = []
        get = getattr(self.lib, "nvmlDeviceGetHandleByIndex_v2", None) or self.lib.nvmlDeviceGetHandleByIndex
        for i in range(count.value):
            h = ctypes.c_void_p()
            if get(i, ctypes.byref(h)) == 0:
                out.append(h)
        return out

    def handle_by_bus(self, bus):
        h = ctypes.c_void_p()
        fn = getattr(self.lib, "nvmlDeviceGetHandleByPciBusId_v2", None) or self.lib.nvmlDeviceGetHandleByPciBusId
        return h if fn(bus.encode(), ctypes.byref(h)) == 0 else None

    def _uint(self, fn, h, *args):
        v = ctypes.c_uint(0)
        return v.value if getattr(self.lib, fn)(h, *args, ctypes.byref(v)) == 0 else None

    def name(self, h):
        buf = ctypes.create_string_buffer(96)
        if self.lib.nvmlDeviceGetName(h, buf, ctypes.c_uint(96)) != 0:
            return None
        return buf.value.decode(errors="replace")

    def sample(self, h):
        util = self.Utilization()
        busy = util.gpu if self.lib.nvmlDeviceGetUtilizationRates(h, ctypes.byref(util)) == 0 else None
        used = total = None
        if self.has_mem_v2:
            # v2 reports "used" without the driver-reserved part, like nvidia-smi.
            m2 = self.MemoryV2(version=ctypes.sizeof(self.MemoryV2) | (2 << 24))
            if self.lib.nvmlDeviceGetMemoryInfo_v2(h, ctypes.byref(m2)) == 0:
                used, total = m2.used, m2.total
        if total is None:
            m = self.Memory()
            if self.lib.nvmlDeviceGetMemoryInfo(h, ctypes.byref(m)) == 0:
                used, total = m.used, m.total
        power = self._uint("nvmlDeviceGetPowerUsage", h)             # mW
        limit = self._uint("nvmlDeviceGetEnforcedPowerLimit", h)     # mW
        return {
            "util": busy,
            "mem_used": used, "mem_total": total,
            "temp": self._uint("nvmlDeviceGetTemperature", h, ctypes.c_int(0)),   # NVML_TEMPERATURE_GPU
            "power": power / 1000 if power is not None else None,
            "power_limit": limit / 1000 if limit else None,
            "clock": self._uint("nvmlDeviceGetClockInfo", h, ctypes.c_int(0)),   # NVML_CLOCK_GRAPHICS
            "fan": self._uint("nvmlDeviceGetFanSpeed", h),
        }


def hwmon_dir(dev):
    try:
        for d in sorted(os.listdir(f"{dev}/hwmon")):
            return f"{dev}/hwmon/{d}"
    except OSError:
        pass
    return None


def hwmon_temp(hw):
    """GPU temperature in °C: AMD 'edge', Intel xe 'pkg', else the first sensor."""
    if not hw:
        return None
    labels = {}
    for i in range(1, 8):
        if read_int(f"{hw}/temp{i}_input") is not None:
            labels[read_str(f"{hw}/temp{i}_label") or ""] = i
    if not labels:
        return None
    pick = next((labels[k] for k in ("edge", "pkg", "junction") if k in labels), min(labels.values()))
    v = read_int(f"{hw}/temp{pick}_input")
    return v // 1000 if v is not None else None


# One NVML session shared by every NVIDIA GPU. A failed open (no driver
# library, nouveau) is retried at most every 30 s, not on every tick.
_nvml = None
_nvml_retry = 0.0


def open_nvml():
    global _nvml, _nvml_retry
    if _nvml is None and time.monotonic() >= _nvml_retry:
        try:
            _nvml = Nvml()
        except (OSError, AttributeError):
            _nvml_retry = time.monotonic() + 30
    return _nvml


def nvidia_devices():
    """PCI dirs of display-class NVIDIA devices bound to the nvidia driver.

    Read from /sys/bus/pci rather than /sys/class/drm, so a laptop dGPU
    without nvidia_drm (no cardN) still gets its runtime-PM state checked.
    Returns None when PCI sysfs is unavailable.
    """
    try:
        names = sorted(os.listdir(PCI))
    except OSError:
        return None
    out = []
    for n in names:
        dev = f"{PCI}/{n}"
        if read_str(f"{dev}/vendor") != "0x10de" or not (read_str(f"{dev}/class") or "").startswith("0x03"):
            continue
        if os.path.basename(os.path.realpath(f"{dev}/driver")) == "nvidia":
            out.append(dev)
    return out


def nvidia_all_awake(devs):
    # nvmlInit attaches to every NVIDIA GPU, so it would wake any of them that
    # is runtime-suspended: only open NVML when none is.
    return all(read_str(f"{d}/power/runtime_status") != "suspended" for d in devs)


class Gpu:
    """One GPU. sample() returns a dict of plain numbers, or a sleeping stub."""

    def __init__(self, vendor, name, card=None, dev=None, nvml=None, handle=None, peers=()):
        self.vendor = vendor
        self.peers = peers      # every NVIDIA PCI dir, for nvidia_all_awake()
        self.name = short_gpu_name(name, vendor)
        self.card = card        # /sys/class/drm/cardN, if the GPU has one
        self.dev = dev          # its PCI device directory
        self.nvml = nvml
        self.handle = handle
        self.hwmon = hwmon_dir(dev) if dev else None
        self.prev_energy = None  # (µJ, monotonic seconds) for energy-only hwmon

    def sleeping(self):
        return bool(self.dev) and read_str(f"{self.dev}/power/runtime_status") == "suspended"

    def sample(self):
        base = {"id": os.path.basename(self.dev) if self.dev else self.name,
                "name": self.name, "vendor": self.vendor}
        if self.sleeping():
            base["sleeping"] = True
            return base
        if self.vendor == "nvidia":
            if self.handle is None and not self.connect():
                return dict(base, **self.finish({}))
            d = self.nvml.sample(self.handle)
        elif self.vendor == "amd":
            d = self.sample_amd()
        else:
            d = self.sample_intel()
        if d.get("power") is None:
            d["power"] = self.energy_power()
        return dict(base, **self.finish(d))

    def connect(self):
        """Open NVML for a GPU that was asleep when the sampler started."""
        if not nvidia_all_awake(self.peers):
            return False
        self.nvml = self.nvml or open_nvml()
        if not self.nvml:
            return False
        self.handle = self.nvml.handle_by_bus(os.path.basename(self.dev)) if self.dev else None
        return self.handle is not None

    def sample_amd(self):
        dev, hw = self.dev, self.hwmon
        power = None
        if hw:
            uw = read_int(f"{hw}/power1_average")
            if uw is None:
                uw = read_int(f"{hw}/power1_input")
            power = uw / 1e6 if uw is not None else None
        cap = read_int(f"{hw}/power1_cap") if hw else None
        sclk = read_int(f"{hw}/freq1_input") if hw else None
        fan = None
        if hw:
            pwm = read_int(f"{hw}/pwm1")
            pmax = read_int(f"{hw}/pwm1_max") or 255
            fan = round(100 * pwm / pmax) if pwm is not None and pmax else None
        return {
            "util": read_int(f"{dev}/gpu_busy_percent"),
            "mem_used": read_int(f"{dev}/mem_info_vram_used"),
            "mem_total": read_int(f"{dev}/mem_info_vram_total"),
            "temp": hwmon_temp(hw),
            "power": power, "power_limit": cap / 1e6 if cap else None,
            "clock": round(sclk / 1e6) if sclk else None,
            "fan": fan,
        }

    def sample_intel(self):
        # i915 and xe expose no busy counter in sysfs (that needs perf), so
        # Intel gets clock, temperature and power only.
        clock = read_int(f"{self.card}/gt_act_freq_mhz") if self.card else None
        if clock is None and self.dev:
            clock = read_int(f"{self.dev}/tile0/gt0/freq0/act_freq")
        return {"util": None, "mem_used": None, "mem_total": None,
                "temp": hwmon_temp(self.hwmon), "power": None, "power_limit": None,
                "clock": clock, "fan": None}

    def energy_power(self):
        """Watts from a cumulative energy counter (Intel dGPUs, some APUs)."""
        uj = read_int(f"{self.hwmon}/energy1_input") if self.hwmon else None
        if uj is None:
            return None
        now = time.monotonic()
        prev, self.prev_energy = self.prev_energy, (uj, now)
        if not prev or now <= prev[1] or uj < prev[0]:
            return None
        return (uj - prev[0]) / 1e6 / (now - prev[1])

    @staticmethod
    def finish(d):
        used, total = d.get("mem_used"), d.get("mem_total")
        power, limit = d.get("power"), d.get("power_limit")
        b2g = lambda b: round(b / 2**30, 1)
        return {
            "util": d.get("util"),
            "mem_used": b2g(used) if used is not None and total else None,
            "mem_total": b2g(total) if total else None,
            "mem_pct": round(100 * used / total) if used is not None and total else None,
            "temp": d.get("temp"),
            "power": round(power) if power is not None else None,
            "power_limit": round(limit) if limit else None,
            "power_pct": min(100, round(100 * power / limit)) if power is not None and limit else None,
            "clock": d.get("clock"),
            "fan": d.get("fan"),
        }


def find_gpus():
    """AMD / Intel GPUs from their DRM cards, NVIDIA ones from PCI + NVML."""
    gpus = []
    try:
        cards = sorted((c for c in os.listdir(DRM) if c.startswith("card") and c[4:].isdigit()),
                       key=lambda c: int(c[4:]))
    except OSError:
        cards = []
    for c in cards:
        card = f"{DRM}/{c}"
        dev = os.path.realpath(f"{card}/device")
        vendor = VENDORS.get(read_str(f"{dev}/vendor") or "")
        if not vendor or vendor == "nvidia":
            continue
        asleep = read_str(f"{dev}/power/runtime_status") == "suspended"
        name = (None if asleep else read_str(f"{dev}/product_name")) or pci_name(read_str(f"{dev}/vendor"),
                                                           read_str(f"{dev}/device") or "0x0000")
        gpus.append(Gpu(vendor, name, card=card, dev=dev))

    devs = nvidia_devices()
    if devs is None:
        # No PCI sysfs (a container?): nothing to check power state with, so
        # just list what NVML sees.
        nvml = open_nvml()
        for h in (nvml.handles() if nvml else []):
            gpus.append(Gpu("nvidia", nvml.name(h), nvml=nvml, handle=h))
        return gpus
    # NVML is only opened when no NVIDIA GPU is asleep, so a sleeping laptop
    # dGPU stays asleep; a GPU that wakes later connects in Gpu.connect().
    nvml = open_nvml() if devs and nvidia_all_awake(devs) else None
    for dev in devs:
        handle = nvml.handle_by_bus(os.path.basename(dev)) if nvml else None
        name = (nvml.name(handle) if handle is not None else None) or \
            pci_name("0x10de", read_str(f"{dev}/device") or "0x0000")
        gpus.append(Gpu("nvidia", name, dev=dev, nvml=nvml, handle=handle, peers=devs))
    return gpus


def gpu_sort_key(g):
    """Awake, busy-capable (not Intel), big-VRAM GPUs first: the box shows gpus[0] in full."""
    return (g.get("sleeping", False), g["vendor"] == "intel", -(g.get("mem_total") or 0))


# ── PROCS ──
_users = {}


def username(pid):
    try:
        uid = os.stat(f"/proc/{pid}").st_uid
    except OSError:
        return "?"
    name = _users.get(uid)
    if name is None:
        try:
            name = pwd.getpwuid(uid).pw_name
        except KeyError:
            name = str(uid)
        _users[uid] = name
    return name


def scan_procs():
    """Return {pid: (name, cpu_ticks, rss_kb)} reading only /proc/PID/stat."""
    out = {}
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            st = read(f"/proc/{name}/stat")
        except OSError:
            continue
        lp = st.find(b"(")
        rp = st.rfind(b")")
        f = st[rp + 2:].split()
        try:
            out[int(name)] = (st[lp + 1:rp], int(f[11]) + int(f[12]), int(f[21]) * PAGE_KB)
        except (IndexError, ValueError):
            continue
    return out


class Sampler:
    def __init__(self, count):
        self.count = count
        self.model = cpu_model()
        self.temp_path = find_temp_path()
        try:
            self.gpus = find_gpus()
        except Exception:  # GPU stats are a bonus; never break the card for them
            self.gpus = []
        self.prev_cpu = cpu_times()
        self.prev_procs = scan_procs()
        self.prev_time = time.monotonic()

    def sample(self):
        now = time.monotonic()
        dt = max(0.001, now - self.prev_time)

        cur_cpu = cpu_times()
        overall = pct(self.prev_cpu.get("cpu", (0, 0)), cur_cpu.get("cpu", (0, 0)))
        core_names = sorted((k for k in cur_cpu if k != "cpu"), key=lambda k: int(k[3:]))
        cores = [round(pct(self.prev_cpu.get(k, cur_cpu[k]), cur_cpu[k])) for k in core_names]

        procs = scan_procs()
        rows = []
        for pid, (comm, ticks, rss) in procs.items():
            old = self.prev_procs.get(pid)
            cpu_pct = 100.0 * max(0, ticks - old[1]) / CLK_TCK / dt if old else 0.0
            rows.append((cpu_pct, rss, pid, comm))
        active = sorted((r for r in rows if r[0] > 0.05), key=lambda r: r[0], reverse=True)
        idle = sorted((r for r in rows if r[0] <= 0.05), key=lambda r: r[1], reverse=True)
        top = (active + idle)[:self.count]

        mem = meminfo()
        total = mem["total"]
        disk = shutil.disk_usage("/")
        load = read("/proc/loadavg").split()[:3]
        rx, tx = net_bytes()
        gpus = []
        for g in self.gpus:
            try:
                gpus.append(g.sample())
            except Exception:
                continue
        gpus.sort(key=gpu_sort_key)

        self.prev_cpu = cur_cpu
        self.prev_procs = procs
        self.prev_time = now

        kb2g = lambda kb: round(kb / 1048576, 1)
        return {
            "cpu": round(overall), "cores": cores, "freq": cpu_freq(),
            "temp": cpu_temp(self.temp_path), "cpu_model": self.model,
            "ram_used": kb2g(mem["used"]), "ram_total": kb2g(total),
            "ram_pct": round(100 * mem["used"] / total) if total else 0,
            "cached": kb2g(mem["cached"]),
            "swap_used": kb2g(mem["swap_used"]), "swap_total": kb2g(mem["swap_total"]),
            "swap_pct": round(100 * mem["swap_used"] / mem["swap_total"]) if mem["swap_total"] else 0,
            "disk_used": round(disk.used / 2**30), "disk_total": round(disk.total / 2**30),
            "disk_pct": round(100 * disk.used / disk.total) if disk.total else 0,
            "load": load[0].decode(), "load5": load[1].decode(), "load15": load[2].decode(),
            "uptime": uptime_human(),
            "net_rx": human(rx), "net_tx": human(tx),
            "net_rx_bytes": rx, "net_tx_bytes": tx,
            "gpus": gpus,
            "procs": [{
                "pid": pid,
                "name": comm.decode(errors="replace"),
                "user": username(pid),
                "mem_mb": round(rss / 1024),
                "mem_pct": round(100 * rss / total, 1) if total else 0,
                "cpu_pct": round(cpu_pct, 1),
            } for cpu_pct, rss, pid, comm in top],
        }


def clamp_int(v, lo, hi, default):
    try:
        return max(lo, min(hi, int(float(v))))
    except (TypeError, ValueError):
        return default


def main(argv):
    if argv and argv[0] == "--watch":
        interval = max(0.5, min(10.0, float(argv[1]) if len(argv) > 1 else 2.0))
        count = clamp_int(argv[2] if len(argv) > 2 else 8, 1, 20, 8)
        parent = os.getppid()
        sampler = Sampler(count)
        time.sleep(0.25)  # short warm-up so the first line already has real deltas
        nxt = time.monotonic()
        while True:
            # Exit if the shell went away and we were re-parented.
            if os.getppid() != parent:
                return
            try:
                sys.stdout.write(json.dumps(sampler.sample(), separators=(",", ":")) + "\n")
                sys.stdout.flush()
            except BrokenPipeError:
                return
            nxt += interval
            time.sleep(max(0.05, nxt - time.monotonic()))
    else:
        count = clamp_int(argv[0] if argv else 8, 1, 20, 8)
        sampler = Sampler(count)
        time.sleep(0.25)
        print(json.dumps(sampler.sample()))


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except KeyboardInterrupt:
        pass

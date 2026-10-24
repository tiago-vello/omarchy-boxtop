#!/usr/bin/env python3
"""Light tests for the Python side of Boxtop (no Qt, no network).

Run:  python3 -m unittest discover -s tests -v
"""
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import boxtop  # noqa: E402
import boxtopctl  # noqa: E402
import updater  # noqa: E402


class SamplerTest(unittest.TestCase):
    def test_compiles(self):
        for name in ("boxtop.py", "boxtopctl.py", "updater.py"):
            with open(os.path.join(ROOT, name), encoding="utf-8") as f:
                compile(f.read(), name, "exec")

    def test_sample_shape_and_cost(self):
        s = boxtop.Sampler(8)
        time.sleep(0.2)
        t0 = time.process_time()
        for _ in range(5):
            d = s.sample()
        per_tick_ms = (time.process_time() - t0) / 5 * 1000
        for key in ("cpu", "cores", "ram_pct", "procs", "net_rx_bytes", "uptime"):
            self.assertIn(key, d)
        self.assertLessEqual(len(d["procs"]), 8)
        print("\n  sampler CPU per tick: %.1f ms" % per_tick_ms)
        self.assertLess(per_tick_ms, 60)  # generous bound for slow CI boxes


class GpuTest(unittest.TestCase):
    """Fake /sys/class/drm trees: AMD readings, sleeping NVIDIA, pci.ids names."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.drm = os.path.join(self.tmp, "drm")
        os.makedirs(self.drm)
        ids = os.path.join(self.tmp, "pci.ids")
        with open(ids, "w") as f:
            f.write("# comment\n1002  Advanced Micro Devices, Inc. [AMD/ATI]\n"
                    "\t73bf  Navi 21 [Radeon RX 6800/6800 XT / 6900 XT]\n"
                    "10de  NVIDIA Corporation\n\t2684  AD102 [GeForce RTX 4090]\n")
        self.pci = os.path.join(self.tmp, "pci")
        os.makedirs(os.path.join(self.tmp, "drivers", "nvidia"))
        self.patches = [mock.patch.object(boxtop, "DRM", self.drm),
                        mock.patch.object(boxtop, "PCI", self.pci),
                        mock.patch.object(boxtop, "PCI_IDS", (ids,)),
                        mock.patch.object(boxtop, "_nvml", None),
                        mock.patch.object(boxtop, "_nvml_retry", 0.0)]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def card(self, n, vendor, device, files, status="active", drm=True):
        dev = os.path.join(self.pci, "0000:0%d:00.0" % n)
        files = dict(files, vendor=vendor, device=device, **{"class": "0x030000"})
        files["power/runtime_status"] = status
        for rel, value in files.items():
            path = os.path.join(dev, rel)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write(str(value) + "\n")
        if vendor == "0x10de":
            os.symlink(os.path.join(self.tmp, "drivers", "nvidia"), os.path.join(dev, "driver"))
        if not drm:
            return
        os.makedirs(os.path.join(self.drm, "card%d" % n))
        os.symlink(dev, os.path.join(self.drm, "card%d" % n, "device"))
        os.makedirs(os.path.join(self.drm, "card%d-DP-1" % n))  # connectors are skipped

    def test_amd_readings(self):
        self.card(0, "0x1002", "0x73bf", {
            "gpu_busy_percent": 37, "mem_info_vram_used": 2 * 2**30, "mem_info_vram_total": 16 * 2**30,
            "hwmon/hwmon3/temp1_label": "edge", "hwmon/hwmon3/temp1_input": 51000,
            "hwmon/hwmon3/power1_average": 120500000, "hwmon/hwmon3/power1_cap": 250000000,
            "hwmon/hwmon3/freq1_input": 2100000000, "hwmon/hwmon3/pwm1": 102,
        })
        g = boxtop.find_gpus()[0].sample()
        self.assertEqual(g["name"], "Navi 21")  # pci.ids can't tell a 6800 from a 6900 XT
        self.assertEqual((g["util"], g["mem_used"], g["mem_total"], g["mem_pct"]), (37, 2.0, 16.0, 12))
        self.assertEqual((g["temp"], g["power"], g["power_limit"], g["power_pct"]), (51, 120, 250, 48))
        self.assertEqual((g["clock"], g["fan"]), (2100, 40))

    def test_sleeping_nvidia_is_never_queried(self):
        # No DRM node (nvidia_drm not loaded) must still be caught via PCI.
        for drm in (True, False):
            shutil.rmtree(self.pci, ignore_errors=True)
            shutil.rmtree(self.drm, ignore_errors=True)
            os.makedirs(self.drm)
            self.card(1, "0x10de", "0x2684", {}, status="suspended", drm=drm)
            with mock.patch.object(boxtop, "Nvml", side_effect=AssertionError("NVML opened")):
                gpus = boxtop.find_gpus()
                self.assertEqual(gpus[0].sample(), {"id": "0000:01:00.0", "name": "RTX 4090",
                                                    "vendor": "nvidia", "sleeping": True})

    def test_xe_pkg_temperature(self):
        self.card(2, "0x8086", "0xe20b", {"hwmon/hwmon5/temp2_label": "pkg", "hwmon/hwmon5/temp2_input": 47000,
                                          "hwmon/hwmon5/temp3_label": "vram", "hwmon/hwmon5/temp3_input": 60000})
        self.assertEqual(boxtop.find_gpus()[0].sample()["temp"], 47)

    def test_failed_nvml_open_backs_off(self):
        with mock.patch.object(boxtop, "Nvml", side_effect=OSError("no lib")) as m:
            boxtop.open_nvml()
            boxtop.open_nvml()
            self.assertEqual(m.call_count, 1)

    def test_awake_gpus_sort_first(self):
        rows = [{"vendor": "nvidia", "sleeping": True}, {"vendor": "intel", "mem_total": None},
                {"vendor": "amd", "mem_total": 16.0}]
        self.assertEqual([r["vendor"] for r in sorted(rows, key=boxtop.gpu_sort_key)], ["amd", "intel", "nvidia"])

    def test_no_gpu_is_empty(self):
        with mock.patch.object(boxtop, "Nvml", side_effect=OSError("no libnvidia-ml")):
            self.assertEqual(boxtop.find_gpus(), [])


class CtlTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.patches = [
            mock.patch.object(boxtopctl, "PRESET_DIR", os.path.join(self.tmp, "presets")),
            mock.patch.object(boxtopctl, "SETTINGS_FILE", os.path.join(self.tmp, "settings.json")),
            mock.patch.object(boxtopctl, "HYPR_TOGGLE", os.path.join(self.tmp, "toggles/boxtop-glass.lua")),
            mock.patch.object(boxtopctl, "THEME_COLORS", os.path.join(self.tmp, "colors.toml")),
            mock.patch.object(boxtopctl, "WALLPAPER", os.path.join(self.tmp, "nowallpaper")),
            mock.patch.object(boxtopctl, "CACHE_DIR", os.path.join(self.tmp, "cache")),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_cmd(self, *argv):
        out = io.StringIO()
        with mock.patch("sys.stdout", out):
            rc = boxtopctl.main(list(argv))
        return rc, out.getvalue().strip()

    def test_presets_roundtrip_strip_local_keys(self):
        rc, _ = self.run_cmd("preset-save", "My look", "--json",
                             json.dumps({"id": "x", "glassStyle": "dark", "hero": "/a.png", "radius": 20}))
        self.assertEqual(rc, 0)
        rc, out = self.run_cmd("preset-list")
        self.assertEqual(json.loads(out), ["My look"])
        rc, out = self.run_cmd("preset-load", "My look")
        self.assertEqual(json.loads(out), {"glassStyle": "dark", "radius": 20})

    def test_bad_preset_name_rejected(self):
        with mock.patch("sys.stderr", io.StringIO()):
            rc, _ = self.run_cmd("preset-load", "../../etc/passwd")
        self.assertEqual(rc, 1)

    def test_mirror_restore(self):
        self.run_cmd("mirror", "--json", json.dumps({"id": "x", "accent": "#112233"}))
        rc, out = self.run_cmd("restore")
        self.assertEqual(json.loads(out), {"accent": "#112233"})

    def test_hypr_glass_only_reloads_on_change(self):
        with mock.patch("shutil.which", return_value=None):
            _, a = self.run_cmd("hypr-glass", "on", "70")
            _, b = self.run_cmd("hypr-glass", "on", "70")
            _, c = self.run_cmd("hypr-glass", "off")
        self.assertTrue(json.loads(a)["changed"])
        self.assertFalse(json.loads(b)["changed"])
        self.assertTrue(json.loads(c)["changed"])
        self.assertFalse(os.path.exists(boxtopctl.HYPR_TOGGLE))
        self.assertNotIn("gradient", boxtopctl.glass_lua(50))

    def test_palette_falls_back_to_theme(self):
        with open(boxtopctl.THEME_COLORS, "w") as f:
            f.write('accent = "#89b4fa"\nblue = "#89b4fa"\nmagenta = "#f5c2e7"\ngreen = "#a6e3a1"\n')
        rc, out = self.run_cmd("palette")
        d = json.loads(out)
        self.assertEqual(d["source"], "theme")
        self.assertEqual(d["colors"][:3], ["#89b4fa", "#f5c2e7", "#a6e3a1"])

    def test_swatches_distinct(self):
        hist = [(500, (30, 30, 46)), (300, (137, 180, 250)), (290, (140, 182, 248)), (200, (245, 194, 231)),
                (100, (166, 227, 161))]
        sw = boxtopctl.pick_swatches(hist, 5)
        self.assertGreaterEqual(len(sw), 3)
        self.assertEqual(len(sw), len(set(sw)))


def make_plugin(dirpath, version):
    os.makedirs(dirpath, exist_ok=True)
    with open(os.path.join(dirpath, "manifest.json"), "w") as f:
        json.dump({"id": "maftuuh.boxtop", "version": version}, f)
    with open(os.path.join(dirpath, "VERSION"), "w") as f:
        f.write(version + "\n")
    for name in ("Panel.qml", "boxtop.py"):
        with open(os.path.join(dirpath, name), "w") as f:
            f.write("# " + version + "\n")


class UpdaterTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.inst = os.path.join(self.tmp, "maftuuh.boxtop")
        make_plugin(self.inst, "1.5.0")
        self.patches = [
            mock.patch.object(updater, "HERE", self.inst),
            mock.patch.object(updater, "CACHE_DIR", os.path.join(self.tmp, "cache")),
            mock.patch.object(updater, "CACHE_FILE", os.path.join(self.tmp, "cache/update.json")),
            mock.patch.object(updater, "BACKUP_DIR", os.path.join(self.tmp, "cache/backups")),
            mock.patch.object(updater, "notify", lambda *a: None),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_version_key(self):
        self.assertGreater(updater.version_key("v1.10.0"), updater.version_key("1.9.9"))
        self.assertEqual(updater.clean_version("v2"), "2.0.0")

    def test_check_release_newer_and_cached(self):
        calls = []

        def fake_rel():
            calls.append(1)
            return ("v1.7.0", "## v1.7.0\n- **glass** fix\n", "https://example/t.tar.gz")
        with mock.patch.object(updater, "latest_release", fake_rel):
            a = updater.do_check()
            b = updater.do_check()
        self.assertTrue(a["available"])
        self.assertEqual(a["latest"], "1.7.0")
        self.assertIn("glass fix", a["notes"])
        self.assertEqual(len(calls), 1, "second check must come from the cache")
        self.assertEqual(b["latest"], "1.7.0")

    def test_check_offline_is_quiet(self):
        def boom():
            raise OSError("offline")
        with mock.patch.object(updater, "latest_release", boom):
            d = updater.do_check(force=True)
        self.assertFalse(d["available"])
        self.assertEqual(d["error"], "OSError")

    def _tarball(self, version, broken=False):
        src = os.path.join(self.tmp, "src", "omarchy-boxtop-abc")
        make_plugin(src, version)
        if broken:
            os.unlink(os.path.join(src, "Panel.qml"))
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            tar.add(src, arcname="omarchy-boxtop-abc")
        shutil.rmtree(os.path.join(self.tmp, "src"))
        return buf.getvalue()

    def test_apply_tarball_swaps_and_backs_up(self):
        data = self._tarball("1.7.0")
        with mock.patch.object(updater, "latest_release", lambda: ("v1.7.0", "", "https://x/t.tgz")), \
                mock.patch.object(updater, "http_get", lambda *a, **k: data), \
                mock.patch.object(updater, "is_git", lambda root=None: False):
            res = updater.do_apply()
        self.assertTrue(res["ok"], res)
        self.assertEqual(updater.local_version(self.inst), "1.7.0")
        self.assertEqual(len(os.listdir(updater.BACKUP_DIR)), 1)

    def test_apply_tarball_broken_keeps_old(self):
        data = self._tarball("1.7.0", broken=True)
        with mock.patch.object(updater, "latest_release", lambda: ("v1.7.0", "", "https://x/t.tgz")), \
                mock.patch.object(updater, "http_get", lambda *a, **k: data), \
                mock.patch.object(updater, "is_git", lambda root=None: False):
            res = updater.do_apply()
        self.assertFalse(res["ok"])
        self.assertEqual(updater.local_version(self.inst), "1.5.0")
        self.assertTrue(os.path.isfile(os.path.join(self.inst, "Panel.qml")))

    @unittest.skipUnless(shutil.which("git"), "git missing")
    def test_apply_git_fast_forward_and_rollback(self):
        origin = os.path.join(self.tmp, "origin")
        make_plugin(origin, "1.5.0")
        env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t",
                   GIT_COMMITTER_EMAIL="t@t")
        run = lambda *a, cwd=origin: subprocess.run(a, cwd=cwd, env=env, check=True, capture_output=True)
        run("git", "init", "-q", "-b", "main")
        run("git", "add", "-A")
        run("git", "commit", "-qm", "v1.5.0")
        shutil.rmtree(self.inst)
        run("git", "clone", "-q", origin, self.inst, cwd=self.tmp)
        make_plugin(origin, "1.7.0")
        run("git", "commit", "-qam", "v1.7.0")
        with mock.patch.object(updater, "latest_release", lambda: None), \
                mock.patch.object(updater, "remote_version_file", lambda: "1.7.0"):
            res = updater.do_apply()
        self.assertTrue(res["ok"], res)
        self.assertEqual(res["method"], "git")
        self.assertEqual(updater.local_version(self.inst), "1.7.0")

        # A broken upstream commit is rolled back.
        os.unlink(os.path.join(origin, "Panel.qml"))
        with open(os.path.join(origin, "VERSION"), "w") as f:
            f.write("1.7.0\n")
        run("git", "commit", "-qam", "broken")
        with mock.patch.object(updater, "latest_release", lambda: None), \
                mock.patch.object(updater, "remote_version_file", lambda: "1.7.0"):
            res = updater.do_apply()
        self.assertFalse(res["ok"])
        self.assertTrue(os.path.isfile(os.path.join(self.inst, "Panel.qml")))
        self.assertEqual(updater.local_version(self.inst), "1.7.0")


class QmlStaticTest(unittest.TestCase):
    """Cheap textual checks on Panel.qml (no Qt needed)."""

    def setUp(self):
        with open(os.path.join(ROOT, "Panel.qml"), encoding="utf-8") as f:
            self.qml = f.read()

    def test_braces_balanced(self):
        depth = 0
        for ch in self.qml:
            depth += ch == "{"
            depth -= ch == "}"
            self.assertGreaterEqual(depth, 0)
        self.assertEqual(depth, 0)

    def test_gradients_only_behind_classic_switch(self):
        # Every Gradient / createLinearGradient must sit behind root.gradients.
        lines = self.qml.splitlines()
        for i, line in enumerate(lines):
            if "createLinearGradient" in line or "gradient: Gradient" in line:
                window = "\n".join(lines[max(0, i - 6):i + 1])
                self.assertIn("root.gradients", window, "unguarded gradient at line %d" % (i + 1))

    def test_no_clipboard_access(self):
        # Qt text inputs watch the system clipboard as soon as they exist, and
        # the Wayland reader has no size cap or deadline: Boxtop must not
        # create any, nor touch the clipboard / primary selection directly.
        code = "\n".join(l.split("//", 1)[0] for l in self.qml.splitlines())
        for token in ("TextInput", "TextEdit", "TextField", "TextArea", "canPaste",
                      "clipboard", "Clipboard", "persistentSelection", "wl-paste", "wl-copy"):
            self.assertNotIn(token, code, "clipboard-capable code: %s" % token)
        for name in ("boxtop.py", "boxtopctl.py", "updater.py"):
            with open(os.path.join(ROOT, name), encoding="utf-8") as f:
                src = f.read().lower()
            for token in ("clipboard", "wl-paste", "wl-copy", "xclip", "xsel"):
                self.assertNotIn(token, src, "%s touches the clipboard (%s)" % (name, token))

    def test_glass_field_caps_input(self):
        self.assertIn("readonly property int hardCap: 128", self.qml)
        self.assertIn("isPaste(e)", self.qml)


if __name__ == "__main__":
    unittest.main()

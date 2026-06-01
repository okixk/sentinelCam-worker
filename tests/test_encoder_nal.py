"""Annex-B NAL helper tests for the pipeline H.264 encoder.

Only the pure-Python helpers are exercised here (no PyAV / GPU needed). The
actual encode path is hardware-dependent and validated at deploy time.
"""
import importlib.util
import pathlib
import sys
import unittest

_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(rel: str, name: str):
    spec = importlib.util.spec_from_file_location(name, _ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


enc = _load("web_pipeline/encoder.py", "wpp_encoder")


class NalHelperTest(unittest.TestCase):
    SPS = b"\x00\x00\x00\x01\x67\x42\x00\x1e"
    PPS = b"\x00\x00\x01\x68\xce\x3c\x80"
    IDR = b"\x00\x00\x01\x65\x88\x84"
    PSLICE = b"\x00\x00\x01\x41\x9a\x00"

    def test_iter_nal_units_handles_3_and_4_byte_start_codes(self):
        au = self.SPS + self.PPS + self.IDR
        types = [n[0] & 0x1F for n in enc.iter_nal_units(au)]
        self.assertEqual(types, [7, 8, 5])

    def test_is_keyframe(self):
        self.assertTrue(enc.is_keyframe(self.SPS + self.PPS + self.IDR))
        self.assertTrue(enc.is_keyframe(self.IDR))
        self.assertFalse(enc.is_keyframe(self.PSLICE))

    def test_empty_buffer(self):
        self.assertEqual(list(enc.iter_nal_units(b"")), [])
        self.assertFalse(enc.is_keyframe(b""))


if __name__ == "__main__":
    unittest.main()

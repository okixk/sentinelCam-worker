"""Wire-format contract test for the web <-> worker channel.

The golden header vector below MUST match the identical test in
sentinelCam-web (tests/test_streaming_protocol.py). If either repo's
protocol.py drifts, its own suite fails — that is the cross-repo guard that
replaces the old "keep both copies in sync by hand" comment.
"""
import importlib.util
import pathlib
import unittest

_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(rel: str, name: str):
    spec = importlib.util.spec_from_file_location(name, _ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


proto = _load("web_pipeline/protocol.py", "wpp_proto")


class ProtocolContractTest(unittest.TestCase):
    def test_constants(self):
        self.assertEqual(proto.PROTOCOL_VERSION, 2)
        self.assertEqual(proto.HEADER_LEN, 17)
        self.assertEqual(proto.MSG_RAW_FRAME, 0x01)
        self.assertEqual(proto.MSG_PROCESSED_FRAME, 0x02)
        self.assertEqual(proto.MSG_KEYFRAME_REQ, 0x03)
        self.assertEqual(proto.MSG_PROCESSED_H264, 0x04)

    def test_golden_header(self):
        # CONTRACT vector — identical to sentinelCam-web's test.
        payload = b"\xff\xd8\xff\x00abc"
        env = proto.encode(proto.MSG_PROCESSED_H264, 7, 1234567, payload)
        expected_header = bytes.fromhex("04" "0000000000000007" "000000000012d687")
        self.assertEqual(env[:17], expected_header)
        self.assertEqual(env[17:], payload)

    def test_round_trip(self):
        payload = b"\x00\x00\x01\x65hello"
        frame = proto.decode(proto.encode(proto.MSG_PROCESSED_H264, 42, 9001, payload))
        self.assertEqual((frame.msg_type, frame.camera_id, frame.capture_ms, frame.payload),
                         (proto.MSG_PROCESSED_H264, 42, 9001, payload))
        self.assertAlmostEqual(frame.capture_ts, 9.001, places=6)

    def test_decode_too_short(self):
        with self.assertRaises(ValueError):
            proto.decode(b"\x04short")


if __name__ == "__main__":
    unittest.main()

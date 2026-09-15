from __future__ import annotations

import numpy as np
import pytest

from sentrysem.traffic import (
    REFRESH_BITS,
    Ledger,
    fixed_audit_traffic,
    make_frame,
    parse_frame,
    quantize_embedding_int8,
)


def test_paper_traffic_ledger() -> None:
    traffic = fixed_audit_traffic(256, 768)
    assert traffic.proposal == 400
    assert traffic.candidate_lock == 272
    assert traffic.audit_seed == 208
    assert traffic.audit == 912
    assert traffic.feedback == 152
    assert traffic.protocol_bits == 1944
    assert REFRESH_BITS == 4272


def test_frame_round_trip_and_crc() -> None:
    raw = make_frame("audit", session=17, attempt=0, payload=b"12345678")
    parsed = parse_frame(raw)
    assert parsed.payload == b"12345678"
    damaged = bytearray(raw)
    damaged[5] ^= 1
    with pytest.raises(ValueError):
        parse_frame(damaged)


def test_int8_quantization_reconstructs_unit_features() -> None:
    rng = np.random.default_rng(8)
    vectors = rng.normal(size=(4, 512)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    result = quantize_embedding_int8(vectors)
    assert result.values.dtype == np.int8
    assert np.allclose(np.linalg.norm(result.reconstructed, axis=1), 1.0, atol=1e-6)


def test_ledger_accepts_one_fixed_audit_trace() -> None:
    ledger = Ledger()
    ledger.add("tx_to_rx", "proposal", make_frame("proposal", 1, 0, b"p" * 32))
    ledger.add("rx_to_tx", "lock", make_frame("lock", 1, 0, b"c" * 16))
    ledger.add("tx_to_rx", "seed", make_frame("seed", 1, 0, b"s" * 8))
    ledger.add("tx_to_rx", "audit", make_frame("audit", 1, 0, b"a" * 96))
    ledger.add("rx_to_tx", "feedback", make_frame("feedback", 1, 0, b"\x01"))
    assert ledger.validate()
    assert ledger.total_bits == 1944

"""Wire-format serialization and logical traffic accounting for SentrySem."""

from __future__ import annotations

import hashlib
import struct
import zlib
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

Direction = Literal["tx_to_rx", "rx_to_tx"]
FrameKind = Literal["proposal", "lock", "seed", "audit", "feedback", "embedding"]

FRAME_HEADER = struct.Struct(">4sBBBBIH")
MAGIC = b"SNT1"
PROTOCOL_VERSION = 1
FRAME_OVERHEAD_BYTES = FRAME_HEADER.size + 4
FRAME_OVERHEAD_BITS = FRAME_OVERHEAD_BYTES * 8
KINDS: dict[FrameKind, int] = {
    "proposal": 1,
    "lock": 2,
    "seed": 3,
    "audit": 4,
    "feedback": 5,
    "embedding": 6,
}
KIND_NAMES = {value: key for key, value in KINDS.items()}

CANDIDATE_LOCK_BYTES = 16
AUDIT_SEED_BYTES = 8
FEEDBACK_BYTES = 1
EMBEDDING_DIMENSION = 512
REFRESH_SCALE_BYTES = 4

CANDIDATE_LOCK_BITS = FRAME_OVERHEAD_BITS + 8 * CANDIDATE_LOCK_BYTES
AUDIT_SEED_BITS = FRAME_OVERHEAD_BITS + 8 * AUDIT_SEED_BYTES
FEEDBACK_BITS = FRAME_OVERHEAD_BITS + 8 * FEEDBACK_BYTES
REFRESH_BITS = FRAME_OVERHEAD_BITS + 8 * (REFRESH_SCALE_BYTES + EMBEDDING_DIMENSION)

FALLBACK_BITS = REFRESH_BITS

VERIFIER_TAG = b"Frozen-ResNet18-ImageNetV1-cosine-v1"


@dataclass(frozen=True)
class ParsedFrame:
    """Decoded fields from one protocol frame."""

    kind: FrameKind
    attempt: int
    session: int
    flags: int
    payload: bytes


@dataclass(frozen=True)
class TrafficBreakdown:
    """Logical bits consumed by a fixed proposal--audit transaction."""

    proposal: int
    candidate_lock: int
    audit_seed: int
    audit: int
    feedback: int

    @property
    def forward_bits(self) -> int:
        return self.proposal + self.audit_seed + self.audit

    @property
    def reverse_bits(self) -> int:
        return self.candidate_lock + self.feedback

    @property
    def protocol_bits(self) -> int:
        return self.forward_bits + self.reverse_bits

    def completion_bits(self, reused: bool) -> int:
        return self.protocol_bits + (0 if reused else REFRESH_BITS)


@dataclass(frozen=True)
class QuantizedEmbedding:
    """A per-vector symmetric int8 representation and its reconstruction."""

    values: NDArray[np.int8]
    scales: NDArray[np.float32]
    reconstructed: NDArray[np.float32]


def frame_bits(payload_bits: int) -> int:
    """Return serialized frame length for a byte-aligned payload."""

    if payload_bits < 0 or payload_bits % 8:
        raise ValueError("payload_bits must be a non-negative multiple of eight")
    return FRAME_OVERHEAD_BITS + payload_bits


proposal_frame_bits = frame_bits


def fixed_audit_traffic(proposal_bits: int = 256, audit_bits: int = 768) -> TrafficBreakdown:
    """Return the declared traffic ledger for the fixed SentrySem protocol."""

    if proposal_bits <= 0 or audit_bits <= 0:
        raise ValueError("proposal_bits and audit_bits must be positive")
    return TrafficBreakdown(
        proposal=frame_bits(proposal_bits),
        candidate_lock=CANDIDATE_LOCK_BITS,
        audit_seed=AUDIT_SEED_BITS,
        audit=frame_bits(audit_bits),
        feedback=FEEDBACK_BITS,
    )


def make_frame(
    kind: FrameKind,
    session: int,
    attempt: int,
    payload: bytes | bytearray | memoryview = b"",
    flags: int = 0,
) -> bytes:
    """Serialize one version-1 frame with a CRC-protected 14-byte header."""

    if kind not in KINDS:
        raise ValueError(f"unknown frame kind: {kind}")
    content = bytes(payload)
    if not 0 <= session <= 0xFFFFFFFF:
        raise ValueError("session must fit in an unsigned 32-bit field")
    if not 0 <= attempt <= 0xFF or not 0 <= flags <= 0xFF:
        raise ValueError("attempt and flags must fit in unsigned 8-bit fields")
    if len(content) > 0xFFFF:
        raise ValueError("payload exceeds the frame length field")
    header = FRAME_HEADER.pack(
        MAGIC, PROTOCOL_VERSION, KINDS[kind], attempt, flags, session, len(content)
    )
    return header + struct.pack(">I", zlib.crc32(header)) + content


def parse_frame(raw: bytes | bytearray | memoryview) -> ParsedFrame:
    """Validate and decode one serialized protocol frame."""

    data = bytes(raw)
    if len(data) < FRAME_OVERHEAD_BYTES:
        raise ValueError("truncated frame")
    header = data[: FRAME_HEADER.size]
    magic, version, kind_code, attempt, flags, session, length = FRAME_HEADER.unpack(header)
    stored_crc = struct.unpack(">I", data[FRAME_HEADER.size : FRAME_OVERHEAD_BYTES])[0]
    if magic != MAGIC or version != PROTOCOL_VERSION or kind_code not in KIND_NAMES:
        raise ValueError("invalid frame identity")
    if stored_crc != zlib.crc32(header):
        raise ValueError("header CRC failure")
    payload = data[FRAME_OVERHEAD_BYTES:]
    if len(payload) != length:
        raise ValueError("payload length mismatch")
    return ParsedFrame(KIND_NAMES[kind_code], attempt, session, flags, payload)


def candidate_lock_record(
    candidate: ArrayLike,
    session: int,
    attempt: int,
    verifier_digest: bytes | bytearray | memoryview,
) -> bytes:
    """Create the 16-byte phase record that binds an audit to one cache item."""

    digest = bytes(verifier_digest)
    if len(digest) != 32:
        raise ValueError("verifier_digest must contain 32 bytes")
    vector = np.ascontiguousarray(np.asarray(candidate, dtype=np.float32))
    binding = (
        VERIFIER_TAG + digest + struct.pack(">IB", int(session), int(attempt)) + vector.tobytes()
    )
    return hashlib.sha256(binding).digest()[:CANDIDATE_LOCK_BYTES]


class Ledger:
    """Validated record of serialized messages in one protocol session."""

    def __init__(self) -> None:
        self.rows: list[dict[str, int | str]] = []

    def add(self, direction: Direction, kind: FrameKind, raw: bytes) -> None:
        if direction not in ("tx_to_rx", "rx_to_tx"):
            raise ValueError(f"unknown direction: {direction}")
        parsed = parse_frame(raw)
        if parsed.kind != kind:
            raise ValueError("ledger kind differs from the serialized frame")
        self.rows.append(
            {
                "direction": direction,
                "kind": kind,
                "attempt": parsed.attempt,
                "session": parsed.session,
                "payload_bytes": len(parsed.payload),
                "payload_hex": parsed.payload.hex(),
                "bytes": len(raw),
            }
        )

    def validate(self, max_attempts: int = 3) -> bool:
        """Check direction, payload size, ordering, and audit-seed uniqueness."""

        if not self.rows:
            raise ValueError("ledger is empty")
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        sessions = {int(row["session"]) for row in self.rows}
        if len(sessions) != 1:
            raise ValueError("ledger combines more than one session")
        expected_direction: dict[str, Direction] = {
            "proposal": "tx_to_rx",
            "lock": "rx_to_tx",
            "seed": "tx_to_rx",
            "audit": "tx_to_rx",
            "feedback": "rx_to_tx",
            "embedding": "tx_to_rx",
        }
        fixed_payload_bytes = {
            "lock": CANDIDATE_LOCK_BYTES,
            "seed": AUDIT_SEED_BYTES,
            "feedback": FEEDBACK_BYTES,
            "embedding": REFRESH_SCALE_BYTES + EMBEDDING_DIMENSION,
        }
        for row in self.rows:
            kind = str(row["kind"])
            if row["direction"] != expected_direction[kind]:
                raise ValueError(f"invalid direction for {kind}")
            if kind in fixed_payload_bytes and row["payload_bytes"] != fixed_payload_bytes[kind]:
                raise ValueError(f"invalid payload size for {kind}")
            if kind in ("proposal", "audit") and int(row["payload_bytes"]) <= 0:
                raise ValueError(f"{kind} payload must be non-empty")
        seeds = [str(row["payload_hex"]) for row in self.rows if row["kind"] == "seed"]
        if len(seeds) != len(set(seeds)):
            raise ValueError("an audit seed was reused within the session")
        interactive = [row for row in self.rows if row["kind"] not in ("proposal", "embedding")]
        if not interactive:
            kinds = [row["kind"] for row in self.rows]
            if kinds not in (["proposal"], ["embedding"]):
                raise ValueError("a non-interactive ledger must contain one complete message")
            return True
        proposal_positions = [i for i, row in enumerate(self.rows) if row["kind"] == "proposal"]
        if proposal_positions != [0]:
            raise ValueError("an audited session must start with exactly one proposal")
        attempts = sorted({int(row["attempt"]) for row in interactive})
        if attempts != list(range(attempts[-1] + 1)) or attempts[-1] >= max_attempts:
            raise ValueError("attempt identifiers are not consecutive and bounded")
        for position, row in enumerate(interactive):
            previous = interactive[position - 1] if position else None
            following = interactive[position + 1] if position + 1 < len(interactive) else None
            kind = row["kind"]
            attempt = int(row["attempt"])
            if kind == "lock":
                initial = attempt == 0 and previous is None
                refinement = (
                    previous is not None
                    and previous["kind"] == "feedback"
                    and previous["payload_hex"] == "02"
                    and attempt == int(previous["attempt"]) + 1
                )
                if not (initial or refinement):
                    raise ValueError("candidate lock appears outside a valid phase boundary")
                if (
                    following is None
                    or following["kind"] != "seed"
                    or following["attempt"] != attempt
                ):
                    raise ValueError("candidate lock must be followed by its audit seed")
            elif kind == "seed":
                if previous is None or previous["kind"] != "lock" or previous["attempt"] != attempt:
                    raise ValueError("audit seed must follow its candidate lock")
                if (
                    following is None
                    or following["kind"] != "audit"
                    or following["attempt"] != attempt
                ):
                    raise ValueError("audit seed must precede its audit signs")
            elif kind == "audit":
                valid_previous = previous is not None and previous["kind"] in ("seed", "feedback")
                if not valid_previous or previous["attempt"] != attempt:
                    raise ValueError("audit signs are outside their locked attempt")
                if previous["kind"] == "feedback" and previous["payload_hex"] != "00":
                    raise ValueError("only a continue response permits another audit batch")
                if (
                    following is None
                    or following["kind"] != "feedback"
                    or following["attempt"] != attempt
                ):
                    raise ValueError("every audit batch must receive feedback")
            elif kind == "feedback":
                if (
                    previous is None
                    or previous["kind"] != "audit"
                    or previous["attempt"] != attempt
                ):
                    raise ValueError("feedback must follow an audit batch")
                code = str(row["payload_hex"])
                if code not in ("00", "01", "02", "03"):
                    raise ValueError("unknown feedback code")
                if code == "00":
                    if (
                        following is None
                        or following["kind"] != "audit"
                        or following["attempt"] != attempt
                    ):
                        raise ValueError("continue feedback must be followed by an audit batch")
                elif code == "02":
                    if (
                        following is None
                        or following["kind"] != "lock"
                        or int(following["attempt"]) != attempt + 1
                    ):
                        raise ValueError("refine feedback must be followed by a new candidate lock")
                elif following is not None:
                    raise ValueError("certify and refresh responses terminate the audit")
        return True

    def breakdown(self) -> dict[str, dict[str, int]]:
        result: dict[str, dict[str, int]] = {}
        for row in self.rows:
            key = f"{row['direction']}:{row['kind']}"
            item = result.setdefault(key, {"messages": 0, "bytes": 0})
            item["messages"] += 1
            item["bytes"] += int(row["bytes"])
        return result

    @property
    def forward_bits(self) -> int:
        return 8 * sum(int(row["bytes"]) for row in self.rows if row["direction"] == "tx_to_rx")

    @property
    def reverse_bits(self) -> int:
        return 8 * sum(int(row["bytes"]) for row in self.rows if row["direction"] == "rx_to_tx")

    @property
    def total_bits(self) -> int:
        return self.forward_bits + self.reverse_bits


def quantize_embedding_int8(vectors: ArrayLike) -> QuantizedEmbedding:
    """Apply the paper's per-vector symmetric int8 semantic refresh codec."""

    source = np.asarray(vectors, dtype=np.float32)
    one_vector = source.ndim == 1
    if one_vector:
        source = source[None, :]
    if source.ndim != 2 or source.shape[1] != EMBEDDING_DIMENSION:
        raise ValueError(f"vectors must have shape (n, {EMBEDDING_DIMENSION})")
    scales = np.maximum(np.max(np.abs(source), axis=1, keepdims=True) / 127.0, 1e-8)
    quantized = np.clip(np.rint(source / scales), -127, 127).astype(np.int8)
    reconstructed = quantized.astype(np.float32) * scales.astype(np.float32)
    norms = np.linalg.norm(reconstructed, axis=1, keepdims=True).clip(1e-12)
    reconstructed = np.asarray(reconstructed / norms, dtype=np.float32)
    if one_vector:
        return QuantizedEmbedding(quantized[0], scales.reshape(-1), reconstructed[0])
    return QuantizedEmbedding(quantized, scales.astype(np.float32), reconstructed)


def serialize_quantized_embedding(scale: float, values: ArrayLike) -> bytes:
    """Serialize one 512-dimensional int8 vector after a big-endian float scale."""

    quantized = np.asarray(values, dtype=np.int8)
    if quantized.shape != (EMBEDDING_DIMENSION,):
        raise ValueError(f"values must have shape ({EMBEDDING_DIMENSION},)")
    return struct.pack(">f", float(scale)) + np.ascontiguousarray(quantized).tobytes()


def int8_semantic_refresh(
    query: ArrayLike,
    bank: ArrayLike,
) -> tuple[NDArray[np.int32], NDArray[np.float64]]:
    """Reconstruct query features and return their diagnostic nearest cache items."""

    source = np.asarray(query, dtype=np.float32)
    cache = np.asarray(bank, dtype=np.float32)
    if source.ndim != 2 or cache.ndim != 2 or source.shape[1] != cache.shape[1]:
        raise ValueError("query and bank must be compatible two-dimensional arrays")
    encoded = quantize_embedding_int8(source)
    reconstructed = np.asarray(encoded.reconstructed, dtype=np.float32)
    nearest = np.argmax(reconstructed @ cache.T, axis=1).astype(np.int32)
    similarity = np.einsum("ij,ij->i", source, reconstructed, dtype=np.float64)
    return nearest, np.asarray(similarity, dtype=np.float64)


def completion_bits(reuse: bool, proposal_bits: int = 256, audit_bits: int = 768) -> int:
    """Return fallback-inclusive logical traffic for one completed session."""

    return fixed_audit_traffic(proposal_bits, audit_bits).completion_bits(reuse)

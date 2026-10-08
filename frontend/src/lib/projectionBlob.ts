// Parser for the binary projection payload produced by
// GET /api/projections/{param_hash}/blob.
//
// Layout (little-endian):
//   bytes  0..3   magic 'KUMP'
//   bytes  4..7   uint32 version (1)
//   bytes  8..11  uint32 count
//   bytes 12..15  uint32 reserved
//   bytes 16..    uint32[count] ids
//                 float32[count] x
//                 float32[count] y
//
// At ~12 bytes/point this is ~3-5x smaller than the JSON
// `[{"id":N,"x":F,"y":F}]` shape and skips JSON.parse entirely.

const MAGIC = 0x504d554b; // 'KUMP' little-endian: 'K' 0x4B, 'U' 0x55, 'M' 0x4D, 'P' 0x50

export interface ProjectionCoord {
  id: number;
  x: number;
  y: number;
}

export function parseProjectionBlob(buf: ArrayBuffer): ProjectionCoord[] {
  if (buf.byteLength < 16) {
    throw new Error("projection blob too short");
  }
  const header = new DataView(buf, 0, 16);
  const magic = header.getUint32(0, true);
  if (magic !== MAGIC) {
    throw new Error(`projection blob bad magic: 0x${magic.toString(16)}`);
  }
  const version = header.getUint32(4, true);
  if (version !== 1) {
    throw new Error(`projection blob unsupported version: ${version}`);
  }
  const count = header.getUint32(8, true);

  const idsOffset = 16;
  const xsOffset = idsOffset + count * 4;
  const ysOffset = xsOffset + count * 4;
  const expected = ysOffset + count * 4;
  if (buf.byteLength < expected) {
    throw new Error(
      `projection blob truncated: expected ${expected} bytes, got ${buf.byteLength}`,
    );
  }

  const ids = new Uint32Array(buf, idsOffset, count);
  const xs = new Float32Array(buf, xsOffset, count);
  const ys = new Float32Array(buf, ysOffset, count);

  const out: ProjectionCoord[] = new Array(count);
  for (let i = 0; i < count; i++) {
    out[i] = { id: ids[i], x: xs[i], y: ys[i] };
  }
  return out;
}

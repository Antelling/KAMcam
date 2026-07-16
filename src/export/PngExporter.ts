const CRC_TABLE = new Int32Array(256);
for (let i = 0; i < 256; i++) {
  let c = i;
  for (let k = 0; k < 8; k++) {
    c = (c & 1) ? (0xEDB88320 ^ (c >>> 1)) : (c >>> 1);
  }
  CRC_TABLE[i] = c;
}

function crc32(data: Uint8Array): number {
  let crc = -1;
  for (let i = 0; i < data.length; i++) {
    crc = CRC_TABLE[(crc ^ data[i]) & 0xFF] ^ (crc >>> 8);
  }
  return (crc ^ -1) >>> 0;
}

function adler32(data: Uint8Array): number {
  let a = 1;
  let b = 0;
  const MOD = 65521;
  for (let i = 0; i < data.length; i++) {
    a = (a + data[i]) % MOD;
    b = (b + a) % MOD;
  }
  return ((b << 16) | a) >>> 0;
}

function writeChunk(type: string, data: Uint8Array, out: number[]): void {
  const typeBytes = new TextEncoder().encode(type);
  const len = data.length;
  out.push((len >>> 24) & 0xFF, (len >>> 16) & 0xFF, (len >>> 8) & 0xFF, len & 0xFF);
  for (let i = 0; i < typeBytes.length; i++) out.push(typeBytes[i]);
  for (let i = 0; i < data.length; i++) out.push(data[i]);
  const crcInput = new Uint8Array(typeBytes.length + data.length);
  crcInput.set(typeBytes, 0);
  crcInput.set(data, typeBytes.length);
  const crc = crc32(crcInput);
  out.push((crc >>> 24) & 0xFF, (crc >>> 16) & 0xFF, (crc >>> 8) & 0xFF, crc & 0xFF);
}

function deflateUncompressed(data: Uint8Array): Uint8Array {
  const MAX_BLOCK = 65535;
  const out: number[] = [];
  let pos = 0;
  while (pos < data.length) {
    const final = pos + MAX_BLOCK >= data.length ? 1 : 0;
    const len = Math.min(MAX_BLOCK, data.length - pos);
    const nlen = (~len) & 0xFFFF;
    out.push(final);
    out.push(len & 0xFF, (len >>> 8) & 0xFF);
    out.push(nlen & 0xFF, (nlen >>> 8) & 0xFF);
    for (let i = 0; i < len; i++) out.push(data[pos + i]);
    pos += len;
  }
  return new Uint8Array(out);
}

function buildImageDataFloat32(values: Float32Array, width: number, height: number): Uint8Array {
  const rowSize = 1 + width * 4;
  const out = new Uint8Array(height * rowSize);
  const u32 = new Uint32Array(values.buffer, values.byteOffset, values.length);
  for (let y = 0; y < height; y++) {
    const rowStart = y * rowSize;
    out[rowStart] = 0;
    for (let x = 0; x < width; x++) {
      const bits = u32[y * width + x];
      const idx = rowStart + 1 + x * 4;
      out[idx] = (bits >>> 24) & 0xFF;
      out[idx + 1] = (bits >>> 16) & 0xFF;
      out[idx + 2] = (bits >>> 8) & 0xFF;
      out[idx + 3] = bits & 0xFF;
    }
  }
  return out;
}

export function encodeFloat32Png(
  values: Float32Array,
  width: number,
  height: number,
  metadata?: Record<string, string>,
): Uint8Array {
  const out: number[] = [137, 80, 78, 71, 13, 10, 26, 10];

  const ihdr = new Uint8Array(13);
  const view = new DataView(ihdr.buffer);
  view.setUint32(0, width);
  view.setUint32(4, height);
  ihdr[8] = 16;
  ihdr[9] = 4;
  ihdr[10] = 0;
  ihdr[11] = 0;
  ihdr[12] = 0;
  writeChunk('IHDR', ihdr, out);

  if (metadata) {
    const encoder = new TextEncoder();
    for (const [key, value] of Object.entries(metadata)) {
      const text = encoder.encode(`${key}\0${value}`);
      writeChunk('tEXt', text, out);
    }
  }

  const imageData = buildImageDataFloat32(values, width, height);
  const deflated = deflateUncompressed(imageData);
  const zlib = new Uint8Array(2 + deflated.length + 4);
  zlib[0] = 0x78;
  zlib[1] = 0x01;
  zlib.set(deflated, 2);
  const checksum = adler32(imageData);
  const zlibView = new DataView(zlib.buffer);
  zlibView.setUint32(2 + deflated.length, checksum);
  writeChunk('IDAT', zlib, out);

  writeChunk('IEND', new Uint8Array(0), out);

  return new Uint8Array(out);
}

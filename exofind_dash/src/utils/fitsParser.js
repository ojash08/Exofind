/**
 * Lightweight FITS file parser for browser use.
 * 
 * FITS files are structured in 2880-byte blocks.
 * Each HDU (Header Data Unit) has:
 *   - A header: multiple 80-char keyword records, padded to 2880-byte boundary
 *   - A data segment: binary data, padded to 2880-byte boundary
 * 
 * We parse:
 *   1. Primary HDU headers
 *   2. BinTable extension HDU to extract TIME and FLUX columns
 */

const BLOCK_SIZE = 2880;
const CARD_SIZE = 80;

/**
 * Parse a FITS file from an ArrayBuffer.
 * Returns { headers: {}, columns: { TIME: Float64Array, FLUX: Float64Array }, error?: string }
 */
export function parseFits(buffer) {
  try {
    const view = new DataView(buffer);
    const bytes = new Uint8Array(buffer);

    // Parse Primary HDU
    let offset = 0;
    const primaryHeader = parseHeader(bytes, offset);
    offset = primaryHeader.nextOffset;

    // Skip primary data if any
    const primaryDataSize = computeDataSize(primaryHeader.cards);
    offset += padTo2880(primaryDataSize);

    // Now look for BinTable extension(s)
    let lightCurveData = null;

    while (offset < buffer.byteLength) {
      const extHeader = parseHeader(bytes, offset);
      offset = extHeader.nextOffset;

      const xtension = extHeader.cards['XTENSION'];
      const naxis1 = parseInt(extHeader.cards['NAXIS1']) || 0;
      const naxis2 = parseInt(extHeader.cards['NAXIS2']) || 0;
      const dataBytes = naxis1 * naxis2;

      if (xtension === 'BINTABLE' && !lightCurveData) {
        // Try to extract TIME and FLUX columns
        lightCurveData = parseBinTable(view, offset, extHeader.cards, naxis1, naxis2);
      }

      offset += padTo2880(dataBytes);

      // Also skip any heap (PCOUNT)
      const pcount = parseInt(extHeader.cards['PCOUNT']) || 0;
      if (pcount > 0) {
        offset += padTo2880(pcount);
      }
    }

    if (lightCurveData) {
      return {
        headers: primaryHeader.cards,
        columns: lightCurveData.columns,
        meta: lightCurveData.meta
      };
    }

    // If no BinTable found, try to return just headers
    return {
      headers: primaryHeader.cards,
      columns: null,
      error: 'No BinTable extension with TIME/FLUX columns found in this FITS file.'
    };
  } catch (e) {
    return { headers: {}, columns: null, error: `FITS parsing error: ${e.message}` };
  }
}

/**
 * Parse header cards starting at offset.
 * Returns { cards: { KEY: VALUE }, nextOffset }
 */
function parseHeader(bytes, startOffset) {
  const cards = {};
  let offset = startOffset;

  while (offset < bytes.length) {
    const cardStr = readAscii(bytes, offset, CARD_SIZE);
    offset += CARD_SIZE;

    if (cardStr.startsWith('END')) {
      // Pad to next 2880-byte boundary
      const headerBytes = offset - startOffset;
      const padded = padTo2880(headerBytes);
      offset = startOffset + padded;
      break;
    }

    const key = cardStr.substring(0, 8).trim();
    if (cardStr[8] === '=' && key) {
      let valueStr = cardStr.substring(10).trim();

      // Remove trailing comment (after /)
      const slashIdx = findCommentSlash(valueStr);
      if (slashIdx >= 0) {
        valueStr = valueStr.substring(0, slashIdx).trim();
      }

      // Parse value
      if (valueStr.startsWith("'")) {
        // String value
        const endQuote = valueStr.indexOf("'", 1);
        cards[key] = endQuote > 0 ? valueStr.substring(1, endQuote).trim() : valueStr.substring(1).trim();
      } else if (valueStr === 'T') {
        cards[key] = true;
      } else if (valueStr === 'F') {
        cards[key] = false;
      } else {
        const num = parseFloat(valueStr);
        cards[key] = isNaN(num) ? valueStr : num;
      }
    }
  }

  return { cards, nextOffset: offset };
}

/**
 * Find the comment slash in a FITS value string,
 * being careful not to split inside a quoted string.
 */
function findCommentSlash(str) {
  let inQuote = false;
  for (let i = 0; i < str.length; i++) {
    if (str[i] === "'") inQuote = !inQuote;
    if (str[i] === '/' && !inQuote) return i;
  }
  return -1;
}

/**
 * Parse a BinTable extension to extract TIME and FLUX columns.
 */
function parseBinTable(view, dataOffset, cards, naxis1, naxis2) {
  const tfields = parseInt(cards['TFIELDS']) || 0;

  // Build column descriptors
  const columns = [];
  for (let i = 1; i <= tfields; i++) {
    const ttype = (cards[`TTYPE${i}`] || '').toUpperCase();
    const tform = cards[`TFORM${i}`] || '';
    columns.push({ index: i, name: ttype, form: tform });
  }

  // Find TIME column
  const timeCol = columns.find(c =>
    c.name === 'TIME' || c.name === 'BARYTIME' || c.name === 'BJD'
  );

  // Find FLUX column (prefer PDCSAP_FLUX, then SAP_FLUX, then FLUX, then DETRENDED_FLUX)
  const fluxCol =
    columns.find(c => c.name === 'PDCSAP_FLUX') ||
    columns.find(c => c.name === 'SAP_FLUX') ||
    columns.find(c => c.name === 'FLUX') ||
    columns.find(c => c.name === 'DETRENDED_FLUX');

  if (!timeCol || !fluxCol) {
    return null;
  }

  // Calculate byte offsets for each column within a row
  const colOffsets = computeColumnOffsets(columns);

  const timeOffset = colOffsets[timeCol.index - 1];
  const fluxOffset = colOffsets[fluxCol.index - 1];

  const timeSize = getFormatByteSize(timeCol.form);
  const fluxSize = getFormatByteSize(fluxCol.form);
  const timeReader = getReader(timeCol.form);
  const fluxReader = getReader(fluxCol.form);

  // Read data rows
  const timeData = [];
  const fluxData = [];

  for (let row = 0; row < naxis2; row++) {
    const rowOffset = dataOffset + (row * naxis1);

    const t = timeReader(view, rowOffset + timeOffset);
    const f = fluxReader(view, rowOffset + fluxOffset);

    // Skip NaN values
    if (!isNaN(t) && !isNaN(f) && isFinite(t) && isFinite(f)) {
      timeData.push(t);
      fluxData.push(f);
    }
  }

  // Determine which flux column was used
  const meta = {
    fluxColumn: fluxCol.name,
    timeColumn: timeCol.name,
    totalRows: naxis2,
    validRows: timeData.length,
  };

  return { columns: { TIME: timeData, FLUX: fluxData }, meta };
}

/**
 * Compute byte offset of each column within a row.
 */
function computeColumnOffsets(columns) {
  const offsets = [];
  let offset = 0;
  for (const col of columns) {
    offsets.push(offset);
    offset += getFormatByteSize(col.form);
  }
  return offsets;
}

/**
 * Get byte size of a TFORM format code.
 * Common: 1D = 8 bytes (float64), 1E = 4 bytes (float32), 1J = 4 bytes (int32), 1K = 8 bytes (int64)
 */
function getFormatByteSize(form) {
  const match = form.match(/^(\d*)([LXBIJKAEDCMP])/);
  if (!match) return 0;
  const count = parseInt(match[1]) || 1;
  const type = match[2];
  const sizes = { L: 1, X: 1, B: 1, I: 2, J: 4, K: 8, A: 1, E: 4, D: 8, C: 8, M: 16, P: 8 };
  return count * (sizes[type] || 0);
}

/**
 * Get a reader function for a TFORM code. Returns (DataView, offset) => number.
 */
function getReader(form) {
  const match = form.match(/^(\d*)([LXBIJKAEDCMP])/);
  if (!match) return () => NaN;
  const type = match[2];

  switch (type) {
    case 'D': return (v, o) => v.getFloat64(o, false); // big-endian double
    case 'E': return (v, o) => v.getFloat32(o, false); // big-endian float
    case 'J': return (v, o) => v.getInt32(o, false);    // big-endian int32
    case 'K': return (v, o) => Number(v.getBigInt64(o, false)); // big-endian int64
    case 'I': return (v, o) => v.getInt16(o, false);    // big-endian int16
    case 'B': return (v, o) => v.getUint8(o);           // byte
    default:  return () => NaN;
  }
}

/**
 * Read ASCII string from byte array.
 */
function readAscii(bytes, offset, length) {
  let s = '';
  for (let i = 0; i < length && (offset + i) < bytes.length; i++) {
    s += String.fromCharCode(bytes[offset + i]);
  }
  return s;
}

/**
 * Compute data size from primary header NAXIS values.
 */
function computeDataSize(cards) {
  const naxis = parseInt(cards['NAXIS']) || 0;
  if (naxis === 0) return 0;
  const bitpix = Math.abs(parseInt(cards['BITPIX']) || 0);
  let size = bitpix / 8;
  for (let i = 1; i <= naxis; i++) {
    size *= parseInt(cards[`NAXIS${i}`]) || 1;
  }
  return size;
}

/**
 * Pad a byte count to the next 2880-byte boundary.
 */
function padTo2880(n) {
  if (n <= 0) return 0;
  return Math.ceil(n / BLOCK_SIZE) * BLOCK_SIZE;
}

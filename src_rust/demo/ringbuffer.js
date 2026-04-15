'use strict';
// Verbatim from samejs — SPSC wait-free ring buffer over SharedArrayBuffer.
// Used by both the AudioWorklet and the Web Worker.

class RingBuffer {
  static getStorageForCapacity(capacity, type) {
    if (!type.BYTES_PER_ELEMENT) throw TypeError("Pass in a ArrayBuffer subclass");
    const bytes = 8 + (capacity + 1) * type.BYTES_PER_ELEMENT;
    return new SharedArrayBuffer(bytes);
  }

  constructor(sab, type) {
    if (type.BYTES_PER_ELEMENT === undefined)
      throw TypeError("Pass a concrete typed array class as second argument");
    this._type = type;
    this._capacity = (sab.byteLength - 8) / type.BYTES_PER_ELEMENT;
    this.buf = sab;
    this.write_ptr = new Uint32Array(this.buf, 0, 1);
    this.read_ptr  = new Uint32Array(this.buf, 4, 1);
    this.storage   = new type(this.buf, 8, this._capacity);
  }

  type() { return this._type.name; }
  capacity() { return this._capacity - 1; }

  push(elements, length, offset = 0) {
    const rd = Atomics.load(this.read_ptr, 0);
    const wr = Atomics.load(this.write_ptr, 0);
    if ((wr + 1) % this._storage_capacity() === rd) return 0;
    const len       = length !== undefined ? length : elements.length;
    const to_write  = Math.min(this._available_write(rd, wr), len);
    const first     = Math.min(this._storage_capacity() - wr, to_write);
    const second    = to_write - first;
    this._copy(elements, offset,         this.storage, wr, first);
    this._copy(elements, offset + first, this.storage, 0,  second);
    Atomics.store(this.write_ptr, 0, (wr + to_write) % this._storage_capacity());
    return to_write;
  }

  pop(elements, length, offset = 0) {
    const rd = Atomics.load(this.read_ptr, 0);
    const wr = Atomics.load(this.write_ptr, 0);
    if (wr === rd) return 0;
    const len      = length !== undefined ? length : elements.length;
    const to_read  = Math.min(this._available_read(rd, wr), len);
    const first    = Math.min(this._storage_capacity() - rd, to_read);
    const second   = to_read - first;
    this._copy(this.storage, rd,      elements, offset,         first);
    this._copy(this.storage, 0,       elements, offset + first, second);
    Atomics.store(this.read_ptr, 0, (rd + to_read) % this._storage_capacity());
    return to_read;
  }

  empty() {
    return Atomics.load(this.write_ptr, 0) === Atomics.load(this.read_ptr, 0);
  }

  available_read() {
    const rd = Atomics.load(this.read_ptr, 0);
    const wr = Atomics.load(this.write_ptr, 0);
    return this._available_read(rd, wr);
  }

  available_write() {
    const rd = Atomics.load(this.read_ptr, 0);
    const wr = Atomics.load(this.write_ptr, 0);
    return this._available_write(rd, wr);
  }

  _available_read(rd, wr)  { return (wr + this._storage_capacity() - rd) % this._storage_capacity(); }
  _available_write(rd, wr) { return this.capacity() - this._available_read(rd, wr); }
  _storage_capacity()      { return this._capacity; }

  _copy(input, offset_input, output, offset_output, size) {
    for (let i = 0; i < size; i++) output[offset_output + i] = input[offset_input + i];
  }
}

class AudioWriter {
  constructor(ringbuf) {
    if (ringbuf.type() !== "Float32Array") throw TypeError("Requires Float32Array ring buffer");
    this.ringbuf = ringbuf;
  }
  enqueue(buf)       { return this.ringbuf.push(buf); }
  available_write()  { return this.ringbuf.available_write(); }
}

class AudioReader {
  constructor(ringbuf) {
    if (ringbuf.type() !== "Float32Array") throw TypeError("Requires Float32Array ring buffer");
    this.ringbuf = ringbuf;
  }
  dequeue(buf)      { if (this.ringbuf.empty()) return 0; return this.ringbuf.pop(buf); }
  available_read()  { return this.ringbuf.available_read(); }
}

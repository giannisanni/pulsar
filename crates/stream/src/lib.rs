//! Expert streaming core. Milestone 2 scope: an io_uring fetch engine that
//! reads expert slabs from a GGUF file with O_DIRECT at real queue depth,
//! plus the plan format shared with the C reference benchmark.

use gguf::Gguf;

/// One disk read: an expert tensor slab at an absolute file offset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Read {
    pub offset: u64,
    pub len: u64,
}

/// Build the universe of per-expert slab reads for every streamed layer of
/// a MoE gguf: for each routed-expert tensor (gate/up/down) of each layer,
/// one Read per expert. Mirrors ds4's expert addressing: slab e lives at
/// tensor_base + e * expert_bytes.
pub fn expert_reads(g: &Gguf, model_len: u64) -> Result<Vec<Read>, String> {
    let n_expert = g
        .arch_meta("expert_count")
        .and_then(gguf::Value::as_u64)
        .ok_or("missing expert_count")?;
    let mut out = Vec::new();
    for t in &g.tensors {
        if !t.name.ends_with("_exps.weight") {
            continue;
        }
        if t.dims.len() != 3 || t.dims[2] != n_expert {
            return Err(format!("{}: unexpected exps dims {:?}", t.name, t.dims));
        }
        let row = t
            .ty
            .row_bytes(t.dims[0])
            .ok_or_else(|| format!("{}: unmodeled type {:?}", t.name, t.ty))?;
        let expert_bytes = row * t.dims[1];
        let base = g.data_offset + t.offset;
        for e in 0..n_expert {
            let offset = base + e * expert_bytes;
            if offset + expert_bytes > model_len {
                return Err(format!("{}: expert {} beyond eof", t.name, e));
            }
            out.push(Read { offset, len: expert_bytes });
        }
    }
    if out.is_empty() {
        return Err("no *_exps.weight tensors found".into());
    }
    Ok(out)
}

/// Plan file: "offset len\n" per read. Shared with the C benchmark so both
/// implementations perform byte-identical I/O.
pub fn plan_to_string(reads: &[Read]) -> String {
    let mut s = String::with_capacity(reads.len() * 24);
    for r in reads {
        s.push_str(&format!("{} {}\n", r.offset, r.len));
    }
    s
}

pub fn plan_from_str(s: &str) -> Result<Vec<Read>, String> {
    s.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let mut it = l.split_whitespace();
            let offset = it
                .next()
                .and_then(|v| v.parse().ok())
                .ok_or_else(|| format!("bad plan line: {l}"))?;
            let len = it
                .next()
                .and_then(|v| v.parse().ok())
                .ok_or_else(|| format!("bad plan line: {l}"))?;
            Ok(Read { offset, len })
        })
        .collect()
}

#[cfg(target_os = "linux")]
pub mod uring {
    //! The fetch engine. One ring, `qd` reads in flight, O_DIRECT with
    //! aligned brackets (same trick as ds4's cuda_model_stage_read: align
    //! the offset down and the length up; the payload is a slice of the
    //! bracket). Buffers are recycled through a free pool keyed by size.

    use super::Read;
    use io_uring::{opcode, types, IoUring};
    use std::os::fd::AsRawFd;

    /// Custom page-aligned allocator (e.g. CUDA pinned memory) injected by
    /// the engine; the stream crate itself stays CUDA-free.
    #[derive(Clone, Copy)]
    pub struct BufAlloc {
        pub alloc: fn(usize) -> *mut u8,
        pub free: fn(*mut u8, usize),
    }

    pub struct Aligned {
        ptr: *mut u8,
        cap: usize,
        custom_free: Option<fn(*mut u8, usize)>,
    }

    unsafe impl Send for Aligned {}

    impl Aligned {
        pub fn ptr(&self) -> *mut u8 {
            self.ptr
        }

        pub fn cap(&self) -> usize {
            self.cap
        }

        pub fn new(cap: usize, align: usize) -> Option<Self> {
            let layout = std::alloc::Layout::from_size_align(cap, align).ok()?;
            let ptr = unsafe { std::alloc::alloc(layout) };
            if ptr.is_null() {
                return None;
            }
            Some(Self { ptr, cap, custom_free: None })
        }

        /// Allocate via `a`, falling back to the default allocator when it
        /// returns null (e.g. pinned memory exhausted).
        pub fn new_with(cap: usize, align: usize, a: Option<BufAlloc>) -> Option<Self> {
            if let Some(a) = a {
                let ptr = (a.alloc)(cap);
                if !ptr.is_null() {
                    return Some(Self { ptr, cap, custom_free: Some(a.free) });
                }
            }
            Self::new(cap, align)
        }
    }

    impl Drop for Aligned {
        fn drop(&mut self) {
            match self.custom_free {
                Some(free) => free(self.ptr, self.cap),
                None => {
                    let layout =
                        std::alloc::Layout::from_size_align(self.cap, 4096).unwrap();
                    unsafe { std::alloc::dealloc(self.ptr, layout) }
                }
            }
        }
    }

    pub struct Stats {
        pub bytes_payload: u64,
        pub bytes_disk: u64,
        pub reads: u64,
        pub secs: f64,
        /// xor of one byte per read, so the compiler cannot elide the I/O
        /// and we can cross-check the two implementations touched the
        /// same data.
        pub checksum: u8,
    }

    struct InFlight {
        buf: Aligned,
        payload_off: usize,
        payload_len: usize,
    }

    /// Run the plan at queue depth `qd`. Returns throughput stats.
    pub fn run_plan(
        file: &std::fs::File,
        reads: &[Read],
        qd: usize,
        align: u64,
    ) -> std::io::Result<Stats> {
        let mut ring = IoUring::new(qd as u32 * 2)?;
        let fd = types::Fd(file.as_raw_fd());
        let t0 = std::time::Instant::now();
        let mut stats = Stats {
            bytes_payload: 0,
            bytes_disk: 0,
            reads: 0,
            secs: 0.0,
            checksum: 0,
        };
        let mut slots: Vec<Option<InFlight>> = Vec::new();
        for _ in 0..qd {
            slots.push(None);
        }
        let mut next = 0usize;
        let mut inflight = 0usize;

        loop {
            // fill the ring
            while inflight < qd && next < reads.len() {
                let r = reads[next];
                let aligned_off = r.offset & !(align - 1);
                let payload_off = (r.offset - aligned_off) as usize;
                let disk_len =
                    ((payload_off as u64 + r.len + align - 1) / align) * align;
                let slot = slots
                    .iter()
                    .position(|s| s.is_none())
                    .expect("free slot when inflight < qd");
                // +256: integer-dot kernels may read phantom tail sub-blocks
                // past a row that ends at the slab edge - up to 7 x 34B for
                // q8_0 (see SLAB_SLACK in engine)
                let buf = Aligned::new(disk_len as usize + 256, align as usize)
                    .ok_or_else(|| {
                        std::io::Error::from(std::io::ErrorKind::OutOfMemory)
                    })?;
                let sqe = opcode::Read::new(fd, buf.ptr, disk_len as u32)
                    .offset(aligned_off)
                    .build()
                    .user_data(slot as u64);
                slots[slot] = Some(InFlight {
                    buf,
                    payload_off,
                    payload_len: r.len as usize,
                });
                unsafe { ring.submission().push(&sqe).expect("sq room") };
                stats.bytes_disk += disk_len;
                inflight += 1;
                next += 1;
            }
            if inflight == 0 {
                break;
            }
            ring.submit_and_wait(1)?;
            let completions: Vec<(u64, i32)> =
                ring.completion().map(|c| (c.user_data(), c.result())).collect();
            for (ud, res) in completions {
                let slot = ud as usize;
                let inf = slots[slot].take().expect("slot occupied");
                inflight -= 1;
                if res < 0 {
                    return Err(std::io::Error::from_raw_os_error(-res));
                }
                stats.reads += 1;
                stats.bytes_payload += inf.payload_len as u64;
                // touch the payload so the read cannot be optimized away
                stats.checksum ^= unsafe {
                    *inf.buf.ptr.add(inf.payload_off + inf.payload_len / 2)
                };
            }
        }
        stats.secs = t0.elapsed().as_secs_f64();
        Ok(stats)
    }
}

#[cfg(target_os = "linux")]
pub mod fetch {
    //! Reusable batch fetcher over the same ring/O_DIRECT machinery as the
    //! bench: submit a batch of slab reads, get owned payloads back.

    use super::uring::Aligned;
    use super::Read;
    use io_uring::{opcode, types, IoUring};
    use std::os::fd::AsRawFd;
    use std::os::unix::fs::OpenOptionsExt;

    const ALIGN: u64 = 4096;

    /// One fetched slab: an aligned bracket plus the payload window.
    pub struct Slab {
        buf: Aligned,
        payload_off: usize,
        payload_len: usize,
    }

    impl Slab {
        pub fn payload(&self) -> &[u8] {
            unsafe {
                std::slice::from_raw_parts(self.buf.ptr().add(self.payload_off), self.payload_len)
            }
        }

        pub fn bytes(&self) -> usize {
            self.buf.cap()
        }
    }

    pub struct Fetcher {
        ring: IoUring,
        /// (virtual base, file), sorted by base. Single-file = one entry
        /// at base 0; split ggufs treat the shard set as one virtual file
        /// and route each read by offset (tensors never straddle shards).
        files: Vec<(u64, std::fs::File)>,
        qd: usize,
        buf_alloc: Option<super::uring::BufAlloc>,
    }

    impl Fetcher {
        pub fn open(path: &std::path::Path, qd: usize) -> std::io::Result<Fetcher> {
            Self::open_with(path, qd, None)
        }

        /// `buf_alloc` supplies the fetch buffers (e.g. CUDA pinned memory
        /// so later H2D copies run at full PCIe rate); buffers outlive the
        /// fetch as cache slabs, so allocate accordingly.
        pub fn open_with(
            path: &std::path::Path,
            qd: usize,
            buf_alloc: Option<super::uring::BufAlloc>,
        ) -> std::io::Result<Fetcher> {
            Self::open_split(std::slice::from_ref(&(0, path.to_path_buf())), qd, buf_alloc)
        }

        /// Open a virtual file spanning shards: `shards[i]` = (virtual
        /// base offset, path). Bases must be ascending.
        pub fn open_split(
            shards: &[(u64, std::path::PathBuf)],
            qd: usize,
            buf_alloc: Option<super::uring::BufAlloc>,
        ) -> std::io::Result<Fetcher> {
            let mut files = Vec::with_capacity(shards.len());
            for (base, path) in shards {
                let file = std::fs::OpenOptions::new()
                    .read(true)
                    .custom_flags(libc::O_DIRECT)
                    .open(path)?;
                files.push((*base, file));
            }
            Ok(Fetcher { ring: IoUring::new(qd as u32 * 2)?, files, qd, buf_alloc })
        }

        /// (fd, local offset) for a virtual offset.
        fn route(&self, offset: u64) -> (types::Fd, u64) {
            let i = match self.files.binary_search_by(|(b, _)| b.cmp(&offset)) {
                Ok(i) => i,
                Err(i) => i.saturating_sub(1),
            };
            (types::Fd(self.files[i].1.as_raw_fd()), offset - self.files[i].0)
        }

        /// Fetch every read; result[i] corresponds to reads[i].
        pub fn fetch(&mut self, reads: &[Read]) -> std::io::Result<Vec<Slab>> {
            let mut out: Vec<Option<Slab>> = Vec::with_capacity(reads.len());
            out.resize_with(reads.len(), || None);
            self.fetch_each(reads, |i, slab| {
                out[i] = Some(slab);
                Ok(())
            })?;
            Ok(out.into_iter().map(|s| s.expect("all fetched")).collect())
        }

        /// Fetch every read, handing each slab to `on_slab(index, slab)` as
        /// its completion lands - so the caller's processing (e.g. H2D
        /// upload) overlaps the remaining disk reads.
        pub fn fetch_each(
            &mut self,
            reads: &[Read],
            mut on_slab: impl FnMut(usize, Slab) -> std::io::Result<()>,
        ) -> std::io::Result<()> {
            let mut pending: Vec<Option<Slab>> = Vec::with_capacity(reads.len());
            pending.resize_with(reads.len(), || None);
            let mut next = 0usize;
            let mut inflight = 0usize;

            loop {
                while inflight < self.qd && next < reads.len() {
                    let r = reads[next];
                    let (fd, local) = self.route(r.offset);
                    let aligned_off = local & !(ALIGN - 1);
                    let payload_off = (local - aligned_off) as usize;
                    let disk_len = (payload_off as u64 + r.len).next_multiple_of(ALIGN);
                    // +256: phantom-tail slack, same invariant as the fetch path
                    let buf = Aligned::new_with(disk_len as usize + 256, ALIGN as usize, self.buf_alloc)
                        .ok_or_else(|| std::io::Error::from(std::io::ErrorKind::OutOfMemory))?;
                    let sqe = opcode::Read::new(fd, buf.ptr(), disk_len as u32)
                        .offset(aligned_off)
                        .build()
                        .user_data(next as u64);
                    pending[next] = Some(Slab {
                        buf,
                        payload_off,
                        payload_len: r.len as usize,
                    });
                    unsafe { self.ring.submission().push(&sqe).expect("sq room") };
                    inflight += 1;
                    next += 1;
                }
                if inflight == 0 {
                    break;
                }
                self.ring.submit_and_wait(1)?;
                let completions: Vec<(u64, i32)> = self
                    .ring
                    .completion()
                    .map(|c| (c.user_data(), c.result()))
                    .collect();
                for (ud, res) in completions {
                    if res < 0 {
                        return Err(std::io::Error::from_raw_os_error(-res));
                    }
                    inflight -= 1;
                    let idx = ud as usize;
                    let slab = pending[idx].take().expect("slot occupied");
                    on_slab(idx, slab)?;
                }
            }
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_roundtrip() {
        let reads = vec![
            Read { offset: 4096, len: 1536 },
            Read { offset: 1 << 33, len: 4718592 },
        ];
        assert_eq!(plan_from_str(&plan_to_string(&reads)).unwrap(), reads);
    }
}

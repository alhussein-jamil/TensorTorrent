//! Production streaming parameter cache: pread + shared inflight + LRU bytes.

use crate::cache::ChunkCache;
use crate::error::{StorageError, StorageResult};
use crate::pack::PackReader;
use parking_lot::{Condvar, Mutex};
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};

struct Shared {
    reader: Mutex<PackReader>,
    cache: ChunkCache,
    state: Mutex<State>,
    cv: Condvar,
    closed: AtomicBool,
    prefetch_hits: AtomicU64,
    waits: AtomicU64,
    bytes_read: AtomicU64,
    prefetch_submitted: AtomicU64,
    acquire_submitted: AtomicU64,
    last_error: Mutex<Option<String>>,
    origin: Mutex<std::time::Instant>,
    io_intervals: Mutex<Vec<(f64, f64, u64)>>,
}

struct State {
    queue: VecDeque<String>,
    inflight: HashSet<String>,
    waiters: HashMap<String, Arc<(Mutex<bool>, Condvar)>>,
}

/// Authoritative native streaming store for pack-backed parameters.
pub struct StreamingStore {
    shared: Arc<Shared>,
    worker: Mutex<Option<JoinHandle<()>>>,
}

impl StreamingStore {
    pub fn open(
        path: impl Into<PathBuf>,
        manifest_json: &str,
        capacity_bytes: u64,
    ) -> StorageResult<Self> {
        let manifest = crate::pack::PackManifest::from_json(manifest_json)?;
        let reader = PackReader::open(path.into(), manifest)?;
        Ok(Self {
            shared: Arc::new(Shared {
                reader: Mutex::new(reader),
                cache: ChunkCache::new(capacity_bytes.max(1)),
                state: Mutex::new(State {
                    queue: VecDeque::new(),
                    inflight: HashSet::new(),
                    waiters: HashMap::new(),
                }),
                cv: Condvar::new(),
                closed: AtomicBool::new(false),
                prefetch_hits: AtomicU64::new(0),
                waits: AtomicU64::new(0),
                bytes_read: AtomicU64::new(0),
                prefetch_submitted: AtomicU64::new(0),
                acquire_submitted: AtomicU64::new(0),
                last_error: Mutex::new(None),
                origin: Mutex::new(std::time::Instant::now()),
                io_intervals: Mutex::new(Vec::new()),
            }),
            worker: Mutex::new(None),
        })
    }

    /// Reset I/O timing origin (call at forward start so intervals align with Python clocks).
    pub fn reset_io_origin(&self) {
        *self.shared.origin.lock() = std::time::Instant::now();
        self.shared.io_intervals.lock().clear();
    }

    /// Timed pread windows relative to the last [`reset_io_origin`] (seconds).
    pub fn io_intervals(&self) -> Vec<(f64, f64, u64)> {
        self.shared.io_intervals.lock().clone()
    }

    /// Queue keys for background load (deduped; shares inflight).
    pub fn prefetch(&self, keys: &[String]) {
        if self.shared.closed.load(Ordering::Acquire) {
            return;
        }
        let mut queued = 0u64;
        {
            let mut st = self.shared.state.lock();
            for key in keys {
                if self.shared.cache.get(key).is_some() {
                    self.shared.cache.release(key);
                    self.shared.prefetch_hits.fetch_add(1, Ordering::Relaxed);
                    continue;
                }
                if st.inflight.contains(key) || st.queue.iter().any(|k| k == key) {
                    continue;
                }
                st.inflight.insert(key.clone());
                st.waiters
                    .entry(key.clone())
                    .or_insert_with(|| Arc::new((Mutex::new(false), Condvar::new())));
                st.queue.push_back(key.clone());
                queued += 1;
            }
        }
        if queued == 0 {
            return;
        }
        self.shared
            .prefetch_submitted
            .fetch_add(queued, Ordering::Relaxed);
        if let Err(e) = self.ensure_worker() {
            *self.shared.last_error.lock() = Some(e.to_string());
            return;
        }
        self.shared.cv.notify_one();
    }

    /// Block until bytes are cached; returns Arc payload (caller must `release`).
    pub fn acquire_bytes(&self, key: &str) -> StorageResult<Arc<Vec<u8>>> {
        if let Some(data) = self.shared.cache.get(key) {
            return Ok(data);
        }
        let waiter = {
            let mut st = self.shared.state.lock();
            if let Some(data) = self.shared.cache.get(key) {
                return Ok(data);
            }
            if let Some(w) = st.waiters.get(key) {
                self.shared.waits.fetch_add(1, Ordering::Relaxed);
                Arc::clone(w)
            } else {
                st.inflight.insert(key.to_owned());
                let w = Arc::new((Mutex::new(false), Condvar::new()));
                st.waiters.insert(key.to_owned(), Arc::clone(&w));
                st.queue.push_front(key.to_owned());
                self.shared
                    .acquire_submitted
                    .fetch_add(1, Ordering::Relaxed);
                drop(st);
                self.ensure_worker()?;
                self.shared.cv.notify_one();
                w
            }
        };
        let (done_lock, done_cv) = (&waiter.0, &waiter.1);
        let mut done = done_lock.lock();
        while !*done {
            if self.shared.closed.load(Ordering::Acquire) {
                return Err(StorageError::Io("streaming store closed".into()));
            }
            done_cv.wait(&mut done);
        }
        drop(done);
        if let Some(data) = self.shared.cache.get(key) {
            return Ok(data);
        }
        let detail = self
            .shared
            .last_error
            .lock()
            .clone()
            .unwrap_or_else(|| format!("acquire missed after load: {key}"));
        Err(StorageError::Io(detail))
    }

    pub fn release(&self, key: &str) {
        self.shared.cache.release(key);
    }

    /// Pack entry metadata for a key (dtype/shape/length).
    pub fn entry(&self, key: &str) -> Option<crate::pack::TensorEntry> {
        let g = self.shared.reader.lock();
        g.manifest().tensors.iter().find(|t| t.name == key).cloned()
    }

    pub fn stats(&self) -> StreamingStats {
        let (hits, misses, live_bytes) = self.shared.cache.stats();
        StreamingStats {
            cache_hits: hits,
            cache_misses: misses,
            live_bytes,
            prefetch_hits: self.shared.prefetch_hits.load(Ordering::Relaxed),
            waits_for_prefetch: self.shared.waits.load(Ordering::Relaxed),
            bytes_read: self.shared.bytes_read.load(Ordering::Relaxed),
            prefetch_submitted: self.shared.prefetch_submitted.load(Ordering::Relaxed),
            native_streaming: true,
        }
    }

    pub fn close(&self) {
        self.shared.closed.store(true, Ordering::Release);
        self.shared.cv.notify_all();
        if let Some(h) = self.worker.lock().take() {
            let _ = h.join();
        }
        // Wake any acquire waiters so they observe closed/miss.
        let waiters: Vec<_> = {
            let mut st = self.shared.state.lock();
            st.queue.clear();
            st.inflight.clear();
            st.waiters.drain().map(|(_, w)| w).collect()
        };
        for w in waiters {
            let mut done = w.0.lock();
            *done = true;
            w.1.notify_all();
        }
    }

    fn ensure_worker(&self) -> StorageResult<()> {
        let mut slot = self.worker.lock();
        if slot.as_ref().is_some_and(|h| !h.is_finished()) {
            return Ok(());
        }
        let shared = Arc::clone(&self.shared);
        let handle = thread::Builder::new()
            .name("sc-native-prefetch".into())
            .spawn(move || worker_loop(shared))
            .map_err(|e| StorageError::Io(format!("spawn prefetch worker: {e}")))?;
        *slot = Some(handle);
        Ok(())
    }
}

fn worker_loop(shared: Arc<Shared>) {
    loop {
        let key = {
            let mut st = shared.state.lock();
            loop {
                if shared.closed.load(Ordering::Acquire) && st.queue.is_empty() {
                    return;
                }
                if let Some(k) = st.queue.pop_front() {
                    break k;
                }
                if shared.closed.load(Ordering::Acquire) {
                    return;
                }
                shared.cv.wait(&mut st);
            }
        };
        let loaded = {
            let t0 = shared.origin.lock().elapsed().as_secs_f64();
            let mut reader = shared.reader.lock();
            let result = reader.pread(&key);
            let t1 = shared.origin.lock().elapsed().as_secs_f64();
            (t0, t1, result)
        };
        match loaded {
            (t0, t1, Ok(bytes)) => {
                shared
                    .bytes_read
                    .fetch_add(bytes.len() as u64, Ordering::Relaxed);
                shared
                    .io_intervals
                    .lock()
                    .push((t0, t1, bytes.len() as u64));
                let _ = shared.cache.insert(key.clone(), bytes);
            }
            (_, _, Err(e)) => {
                *shared.last_error.lock() = Some(e.to_string());
            }
        }
        let waiter = {
            let mut st = shared.state.lock();
            st.inflight.remove(&key);
            st.waiters.remove(&key)
        };
        if let Some(w) = waiter {
            let mut done = w.0.lock();
            *done = true;
            w.1.notify_all();
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct StreamingStats {
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub live_bytes: u64,
    pub prefetch_hits: u64,
    pub waits_for_prefetch: u64,
    pub bytes_read: u64,
    pub prefetch_submitted: u64,
    pub native_streaming: bool,
}

impl Drop for StreamingStore {
    fn drop(&mut self) {
        self.close();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pack::{PackManifest, TensorEntry, PACK_FORMAT_VERSION};
    use std::fs::File;
    use std::io::Write;

    fn tiny_pack() -> (PathBuf, String) {
        // Unique dir: parallel tests must not share one pack file.
        let dir = std::env::temp_dir().join(format!(
            "sc-streaming-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("data.bin");
        let payload = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
        {
            let mut f = File::create(&path).unwrap();
            f.write_all(&payload).unwrap();
        }
        let manifest = PackManifest {
            version: PACK_FORMAT_VERSION,
            tensors: vec![TensorEntry {
                name: "w".into(),
                offset: 0,
                length: payload.len() as u64,
                dtype: "float32".into(),
                shape: vec![2, 1],
                checksum_crc32: None,
            }],
            notes: vec![],
        };
        (path, serde_json::to_string(&manifest).unwrap())
    }

    #[test]
    fn acquire_after_prefetch() {
        let (path, json) = tiny_pack();
        let store = StreamingStore::open(&path, &json, 1024).unwrap();
        store.prefetch(&[String::from("w")]);
        let bytes = store.acquire_bytes("w").unwrap();
        assert_eq!(bytes.len(), 8);
        store.release("w");
        store.close();
    }

    #[test]
    fn acquire_loads_synchronously() {
        let (path, json) = tiny_pack();
        let store = StreamingStore::open(&path, &json, 1024).unwrap();
        let bytes = store.acquire_bytes("w").unwrap();
        assert_eq!(&bytes[..], &[1, 2, 3, 4, 5, 6, 7, 8]);
        store.release("w");
        let stats = store.stats();
        assert!(stats.bytes_read >= 8);
        assert!(stats.native_streaming);
        store.close();
    }
}

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

const DEFAULT_PREFETCH_QUEUE: usize = 4096;
const DEFAULT_IO_WORKERS: usize = 2;
const MAX_IO_WORKERS: usize = 64;

struct Shared {
    readers: Vec<Mutex<PackReader>>,
    entries: HashMap<String, crate::pack::TensorEntry>,
    cache: ChunkCache,
    state: Mutex<State>,
    cv: Condvar,
    closed: AtomicBool,
    prefetch_hits: AtomicU64,
    waits: AtomicU64,
    bytes_read: AtomicU64,
    prefetch_submitted: AtomicU64,
    prefetch_dropped: AtomicU64,
    acquire_submitted: AtomicU64,
    origin: Mutex<std::time::Instant>,
    io_intervals: Mutex<Vec<(f64, f64, u64)>>,
    queue_limit: usize,
    worker_count: usize,
}

struct State {
    queue: VecDeque<String>,
    inflight: HashSet<String>,
    waiters: HashMap<String, Arc<(Mutex<bool>, Condvar)>>,
    errors: HashMap<String, String>,
}

/// Authoritative native streaming store for pack-backed parameters.
pub struct StreamingStore {
    shared: Arc<Shared>,
    workers: Mutex<Vec<JoinHandle<()>>>,
}

impl StreamingStore {
    pub fn open(
        path: impl Into<PathBuf>,
        manifest_json: &str,
        capacity_bytes: u64,
    ) -> StorageResult<Self> {
        Self::open_with_options(
            path,
            manifest_json,
            capacity_bytes,
            DEFAULT_IO_WORKERS,
            DEFAULT_PREFETCH_QUEUE,
        )
    }

    pub fn open_with_options(
        path: impl Into<PathBuf>,
        manifest_json: &str,
        capacity_bytes: u64,
        io_workers: usize,
        queue_limit: usize,
    ) -> StorageResult<Self> {
        if io_workers == 0 || io_workers > MAX_IO_WORKERS {
            return Err(StorageError::Invalid(format!(
                "io_workers must be in 1..={MAX_IO_WORKERS}, got {io_workers}"
            )));
        }
        if queue_limit == 0 {
            return Err(StorageError::Invalid("queue_limit must be >= 1".into()));
        }
        let manifest = crate::pack::PackManifest::from_json(manifest_json)?;
        let path = path.into();
        let entries = manifest
            .tensors
            .iter()
            .cloned()
            .map(|entry| (entry.name.clone(), entry))
            .collect();
        let manifest = std::sync::Arc::new(manifest);
        let mut readers = Vec::with_capacity(io_workers);
        for _ in 0..io_workers {
            readers.push(Mutex::new(PackReader::open(&path, manifest.clone())?));
        }
        Ok(Self {
            shared: Arc::new(Shared {
                readers,
                entries,
                cache: ChunkCache::new(capacity_bytes.max(1)),
                state: Mutex::new(State {
                    queue: VecDeque::new(),
                    inflight: HashSet::new(),
                    waiters: HashMap::new(),
                    errors: HashMap::new(),
                }),
                cv: Condvar::new(),
                closed: AtomicBool::new(false),
                prefetch_hits: AtomicU64::new(0),
                waits: AtomicU64::new(0),
                bytes_read: AtomicU64::new(0),
                prefetch_submitted: AtomicU64::new(0),
                prefetch_dropped: AtomicU64::new(0),
                acquire_submitted: AtomicU64::new(0),
                origin: Mutex::new(std::time::Instant::now()),
                io_intervals: Mutex::new(Vec::new()),
                queue_limit,
                worker_count: io_workers,
            }),
            workers: Mutex::new(Vec::new()),
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
            for (index, key) in keys.iter().enumerate() {
                if st.queue.len() >= self.shared.queue_limit {
                    self.shared
                        .prefetch_dropped
                        .fetch_add((keys.len() - index) as u64, Ordering::Relaxed);
                    break;
                }
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
        if let Err(e) = self.ensure_workers() {
            fail_pending(&self.shared, e.to_string());
            return;
        }
        self.shared.cv.notify_all();
    }

    /// Block until bytes are cached; returns Arc payload (caller must `release`).
    pub fn acquire_bytes(&self, key: &str) -> StorageResult<Arc<Vec<u8>>> {
        loop {
            if self.shared.closed.load(Ordering::Acquire) {
                return Err(StorageError::Io("streaming store closed".into()));
            }
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
                    self.ensure_workers()?;
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
            if let Some(detail) = self.shared.state.lock().errors.remove(key) {
                return Err(StorageError::Io(detail));
            }
            // A later queued insert may evict this unleased chunk between the
            // worker's cache insert and our wakeup. Requeue it instead of
            // reporting a false I/O failure. The bounded worker pool will
            // eventually service the requeued key once prior reads complete.
        }
    }

    pub fn release(&self, key: &str) {
        self.shared.cache.release(key);
    }

    /// Pack entry metadata for a key (dtype/shape/length).
    pub fn entry(&self, key: &str) -> Option<crate::pack::TensorEntry> {
        self.shared.entries.get(key).cloned()
    }

    pub fn stats(&self) -> StreamingStats {
        let (hits, misses, live_bytes) = self.shared.cache.stats();
        let state = self.shared.state.lock();
        StreamingStats {
            cache_hits: hits,
            cache_misses: misses,
            live_bytes,
            prefetch_hits: self.shared.prefetch_hits.load(Ordering::Relaxed),
            waits_for_prefetch: self.shared.waits.load(Ordering::Relaxed),
            bytes_read: self.shared.bytes_read.load(Ordering::Relaxed),
            prefetch_submitted: self.shared.prefetch_submitted.load(Ordering::Relaxed),
            prefetch_dropped: self.shared.prefetch_dropped.load(Ordering::Relaxed),
            io_workers: self.shared.worker_count as u64,
            queue_depth: state.queue.len() as u64,
            inflight_reads: state.inflight.len() as u64,
            native_streaming: true,
        }
    }

    pub fn close(&self) {
        if self.shared.closed.swap(true, Ordering::AcqRel) {
            return;
        }
        // Drop queued hints before joining. Otherwise close could wait for a large
        // speculative queue to drain even though no caller can consume the data.
        // Active positional reads are allowed to finish, then workers observe the
        // closed flag and exit.
        let waiters: Vec<_> = {
            let mut st = self.shared.state.lock();
            st.queue.clear();
            st.inflight.clear();
            st.waiters.drain().map(|(_, waiter)| waiter).collect()
        };
        self.shared.cv.notify_all();
        for waiter in waiters {
            let mut done = waiter.0.lock();
            *done = true;
            waiter.1.notify_all();
        }
        let handles = std::mem::take(&mut *self.workers.lock());
        for handle in handles {
            let _ = handle.join();
        }
    }

    fn ensure_workers(&self) -> StorageResult<()> {
        let mut workers = self.workers.lock();
        if workers.iter().any(|handle| handle.is_finished()) {
            return Err(StorageError::Io(
                "a native prefetch worker terminated unexpectedly".into(),
            ));
        }
        while workers.len() < self.shared.worker_count {
            let worker_id = workers.len();
            let shared = Arc::clone(&self.shared);
            let handle = thread::Builder::new()
                .name(format!("tt-native-prefetch-{worker_id}"))
                .spawn(move || worker_loop(shared, worker_id))
                .map_err(|e| StorageError::Io(format!("spawn prefetch worker {worker_id}: {e}")))?;
            workers.push(handle);
        }
        Ok(())
    }
}

fn fail_pending(shared: &Shared, detail: String) {
    let waiters: Vec<_> = {
        let mut st = shared.state.lock();
        st.queue.clear();
        st.inflight.clear();
        let keys: Vec<String> = st.waiters.keys().cloned().collect();
        for key in keys {
            st.errors.insert(key, detail.clone());
        }
        st.waiters.drain().map(|(_, waiter)| waiter).collect()
    };
    for waiter in waiters {
        let mut done = waiter.0.lock();
        *done = true;
        waiter.1.notify_all();
    }
}

fn worker_loop(shared: Arc<Shared>, worker_id: usize) {
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
            let mut reader = shared.readers[worker_id].lock();
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
                shared.state.lock().errors.remove(&key);
            }
            (_, _, Err(e)) => {
                shared
                    .state
                    .lock()
                    .errors
                    .insert(key.clone(), e.to_string());
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
    pub prefetch_dropped: u64,
    pub io_workers: u64,
    pub queue_depth: u64,
    pub inflight_reads: u64,
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
            "tt-streaming-test-{}-{}",
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

    #[test]
    fn acquire_after_close_fails_without_spawning_worker() {
        let (path, json) = tiny_pack();
        let store = StreamingStore::open(&path, &json, 1024).unwrap();
        store.close();
        let error = store.acquire_bytes("w").unwrap_err();
        assert!(error.to_string().contains("closed"));
        assert!(store.workers.lock().is_empty());
    }

    #[test]
    fn load_errors_are_scoped_to_the_requested_key() {
        let (path, json) = tiny_pack();
        let store = StreamingStore::open(&path, &json, 1024).unwrap();
        assert!(store.acquire_bytes("missing").is_err());
        let bytes = store.acquire_bytes("w").unwrap();
        assert_eq!(bytes.len(), 8);
        store.release("w");
        store.close();
    }

    #[test]
    fn saturated_prefetch_queue_reports_dropped_requests() {
        let (path, json) = tiny_pack();
        let store = StreamingStore::open_with_options(&path, &json, 1024, 2, 4).unwrap();
        let keys = (0..=4)
            .map(|index| format!("missing-{index}"))
            .collect::<Vec<_>>();
        store.prefetch(&keys);
        let stats = store.stats();
        assert_eq!(stats.prefetch_submitted, 4);
        assert_eq!(stats.prefetch_dropped, 1);
        assert_eq!(stats.io_workers, 2);
        store.close();
    }

    #[test]
    fn invalid_worker_or_queue_configuration_is_rejected() {
        let (path, json) = tiny_pack();
        assert!(StreamingStore::open_with_options(&path, &json, 1024, 0, 4).is_err());
        assert!(StreamingStore::open_with_options(&path, &json, 1024, 1, 0).is_err());
    }
}

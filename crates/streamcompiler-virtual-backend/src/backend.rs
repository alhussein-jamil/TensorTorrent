//! Deterministic simulated accelerator with async pending events.
//!
//! Submission returns `Pending` immediately. Background stream workers complete
//! events after configured delays. All results are labelled `simulated=true`.

use parking_lot::{Condvar, Mutex};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};
use streamcompiler_backend_api::{
    Backend, BackendCapabilities, BackendError, BackendResult, BufferHandle, EventHandle,
    EventStatus, ExecutableHandle,
};
use streamcompiler_core::{ResourceId, StreamId};

#[derive(Clone, Debug)]
pub struct VirtualBackendConfig {
    pub name: String,
    pub memory_bytes: u64,
    pub compute_delay_s: f64,
    pub transfer_bandwidth_bytes_per_s: f64,
    pub transfer_latency_s: f64,
    pub max_copy_engines: u32,
    pub supports_p2p: bool,
}

impl Default for VirtualBackendConfig {
    fn default() -> Self {
        Self {
            name: "mock_accel0".into(),
            memory_bytes: 8 * 1024 * 1024 * 1024,
            compute_delay_s: 0.05,
            transfer_bandwidth_bytes_per_s: 12e9,
            transfer_latency_s: 1e-5,
            max_copy_engines: 2,
            supports_p2p: false,
        }
    }
}

struct BufferRec {
    #[allow(dead_code)]
    resource: String,
    bytes: usize,
    /// Distinct virtual-device storage — not a host alias.
    #[allow(dead_code)]
    payload: Vec<u8>,
}

struct EventRec {
    status: EventStatus,
}

enum JobKind {
    Compute,
    Transfer {
        #[allow(dead_code)]
        src: u64,
        #[allow(dead_code)]
        dst: u64,
        #[allow(dead_code)]
        bytes: usize,
    },
}

struct Job {
    event_id: u64,
    delay_s: f64,
    kind: JobKind,
}

struct StreamWorker {
    tx: crossbeam_channel::Sender<Job>,
    handle: JoinHandle<()>,
}

/// Deterministic simulated accelerator with ordered compute/transfer streams.
pub struct VirtualBackend {
    config: VirtualBackendConfig,
    next_buf: AtomicU64,
    next_evt: AtomicU64,
    buffers: Mutex<HashMap<u64, BufferRec>>,
    events: Arc<Mutex<HashMap<u64, EventRec>>>,
    event_cv: Arc<Condvar>,
    used_bytes: Mutex<u64>,
    workers: Mutex<HashMap<String, StreamWorker>>,
    shutdown: Arc<AtomicBool>,
    origin: Instant,
}

impl VirtualBackend {
    #[must_use]
    pub fn new(config: VirtualBackendConfig) -> Self {
        Self {
            config,
            next_buf: AtomicU64::new(1),
            next_evt: AtomicU64::new(1),
            buffers: Mutex::new(HashMap::new()),
            events: Arc::new(Mutex::new(HashMap::new())),
            event_cv: Arc::new(Condvar::new()),
            used_bytes: Mutex::new(0),
            workers: Mutex::new(HashMap::new()),
            shutdown: Arc::new(AtomicBool::new(false)),
            origin: Instant::now(),
        }
    }

    fn ensure_worker(&self, stream: &str) -> crossbeam_channel::Sender<Job> {
        let mut workers = self.workers.lock();
        if let Some(w) = workers.get(stream) {
            return w.tx.clone();
        }
        let (tx, rx) = crossbeam_channel::unbounded::<Job>();
        let events = Arc::clone(&self.events);
        let event_cv = Arc::clone(&self.event_cv);
        let shutdown = Arc::clone(&self.shutdown);
        let stream_name = stream.to_owned();
        let handle = thread::Builder::new()
            .name(format!("virt-{}", stream_name))
            .spawn(move || {
                // Ordered stream: process jobs FIFO; sleep per job on this worker only.
                let mut queue: VecDeque<Job> = VecDeque::new();
                loop {
                    if shutdown.load(Ordering::Acquire) && queue.is_empty() {
                        while let Ok(job) = rx.try_recv() {
                            queue.push_back(job);
                        }
                        if queue.is_empty() {
                            break;
                        }
                    }
                    if queue.is_empty() {
                        match rx.recv_timeout(Duration::from_millis(50)) {
                            Ok(job) => queue.push_back(job),
                            Err(crossbeam_channel::RecvTimeoutError::Timeout) => continue,
                            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
                        }
                    }
                    while let Ok(job) = rx.try_recv() {
                        queue.push_back(job);
                    }
                    let Some(job) = queue.pop_front() else {
                        continue;
                    };
                    let _ = &job.kind; // payload copies reserved for future device buffers
                    if job.delay_s > 0.0 {
                        let nanos = (job.delay_s * 1e9) as u64;
                        thread::sleep(Duration::from_nanos(nanos.max(1)));
                    }
                    {
                        let mut ev = events.lock();
                        if let Some(rec) = ev.get_mut(&job.event_id) {
                            rec.status = EventStatus::Complete;
                        }
                    }
                    event_cv.notify_all();
                }
            })
            .expect("spawn virtual stream worker");
        let tx_clone = tx.clone();
        workers.insert(stream_name, StreamWorker { tx, handle });
        tx_clone
    }

    /// Simulated compute on an ordered stream: pending event → worker sleep → wait.
    pub fn run_compute(&self, stream: &str, delay_s: f64) -> BackendResult<()> {
        let ev = self.submit_job(stream, delay_s.max(0.0), JobKind::Compute)?;
        self.wait_event(ev)
    }

    /// Simulated transfer using native buffers + capacity checks + pending event.
    ///
    /// Host staging is virtual device buffers (not host aliases). Results are simulated.
    pub fn run_transfer(
        &self,
        stream: &str,
        bytes: usize,
        delay_s: Option<f64>,
    ) -> BackendResult<()> {
        let n = bytes.max(1);
        let resource = ResourceId::new(&self.config.name);
        let src = self.allocate(resource.clone(), n, 64)?;
        let dst = self.allocate(resource, n, 64)?;
        let delay = delay_s.unwrap_or_else(|| {
            self.config.transfer_latency_s
                + (n as f64) / self.config.transfer_bandwidth_bytes_per_s.max(1.0)
        });
        let result = self
            .submit_job(
                stream,
                delay.max(0.0),
                JobKind::Transfer {
                    src: src.0,
                    dst: dst.0,
                    bytes: n,
                },
            )
            .and_then(|ev| self.wait_event(ev));
        let _ = self.free(src);
        let _ = self.free(dst);
        result
    }

    fn submit_job(&self, stream: &str, delay_s: f64, kind: JobKind) -> BackendResult<EventHandle> {
        if self.shutdown.load(Ordering::Acquire) {
            return Err(BackendError::Other {
                backend: self.config.name.clone(),
                cause: "virtual backend shut down".into(),
            });
        }
        let id = self.next_evt.fetch_add(1, Ordering::Relaxed);
        self.events.lock().insert(
            id,
            EventRec {
                status: EventStatus::Pending,
            },
        );
        let tx = self.ensure_worker(stream);
        tx.send(Job {
            event_id: id,
            delay_s,
            kind,
        })
        .map_err(|_| BackendError::Other {
            backend: self.config.name.clone(),
            cause: "virtual stream worker closed".into(),
        })?;
        Ok(EventHandle(id))
    }

    /// Monotonic simulated time since backend creation (diagnostics).
    #[must_use]
    pub fn elapsed_s(&self) -> f64 {
        self.origin.elapsed().as_secs_f64()
    }

    /// Write host bytes into a virtual device buffer (not a host alias).
    pub fn write_bytes(&self, buffer: BufferHandle, data: &[u8]) -> BackendResult<()> {
        let mut buffers = self.buffers.lock();
        let Some(rec) = buffers.get_mut(&buffer.0) else {
            return Err(BackendError::Other {
                backend: self.config.name.clone(),
                cause: format!("unknown buffer {}", buffer.0),
            });
        };
        if data.len() > rec.payload.len() {
            return Err(BackendError::Other {
                backend: self.config.name.clone(),
                cause: format!(
                    "write {} bytes exceeds buffer capacity {}",
                    data.len(),
                    rec.payload.len()
                ),
            });
        }
        rec.payload[..data.len()].copy_from_slice(data);
        if data.len() < rec.payload.len() {
            rec.payload[data.len()..].fill(0);
        }
        Ok(())
    }

    /// Read virtual device buffer bytes into a host Vec.
    pub fn read_bytes(&self, buffer: BufferHandle) -> BackendResult<Vec<u8>> {
        let buffers = self.buffers.lock();
        let Some(rec) = buffers.get(&buffer.0) else {
            return Err(BackendError::Other {
                backend: self.config.name.clone(),
                cause: format!("unknown buffer {}", buffer.0),
            });
        };
        Ok(rec.payload.clone())
    }
}

impl Drop for VirtualBackend {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Release);
        let mut workers = self.workers.lock();
        let taken: Vec<StreamWorker> = workers.drain().map(|(_, w)| w).collect();
        drop(workers);
        for w in taken {
            drop(w.tx);
            let _ = w.handle.join();
        }
    }
}

impl Backend for VirtualBackend {
    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            name: self.config.name.clone(),
            supports_p2p: self.config.supports_p2p,
            supports_async_compute: true,
            supports_ordered_streams: true,
            max_streams: 8.max(self.config.max_copy_engines + 2),
            device_memory_bytes: self.config.memory_bytes,
            simulated: true,
        }
    }

    fn allocate(
        &self,
        resource: ResourceId,
        bytes: usize,
        _alignment: usize,
    ) -> BackendResult<BufferHandle> {
        let mut used = self.used_bytes.lock();
        if used.saturating_add(bytes as u64) > self.config.memory_bytes {
            return Err(BackendError::Allocate {
                backend: self.config.name.clone(),
                resource: resource.to_string(),
                cause: format!(
                    "capacity exceeded: need {bytes}, free {}",
                    self.config.memory_bytes.saturating_sub(*used)
                ),
            });
        }
        *used += bytes as u64;
        let id = self.next_buf.fetch_add(1, Ordering::Relaxed);
        self.buffers.lock().insert(
            id,
            BufferRec {
                resource: resource.to_string(),
                bytes,
                // Payload must match capacity — write_bytes/read_bytes use real storage.
                payload: vec![0u8; bytes],
            },
        );
        Ok(BufferHandle(id))
    }

    fn free(&self, buffer: BufferHandle) -> BackendResult<()> {
        let mut buffers = self.buffers.lock();
        let Some(rec) = buffers.remove(&buffer.0) else {
            return Err(BackendError::Other {
                backend: self.config.name.clone(),
                cause: format!("unknown buffer {}", buffer.0),
            });
        };
        let mut used = self.used_bytes.lock();
        *used = used.saturating_sub(rec.bytes as u64);
        Ok(())
    }

    fn transfer(
        &self,
        src: BufferHandle,
        dst: BufferHandle,
        bytes: usize,
        stream: StreamId,
    ) -> BackendResult<EventHandle> {
        {
            let buffers = self.buffers.lock();
            if !buffers.contains_key(&src.0) || !buffers.contains_key(&dst.0) {
                return Err(BackendError::Transfer {
                    backend: self.config.name.clone(),
                    cause: "src or dst buffer missing (virtual buffers are not host aliases)"
                        .into(),
                });
            }
        }
        let dur = self.config.transfer_latency_s
            + (bytes as f64) / self.config.transfer_bandwidth_bytes_per_s.max(1.0);
        // Pending immediately — worker sleeps.
        self.submit_job(
            stream.as_str(),
            dur,
            JobKind::Transfer {
                src: src.0,
                dst: dst.0,
                bytes,
            },
        )
    }

    fn launch(
        &self,
        _executable: ExecutableHandle,
        _inputs: &[BufferHandle],
        _outputs: &[BufferHandle],
        stream: StreamId,
    ) -> BackendResult<EventHandle> {
        self.submit_job(
            stream.as_str(),
            self.config.compute_delay_s,
            JobKind::Compute,
        )
    }

    fn query_event(&self, event: EventHandle) -> BackendResult<EventStatus> {
        self.events
            .lock()
            .get(&event.0)
            .map(|e| e.status)
            .ok_or(BackendError::Event {
                backend: self.config.name.clone(),
                event: event.0,
                cause: "unknown event".into(),
            })
    }

    fn wait_event(&self, event: EventHandle) -> BackendResult<()> {
        let mut guard = self.events.lock();
        loop {
            let status = guard
                .get(&event.0)
                .map(|e| e.status)
                .ok_or(BackendError::Event {
                    backend: self.config.name.clone(),
                    event: event.0,
                    cause: "unknown event".into(),
                })?;
            match status {
                EventStatus::Complete => return Ok(()),
                EventStatus::Error => {
                    return Err(BackendError::Event {
                        backend: self.config.name.clone(),
                        event: event.0,
                        cause: "event in error state".into(),
                    })
                }
                EventStatus::Pending => {
                    self.event_cv
                        .wait_for(&mut guard, Duration::from_millis(100));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distinct_buffers_not_host_aliases() {
        let be = VirtualBackend::new(VirtualBackendConfig::default());
        let a = be.allocate(ResourceId::new("mock0"), 128, 64).unwrap();
        let b = be.allocate(ResourceId::new("mock0"), 128, 64).unwrap();
        assert_ne!(a.0, b.0);
        assert!(be.capabilities().simulated);
        be.free(a).unwrap();
        be.free(b).unwrap();
    }

    #[test]
    fn write_bytes_supports_buffers_larger_than_4kib() {
        let be = VirtualBackend::new(VirtualBackendConfig::default());
        let n = 16 * 1024;
        let h = be.allocate(ResourceId::new("mock0"), n, 64).unwrap();
        let data = vec![0xABu8; n];
        be.write_bytes(h, &data).unwrap();
        assert_eq!(be.read_bytes(h).unwrap(), data);
        be.free(h).unwrap();
    }

    #[test]
    fn submit_returns_pending_immediately() {
        let be = VirtualBackend::new(VirtualBackendConfig {
            compute_delay_s: 0.05,
            ..Default::default()
        });
        let stream = StreamId::new("compute0");
        let t0 = Instant::now();
        let ev = be.launch(ExecutableHandle(1), &[], &[], stream).unwrap();
        assert!(
            t0.elapsed().as_secs_f64() < 0.01,
            "launch must not sleep on caller"
        );
        assert_eq!(be.query_event(ev).unwrap(), EventStatus::Pending);
        be.wait_event(ev).unwrap();
        assert_eq!(be.query_event(ev).unwrap(), EventStatus::Complete);
    }

    #[test]
    fn ordered_stream_serializes() {
        let be = VirtualBackend::new(VirtualBackendConfig {
            compute_delay_s: 0.02,
            ..Default::default()
        });
        let stream = StreamId::new("compute0");
        let t0 = Instant::now();
        let e1 = be
            .launch(ExecutableHandle(1), &[], &[], stream.clone())
            .unwrap();
        let e2 = be.launch(ExecutableHandle(1), &[], &[], stream).unwrap();
        be.wait_event(e1).unwrap();
        be.wait_event(e2).unwrap();
        // Two serial 20ms jobs on one ordered stream ≥ ~40ms wall.
        assert!(t0.elapsed().as_secs_f64() >= 0.035);
    }

    #[test]
    fn separate_streams_overlap() {
        let be = VirtualBackend::new(VirtualBackendConfig {
            compute_delay_s: 0.03,
            ..Default::default()
        });
        let t0 = Instant::now();
        let e1 = be
            .launch(ExecutableHandle(1), &[], &[], StreamId::new("compute0"))
            .unwrap();
        let e2 = be
            .launch(ExecutableHandle(1), &[], &[], StreamId::new("compute1"))
            .unwrap();
        be.wait_event(e1).unwrap();
        be.wait_event(e2).unwrap();
        // Parallel streams should finish near one delay, not two.
        assert!(t0.elapsed().as_secs_f64() < 0.055);
    }
}

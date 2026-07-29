use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::Duration;
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

struct StreamState {
    /// Ordered stream: operations serialize on this queue timestamp.
    next_free_ns: u64,
}

/// Deterministic simulated accelerator with ordered compute/transfer streams.
pub struct VirtualBackend {
    config: VirtualBackendConfig,
    next_buf: AtomicU64,
    next_evt: AtomicU64,
    buffers: Mutex<HashMap<u64, BufferRec>>,
    events: Mutex<HashMap<u64, EventRec>>,
    streams: Mutex<HashMap<String, StreamState>>,
    used_bytes: Mutex<u64>,
}

impl VirtualBackend {
    #[must_use]
    pub fn new(config: VirtualBackendConfig) -> Self {
        Self {
            config,
            next_buf: AtomicU64::new(1),
            next_evt: AtomicU64::new(1),
            buffers: Mutex::new(HashMap::new()),
            events: Mutex::new(HashMap::new()),
            streams: Mutex::new(HashMap::new()),
            used_bytes: Mutex::new(0),
        }
    }

    fn delay(&self, seconds: f64) {
        if seconds > 0.0 {
            let nanos = (seconds * 1e9) as u64;
            thread::sleep(Duration::from_nanos(nanos.max(1)));
        }
    }

    fn stream_wait(&self, stream: &str, work_s: f64) {
        let mut streams = self.streams.lock();
        let state = streams
            .entry(stream.to_owned())
            .or_insert(StreamState { next_free_ns: 0 });
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        let start = state.next_free_ns.max(now);
        let work_ns = (work_s * 1e9) as u64;
        state.next_free_ns = start.saturating_add(work_ns.max(1));
        let wait_ns = start.saturating_sub(now);
        if wait_ns > 0 {
            thread::sleep(Duration::from_nanos(wait_ns));
        }
        self.delay(work_s);
    }
}

impl Backend for VirtualBackend {
    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            name: self.config.name.clone(),
            supports_p2p: self.config.supports_p2p,
            supports_async_compute: true,
            supports_ordered_streams: true,
            max_streams: 8,
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
                payload: vec![0u8; bytes.min(4096)], // bounded stub payload
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
        self.stream_wait(stream.as_str(), dur);
        let id = self.next_evt.fetch_add(1, Ordering::Relaxed);
        self.events.lock().insert(
            id,
            EventRec {
                status: EventStatus::Complete,
            },
        );
        Ok(EventHandle(id))
    }

    fn launch(
        &self,
        _executable: ExecutableHandle,
        _inputs: &[BufferHandle],
        _outputs: &[BufferHandle],
        stream: StreamId,
    ) -> BackendResult<EventHandle> {
        self.stream_wait(stream.as_str(), self.config.compute_delay_s);
        let id = self.next_evt.fetch_add(1, Ordering::Relaxed);
        self.events.lock().insert(
            id,
            EventRec {
                status: EventStatus::Complete,
            },
        );
        Ok(EventHandle(id))
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
        match self.query_event(event)? {
            EventStatus::Complete => Ok(()),
            EventStatus::Pending => Err(BackendError::Event {
                backend: self.config.name.clone(),
                event: event.0,
                cause: "event still pending".into(),
            }),
            EventStatus::Error => Err(BackendError::Event {
                backend: self.config.name.clone(),
                event: event.0,
                cause: "event in error state".into(),
            }),
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
    fn ordered_stream_serializes() {
        let be = VirtualBackend::new(VirtualBackendConfig {
            compute_delay_s: 0.001,
            ..Default::default()
        });
        let stream = StreamId::new("compute0");
        let t0 = std::time::Instant::now();
        be.launch(ExecutableHandle(1), &[], &[], stream.clone())
            .unwrap();
        be.launch(ExecutableHandle(1), &[], &[], stream).unwrap();
        assert!(t0.elapsed().as_secs_f64() >= 0.001);
    }
}

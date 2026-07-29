//! Explicit execution resources: streams, copy engines, links, I/O queues.

use std::collections::HashMap;

/// Ordered stream occupancy (submission order preserved).
#[derive(Clone, Debug, Default)]
pub struct OrderedStreamState {
    pub last_end_s: f64,
    pub inflight: u32,
}

/// Capacity-limited engine / queue.
#[derive(Clone, Debug)]
pub struct CapacityState {
    pub max_concurrent: u32,
    pub inflight: u32,
}

impl CapacityState {
    #[must_use]
    pub fn new(max_concurrent: u32) -> Self {
        Self {
            max_concurrent: max_concurrent.max(1),
            inflight: 0,
        }
    }

    pub fn try_acquire(&mut self) -> bool {
        if self.inflight >= self.max_concurrent {
            return false;
        }
        self.inflight += 1;
        true
    }

    pub fn release(&mut self) {
        self.inflight = self.inflight.saturating_sub(1);
    }
}

/// Bandwidth-limited interconnect.
#[derive(Clone, Debug, Default)]
pub struct BandwidthState {
    pub busy_until_s: f64,
    pub bytes_in_flight: u64,
}

/// Runtime resource occupancy keyed by schedule ids.
#[derive(Clone, Debug, Default)]
pub struct ResourceState {
    pub streams: HashMap<String, OrderedStreamState>,
    pub copy_engines: HashMap<String, CapacityState>,
    pub links: HashMap<String, BandwidthState>,
    pub io_queues: HashMap<String, CapacityState>,
}

impl ResourceState {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn note_stream_submit(&mut self, stream_id: &str, end_s: f64) {
        let st = self.streams.entry(stream_id.to_owned()).or_default();
        st.last_end_s = st.last_end_s.max(end_s);
        st.inflight = st.inflight.saturating_add(1);
    }

    pub fn note_stream_complete(&mut self, stream_id: &str) {
        if let Some(st) = self.streams.get_mut(stream_id) {
            st.inflight = st.inflight.saturating_sub(1);
        }
    }

    pub fn ensure_copy_engine(&mut self, engine_id: &str, max_concurrent: u32) -> &mut CapacityState {
        self.copy_engines
            .entry(engine_id.to_owned())
            .or_insert_with(|| CapacityState::new(max_concurrent))
    }

    pub fn ensure_io_queue(&mut self, queue_id: &str, max_depth: u32) -> &mut CapacityState {
        self.io_queues
            .entry(queue_id.to_owned())
            .or_insert_with(|| CapacityState::new(max_depth))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn copy_engine_capacity() {
        let mut s = ResourceState::new();
        let eng = s.ensure_copy_engine("cpu::copy0", 1);
        assert!(eng.try_acquire());
        assert!(!eng.try_acquire());
        eng.release();
        assert!(eng.try_acquire());
    }
}

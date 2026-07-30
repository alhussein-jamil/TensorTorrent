//! Reference-counted chunk cache with simple LRU eviction.

use parking_lot::Mutex;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

#[derive(Clone)]
struct Entry {
    data: Arc<Vec<u8>>,
    refs: usize,
}

pub struct ChunkCache {
    inner: Mutex<Inner>,
}

struct Inner {
    map: HashMap<String, Entry>,
    order: VecDeque<String>,
    capacity_bytes: u64,
    live_bytes: u64,
    hits: u64,
    misses: u64,
}

impl ChunkCache {
    #[must_use]
    pub fn new(capacity_bytes: u64) -> Self {
        Self {
            inner: Mutex::new(Inner {
                map: HashMap::new(),
                order: VecDeque::new(),
                capacity_bytes,
                live_bytes: 0,
                hits: 0,
                misses: 0,
            }),
        }
    }

    pub fn get(&self, key: &str) -> Option<Arc<Vec<u8>>> {
        let mut g = self.inner.lock();
        let data = {
            let Some(e) = g.map.get_mut(key) else {
                g.misses += 1;
                return None;
            };
            e.refs += 1;
            Arc::clone(&e.data)
        };
        g.hits += 1;
        if let Some(pos) = g.order.iter().position(|k| k == key) {
            g.order.remove(pos);
        }
        g.order.push_back(key.to_owned());
        Some(data)
    }

    pub fn insert(&self, key: impl Into<String>, data: Vec<u8>) -> Arc<Vec<u8>> {
        let key = key.into();
        let arc = Arc::new(data);
        let mut g = self.inner.lock();
        let nbytes = arc.len() as u64;
        while g.live_bytes + nbytes > g.capacity_bytes && !g.order.is_empty() {
            if let Some(old) = g.order.pop_front() {
                if let Some(e) = g.map.get(&old) {
                    if e.refs > 0 {
                        // skip in-use; push back
                        g.order.push_back(old);
                        break;
                    }
                }
                if let Some(e) = g.map.remove(&old) {
                    g.live_bytes = g.live_bytes.saturating_sub(e.data.len() as u64);
                }
            }
        }
        g.live_bytes = g.live_bytes.saturating_add(nbytes);
        g.map.insert(
            key.clone(),
            Entry {
                data: Arc::clone(&arc),
                refs: 1,
            },
        );
        g.order.push_back(key);
        arc
    }

    pub fn release(&self, key: &str) {
        let mut g = self.inner.lock();
        if let Some(e) = g.map.get_mut(key) {
            e.refs = e.refs.saturating_sub(1);
        }
    }

    #[must_use]
    pub fn stats(&self) -> (u64, u64, u64) {
        let g = self.inner.lock();
        (g.hits, g.misses, g.live_bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hit_miss() {
        let c = ChunkCache::new(1024);
        assert!(c.get("a").is_none());
        c.insert("a", vec![1, 2, 3]);
        assert!(c.get("a").is_some());
        let (h, m, _) = c.stats();
        assert_eq!(h, 1);
        assert_eq!(m, 1);
    }
}

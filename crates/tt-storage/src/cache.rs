//! Reference-counted chunk cache with simple LRU eviction.

use parking_lot::Mutex;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

#[derive(Clone)]
struct Entry {
    data: Arc<Vec<u8>>,
    refs: usize,
    generation: u64,
}

pub struct ChunkCache {
    inner: Mutex<Inner>,
}

struct Inner {
    map: HashMap<String, Entry>,
    order: VecDeque<(String, u64)>,
    generation: u64,
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
                generation: 0,
                capacity_bytes,
                live_bytes: 0,
                hits: 0,
                misses: 0,
            }),
        }
    }

    pub fn get(&self, key: &str) -> Option<Arc<Vec<u8>>> {
        let mut g = self.inner.lock();
        g.generation = g.generation.wrapping_add(1);
        let generation = g.generation;
        let data = {
            let Some(e) = g.map.get_mut(key) else {
                g.misses += 1;
                return None;
            };
            e.refs = e.refs.saturating_add(1);
            e.generation = generation;
            Arc::clone(&e.data)
        };
        g.hits += 1;
        if g.map.len() <= 32 {
            if let Some(position) = g.order.iter().position(|(candidate, _)| candidate == key) {
                g.order.remove(position);
            }
        }
        g.order.push_back((key.to_owned(), generation));
        if g.order.len() > g.map.len().saturating_mul(4).max(64) {
            let mut order = std::mem::take(&mut g.order);
            order.retain(|(candidate, candidate_generation)| {
                g.map
                    .get(candidate)
                    .is_some_and(|entry| entry.generation == *candidate_generation)
            });
            g.order = order;
        }
        Some(data)
    }

    pub fn insert(&self, key: impl Into<String>, data: Vec<u8>) -> Arc<Vec<u8>> {
        let key = key.into();
        let arc = Arc::new(data);
        let mut g = self.inner.lock();
        let nbytes = arc.len() as u64;
        if let Some(previous) = g.map.remove(&key) {
            g.live_bytes = g.live_bytes.saturating_sub(previous.data.len() as u64);
        }
        let mut candidates = g.map.len();
        while g.live_bytes.saturating_add(nbytes) > g.capacity_bytes && candidates > 0 {
            if let Some((old, generation)) = g.order.pop_front() {
                let Some(entry) = g.map.get(&old) else {
                    continue;
                };
                if entry.generation != generation {
                    continue;
                }
                candidates -= 1;
                if entry.refs > 0 {
                    g.order.push_back((old, generation));
                    continue;
                }
                if let Some(e) = g.map.remove(&old) {
                    g.live_bytes = g.live_bytes.saturating_sub(e.data.len() as u64);
                }
            }
        }
        g.generation = g.generation.wrapping_add(1);
        let generation = g.generation;
        g.live_bytes = g.live_bytes.saturating_add(nbytes);
        g.map.insert(
            key.clone(),
            Entry {
                data: Arc::clone(&arc),
                // `insert` does not lease the entry. Only successful `get`
                // calls pin it until the matching `release`.
                refs: 0,
                generation,
            },
        );
        g.order.push_back((key, generation));
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

    #[test]
    fn insert_is_not_permanently_pinned_and_evicts_lru() {
        let cache = ChunkCache::new(3);
        cache.insert("a", vec![1, 2, 3]);
        cache.insert("b", vec![4, 5, 6]);
        assert!(cache.get("a").is_none());
        assert!(cache.get("b").is_some());
        cache.release("b");
        assert_eq!(cache.stats().2, 3);
    }

    #[test]
    fn eviction_skips_pinned_entry_and_replacement_does_not_double_count() {
        let cache = ChunkCache::new(6);
        cache.insert("a", vec![1, 2, 3]);
        cache.insert("b", vec![4, 5, 6]);
        assert!(cache.get("a").is_some());
        cache.insert("c", vec![7, 8, 9]);
        assert!(cache.get("b").is_none());
        assert_eq!(cache.stats().2, 6);
        cache.insert("c", vec![10]);
        assert_eq!(cache.stats().2, 4);
        cache.release("a");
    }

    #[test]
    fn repeated_hits_keep_lru_bookkeeping_bounded() {
        let cache = ChunkCache::new(2);
        cache.insert("a", vec![1]);
        cache.insert("b", vec![2]);
        for _ in 0..1_000 {
            assert!(cache.get("a").is_some());
            cache.release("a");
        }
        assert!(cache.inner.lock().order.len() <= 64);

        cache.insert("c", vec![3]);
        assert!(cache.get("a").is_some());
        assert!(cache.get("b").is_none());
        assert!(cache.get("c").is_some());
    }
}

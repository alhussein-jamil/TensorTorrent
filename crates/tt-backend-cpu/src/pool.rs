//! Bounded CPU worker pools (compute vs I/O). Work stealing stays within one pool.

use crossbeam_channel::{bounded, Receiver, Sender};
use parking_lot::{Condvar, Mutex};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorkerPoolKind {
    Compute,
    Io,
}

#[derive(Clone, Debug)]
pub struct CpuPoolConfig {
    pub name: String,
    pub workers: usize,
    pub kind: WorkerPoolKind,
}

type Job = Box<dyn FnOnce() + Send + 'static>;

struct Shared {
    pending: AtomicUsize,
    cancel: AtomicBool,
    shutdown: AtomicBool,
    lock: Mutex<()>,
    cv: Condvar,
}

pub struct BoundedPool {
    tx: Mutex<Option<Sender<Job>>>,
    shared: Arc<Shared>,
    handles: Mutex<Vec<JoinHandle<()>>>,
    workers: usize,
    #[allow(dead_code)]
    kind: WorkerPoolKind,
}

#[allow(dead_code)]
impl BoundedPool {
    /// Build a pool, propagating thread-spawn failure instead of panicking.
    ///
    /// Spawn can fail under pid-cgroup limits or memory pressure; a panic here
    /// would abort the embedding Python interpreter with no traceback. Threads
    /// spawned before a failure are shut down cleanly.
    pub fn try_new(name: &str, workers: usize, kind: WorkerPoolKind) -> Result<Self, String> {
        let workers = workers.max(1);
        let (tx, rx) = bounded::<Job>(workers.saturating_mul(8).max(1));
        let shared = Arc::new(Shared {
            pending: AtomicUsize::new(0),
            cancel: AtomicBool::new(false),
            shutdown: AtomicBool::new(false),
            lock: Mutex::new(()),
            cv: Condvar::new(),
        });
        let mut handles = Vec::with_capacity(workers);
        for i in 0..workers {
            let rx = rx.clone();
            let shared_for_worker = Arc::clone(&shared);
            let thread_name = format!("{name}-{i}");
            match thread::Builder::new()
                .name(thread_name)
                .spawn(move || worker_loop(rx, shared_for_worker))
            {
                Ok(handle) => handles.push(handle),
                Err(error) => {
                    shared.shutdown.store(true, Ordering::Release);
                    drop(tx);
                    for h in handles.drain(..) {
                        let _ = h.join();
                    }
                    return Err(format!(
                        "failed to spawn cpu pool thread {name}-{i} \
                         (check process/thread limits): {error}"
                    ));
                }
            }
        }
        Ok(Self {
            tx: Mutex::new(Some(tx)),
            shared,
            handles: Mutex::new(handles),
            workers,
            kind,
        })
    }

    pub fn workers(&self) -> usize {
        self.workers
    }

    pub fn kind(&self) -> WorkerPoolKind {
        self.kind
    }

    pub fn submit<F>(&self, job: F) -> Result<(), String>
    where
        F: FnOnce() + Send + 'static,
    {
        if self.shared.shutdown.load(Ordering::Acquire)
            || self.shared.cancel.load(Ordering::Acquire)
        {
            return Err("pool unavailable".into());
        }
        let guard = self.tx.lock();
        let Some(tx) = guard.as_ref() else {
            return Err("pool shut down".into());
        };
        self.shared.pending.fetch_add(1, Ordering::AcqRel);
        tx.try_send(Box::new(job)).map_err(|error| {
            self.shared.pending.fetch_sub(1, Ordering::AcqRel);
            format!("pool queue unavailable: {error}")
        })?;
        Ok(())
    }

    pub fn synchronize(&self) {
        let mut lock = self.shared.lock.lock();
        while self.shared.pending.load(Ordering::Acquire) > 0 {
            self.shared.cv.wait(&mut lock);
        }
    }

    pub fn cancel(&self) {
        self.shared.cancel.store(true, Ordering::Release);
    }

    pub fn shutdown(&self) {
        self.shared.shutdown.store(true, Ordering::Release);
        *self.tx.lock() = None;
        let mut handles = self.handles.lock();
        for h in handles.drain(..) {
            let _ = h.join();
        }
    }
}

fn worker_loop(rx: Receiver<Job>, shared: Arc<Shared>) {
    while let Ok(job) = rx.recv() {
        if shared.shutdown.load(Ordering::Acquire) {
            shared.pending.fetch_sub(1, Ordering::AcqRel);
            shared.cv.notify_all();
            break;
        }
        if !shared.cancel.load(Ordering::Acquire) {
            // User/backend work must not kill a pool thread or strand the
            // pending counter, which would make `synchronize` wait forever.
            let _ = catch_unwind(AssertUnwindSafe(job));
        }
        shared.pending.fetch_sub(1, Ordering::AcqRel);
        shared.cv.notify_all();
    }
}

impl Drop for BoundedPool {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};

    #[test]
    fn panicking_job_does_not_break_pool_or_synchronize() {
        let pool = BoundedPool::try_new("panic-test", 1, WorkerPoolKind::Compute).unwrap();
        let completed = Arc::new(AtomicBool::new(false));
        pool.submit(|| panic!("intentional test panic")).unwrap();
        let completed_job = Arc::clone(&completed);
        pool.submit(move || completed_job.store(true, Ordering::Release))
            .unwrap();
        pool.synchronize();
        assert!(completed.load(Ordering::Acquire));
    }
}

//! CPU / I/O / transfer worker pools with bounded in-flight work.

use crate::error::{RuntimeError, RuntimeResult};
use crossbeam_channel::{bounded, Receiver, Sender};
use parking_lot::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};

type Job = Box<dyn FnOnce() + Send + 'static>;

pub struct WorkerPool {
    name: String,
    tx: Sender<Job>,
    handles: Mutex<Vec<JoinHandle<()>>>,
    shutdown: Arc<AtomicBool>,
}

impl WorkerPool {
    pub fn try_new(
        name: impl Into<String>,
        workers: usize,
        queue_capacity: usize,
    ) -> RuntimeResult<Self> {
        let name = name.into();
        let (tx, rx) = bounded::<Job>(queue_capacity.max(1));
        let shutdown = Arc::new(AtomicBool::new(false));
        let mut handles = Vec::new();
        for i in 0..workers.max(1) {
            let rx = rx.clone();
            let shutdown = Arc::clone(&shutdown);
            let worker_name = format!("{name}-{i}");
            let handle = thread::Builder::new()
                .name(worker_name)
                .spawn(move || worker_loop(rx, shutdown))
                .map_err(|e| {
                    Box::new(RuntimeError::Other(format!(
                        "failed to spawn worker pool {name}: {e}"
                    )))
                })?;
            handles.push(handle);
        }
        Ok(Self {
            name,
            tx,
            handles: Mutex::new(handles),
            shutdown,
        })
    }

    pub fn submit<F>(&self, job: F) -> bool
    where
        F: FnOnce() + Send + 'static,
    {
        if self.shutdown.load(Ordering::Acquire) {
            return false;
        }
        self.tx.send(Box::new(job)).is_ok()
    }

    pub fn shutdown(&self) {
        self.shutdown.store(true, Ordering::Release);
        let n = self.handles.lock().len();
        for _ in 0..n {
            let _ = self.tx.send(Box::new(|| {}));
        }
    }

    pub fn join(&self) {
        self.shutdown();
        let mut handles = self.handles.lock();
        for h in handles.drain(..) {
            let _ = h.join();
        }
    }

    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }
}

impl Drop for WorkerPool {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Release);
    }
}

fn worker_loop(rx: Receiver<Job>, shutdown: Arc<AtomicBool>) {
    while !shutdown.load(Ordering::Acquire) {
        match rx.recv() {
            // A panicking job must not kill the worker thread: the pool would
            // silently shrink and completions owed by later jobs would never
            // arrive. Jobs report their own failure; a completion lost to a
            // panic is caught by the executor's stall watchdog.
            Ok(job) => {
                let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(job));
            }
            Err(_) => break,
        }
    }
}

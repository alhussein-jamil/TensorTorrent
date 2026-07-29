//! CPU / I/O / transfer worker pools with bounded in-flight work.

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
    #[must_use]
    pub fn new(name: impl Into<String>, workers: usize, queue_capacity: usize) -> Self {
        let name = name.into();
        let (tx, rx) = bounded::<Job>(queue_capacity.max(1));
        let shutdown = Arc::new(AtomicBool::new(false));
        let mut handles = Vec::new();
        for i in 0..workers.max(1) {
            let rx = rx.clone();
            let shutdown = Arc::clone(&shutdown);
            let worker_name = format!("{name}-{i}");
            handles.push(
                thread::Builder::new()
                    .name(worker_name)
                    .spawn(move || worker_loop(rx, shutdown))
                    .expect("spawn worker"),
            );
        }
        Self {
            name,
            tx,
            handles: Mutex::new(handles),
            shutdown,
        }
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
        // Drop senders by replacing channel... workers exit when rx disconnects.
        // Send no-op poison pills equal to worker count.
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
            Ok(job) => job(),
            Err(_) => break,
        }
    }
}

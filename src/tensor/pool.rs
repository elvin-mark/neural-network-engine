//! High-performance thread-local zero-allocation tensor buffer pool.
//!
//! Eliminates heap allocation (`malloc`/`free`) overhead during training and inference loops
//! by caching and recycling `Vec<f32>` scratch buffers using size-bucketed free-lists.

use std::cell::RefCell;
use std::fmt;

/// Maximum number of bytes cached per thread (default 512 MiB).
pub const DEFAULT_MAX_CACHED_BYTES: usize = 512 * 1024 * 1024;

/// Minimum bucket power of two (2^6 = 64 floats = 256 bytes).
pub const MIN_BUCKET_POW2: usize = 6;
/// Maximum bucket power of two (2^28 = ~268M floats = ~1 GiB).
pub const MAX_BUCKET_POW2: usize = 28;

thread_local! {
    static LOCAL_POOL: RefCell<TensorPool> = RefCell::new(TensorPool::new(DEFAULT_MAX_CACHED_BYTES));
}

/// Telemetry statistics for the tensor memory pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PoolStats {
    /// Number of successful buffer reuses (cache hits).
    pub hits: usize,
    /// Number of new allocations performed when no cached buffer was available.
    pub misses: usize,
    /// Total number of bytes currently cached in the pool.
    pub cached_bytes: usize,
    /// Total number of bytes requested and allocated through the pool.
    pub allocated_bytes: usize,
    /// Number of active recycled buffers in the free-lists.
    pub free_buffers: usize,
}

impl PoolStats {
    /// Returns the cache hit rate percentage (0.0 to 100.0%).
    pub fn hit_rate(&self) -> f32 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            (self.hits as f32 / total as f32) * 100.0
        }
    }
}

impl fmt::Display for PoolStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PoolStats {{ hits: {}, misses: {}, hit_rate: {:.1}%, cached: {:.2} MB, active_buffers: {} }}",
            self.hits,
            self.misses,
            self.hit_rate(),
            self.cached_bytes as f32 / (1024.0 * 1024.0),
            self.free_buffers
        )
    }
}

/// Size-bucketed recycling memory pool for `Vec<f32>` buffers.
pub struct TensorPool {
    buckets: Vec<Vec<Vec<f32>>>,
    max_cached_bytes: usize,
    cached_bytes: usize,
    hits: usize,
    misses: usize,
    allocated_bytes: usize,
    enabled: bool,
}

impl TensorPool {
    /// Creates a new buffer pool with the specified memory limit.
    pub fn new(max_cached_bytes: usize) -> Self {
        let num_buckets = MAX_BUCKET_POW2 - MIN_BUCKET_POW2 + 1;
        let mut buckets = Vec::with_capacity(num_buckets);
        for _ in 0..num_buckets {
            buckets.push(Vec::new());
        }
        Self {
            buckets,
            max_cached_bytes,
            cached_bytes: 0,
            hits: 0,
            misses: 0,
            allocated_bytes: 0,
            enabled: true,
        }
    }

    #[inline]
    fn bucket_idx(capacity: usize) -> usize {
        let cap = capacity.max(1 << MIN_BUCKET_POW2);
        let next_pow2 = cap.next_power_of_two();
        let pow2 = next_pow2.trailing_zeros() as usize;
        let clamped_pow2 = pow2.clamp(MIN_BUCKET_POW2, MAX_BUCKET_POW2);
        clamped_pow2 - MIN_BUCKET_POW2
    }

    /// Acquires a `Vec<f32>` with at least `capacity` capacity.
    pub fn pop(&mut self, capacity: usize) -> Vec<f32> {
        if !self.enabled || capacity == 0 {
            return Vec::with_capacity(capacity);
        }

        let idx = Self::bucket_idx(capacity);
        if let Some(mut vec) = self.buckets[idx].pop() {
            self.hits += 1;
            let bytes = vec.capacity() * std::mem::size_of::<f32>();
            self.cached_bytes = self.cached_bytes.saturating_sub(bytes);
            vec.clear();
            vec
        } else {
            self.misses += 1;
            let target_cap = capacity.max(1 << (idx + MIN_BUCKET_POW2));
            let bytes = target_cap * std::mem::size_of::<f32>();
            self.allocated_bytes += bytes;
            Vec::with_capacity(target_cap)
        }
    }

    /// Recycles a `Vec<f32>` back into its size bucket.
    pub fn push(&mut self, mut vec: Vec<f32>) {
        let cap = vec.capacity();
        let bytes = cap * std::mem::size_of::<f32>();

        if !self.enabled
            || cap < (1 << MIN_BUCKET_POW2)
            || self.cached_bytes + bytes > self.max_cached_bytes
        {
            // Evict / drop to OS if pool is full or disabled
            return;
        }

        let idx = Self::bucket_idx(cap);
        vec.clear();
        self.cached_bytes += bytes;
        self.buckets[idx].push(vec);
    }

    /// Clears all cached buffers, releasing memory to the OS.
    pub fn clear(&mut self) {
        for b in &mut self.buckets {
            b.clear();
        }
        self.cached_bytes = 0;
    }

    /// Returns pool telemetry stats.
    pub fn stats(&self) -> PoolStats {
        let free_buffers: usize = self.buckets.iter().map(|b| b.len()).sum();
        PoolStats {
            hits: self.hits,
            misses: self.misses,
            cached_bytes: self.cached_bytes,
            allocated_bytes: self.allocated_bytes,
            free_buffers,
        }
    }

    /// Enables or disables recycling.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
        if !enabled {
            self.clear();
        }
    }

    /// Checks if the pool is active.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    // =========================================================================
    // Thread-Local Global Static Helpers
    // =========================================================================

    /// Acquires a scratch buffer from the current thread's pool.
    #[inline]
    pub fn acquire(capacity: usize) -> Vec<f32> {
        LOCAL_POOL.with(|p| p.borrow_mut().pop(capacity))
    }

    /// Recycles a buffer into the current thread's pool.
    #[inline]
    pub fn recycle(vec: Vec<f32>) {
        LOCAL_POOL.with(|p| p.borrow_mut().push(vec));
    }

    /// Clears the current thread's buffer pool.
    pub fn clear_local() {
        LOCAL_POOL.with(|p| p.borrow_mut().clear());
    }

    /// Returns telemetry statistics for the current thread's pool.
    pub fn local_stats() -> PoolStats {
        LOCAL_POOL.with(|p| p.borrow().stats())
    }

    /// Enables or disables the current thread's pool.
    pub fn set_local_enabled(enabled: bool) {
        LOCAL_POOL.with(|p| p.borrow_mut().set_enabled(enabled));
    }
}

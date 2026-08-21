//! Context managers and thread-local state for autograd (e.g., `no_grad`).

use std::cell::RefCell;

thread_local! {
    static GRAD_ENABLED: RefCell<bool> = const { RefCell::new(true) };
}

/// Returns whether gradient computation is currently enabled for this thread.
pub fn is_grad_enabled() -> bool {
    GRAD_ENABLED.with(|g| *g.borrow())
}

/// Sets whether gradient computation is enabled for this thread.
pub fn set_grad_enabled(enabled: bool) {
    GRAD_ENABLED.with(|g| *g.borrow_mut() = enabled);
}

/// RAII guard that disables gradient tracking while in scope.
pub struct NoGradGuard {
    prev_state: bool,
}

impl Default for NoGradGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl NoGradGuard {
    pub fn new() -> Self {
        let prev_state = is_grad_enabled();
        set_grad_enabled(false);
        Self { prev_state }
    }
}

impl Drop for NoGradGuard {
    fn drop(&mut self) {
        set_grad_enabled(self.prev_state);
    }
}

/// Executes a closure with gradient computation disabled.
pub fn no_grad<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    let _guard = NoGradGuard::new();
    f()
}

//! The zero-allocation proof.
//!
//! A counting global allocator wraps the system allocator. The test snapshots
//! the counter, runs the pack + kernel + unpack hot paths on caller-owned
//! buffers, and asserts the counter did not move. This is how "zero
//! allocations" is a verified claim rather than a README sentence.
//!
//! The proof is a single test on purpose: the counter is process-global and
//! the test harness runs tests in parallel threads, so two concurrent tests
//! would see each other's allocations.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

struct Counting;

static ALLOCS: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: forwarded to the system allocator with the same layout.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: forwarded with the same pointer/layout the allocator handed out.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: forwarded with the same pointer/layout contract.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

#[test]
fn hot_paths_perform_zero_allocations() {
    use gaia_mlx_quant::{pack_ternary_into, ternary_matmul, unpack_ternary_into};

    const ROWS: usize = 64;
    const COLS: usize = 64;

    // Setup (allocates — that is fine, it is not the hot path).
    let trits: Vec<i8> = (0..ROWS * COLS)
        .map(|i| [-1, 0, 1][(i * 7 + 3) % 3])
        .collect();
    let x: Vec<f64> = (0..ROWS).map(|j| (j as f64 + 0.5) / ROWS as f64).collect();
    let scales: Vec<f64> = (0..ROWS).map(|j| 0.5 + (j % 7) as f64 * 0.1).collect();

    let mut packed = vec![0u8; (ROWS * COLS).div_ceil(5)];
    let mut out = vec![0f64; COLS];
    let mut unpacked = vec![0i8; ROWS * COLS];

    // Warm the paths (first-call branches and tables are settled).
    pack_ternary_into(&trits, &mut packed);
    ternary_matmul(&packed, ROWS, COLS, &x, Some(&scales), 1.0, &mut out);
    unpack_ternary_into(&packed, ROWS * COLS, &mut unpacked);

    // The proof itself. `packed` is zeroed between iterations per the packing
    // contract (the accumulate is `+=`); zeroing is a memset, not an
    // allocation.
    let before = ALLOCS.load(Ordering::Relaxed);
    for _ in 0..100 {
        packed.fill(0);
        pack_ternary_into(&trits, &mut packed);
        ternary_matmul(&packed, ROWS, COLS, &x, Some(&scales), 1.0, &mut out);
        unpack_ternary_into(&packed, ROWS * COLS, &mut unpacked);
    }
    let after = ALLOCS.load(Ordering::Relaxed);

    assert_eq!(
        before, after,
        "the hot paths allocated {} times",
        after - before
    );
}

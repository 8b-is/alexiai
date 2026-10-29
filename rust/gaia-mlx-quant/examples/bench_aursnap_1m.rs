//! 1,000,000-Node `AURSNAP1` Benchmark Suite:
//! Compares lookup latency between:
//!   1. `get_by_id` (8-byte inlined Big-Endian `key_prefix: u64` single-register comparison)
//!   2. `get_by_id_arena_only` (standard arena-pointer binary search dereferencing `id_off..id_end` on every probe)
//!   3. `scan_prefix_into` (register-bitmask prefix range scan `(entry_prefix & mask) == prefix_u64`)
//!
//! Snapshot geometry (1,000,000 nodes):
//!   - Header: 64 B
//!   - Index Table: 1,000,000 * 32 B = 32.0 MB
//!   - String & Payload Arena (24B key + 64B payload): 1,000,000 * 88 B = 88.0 MB
//!   - Total Mapped Snapshot: ~120.0 MB (exceeds L2/L3 cache capacity)

use std::hint::black_box;
use std::time::Instant;

use gaia_mlx_quant::{
    AURSNAP1_HEADER_LEN, AURSNAP1_NODE_ENTRY_LEN, AurSnapDecodedNode, AurSnapHeader64,
    AurSnapIndexView, AurSnapNodeEntry32, wupz_encode_aursnap1_header_64_into,
    wupz_encode_aursnap1_node_32_into, wupz_key_prefix_be_u64,
};

const NODE_COUNT: usize = 1_000_000;
const KEY_LEN: usize = 24;
const PAYLOAD_LEN: usize = 64;
const QUERY_BATCH: usize = 4_096;

/// Format a 24-byte ASCII key with a distinct 8-byte hex prefix (`"{i:08x}_node_key_{i:06x}"`).
#[inline]
fn format_distinct_key(i: usize, out: &mut [u8; KEY_LEN]) {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let v = i as u32;
    for (k, slot) in out.iter_mut().enumerate().take(8) {
        let nib = ((v >> ((7 - k) * 4)) & 0xF) as usize;
        *slot = HEX[nib];
    }
    out[8..18].copy_from_slice(b"_node_key_");
    for (k, slot) in out[18..24].iter_mut().enumerate() {
        let nib = ((v >> ((5 - k) * 4)) & 0xF) as usize;
        *slot = HEX[nib];
    }
}

/// Format a 24-byte ASCII key where clusters of 16 keys share an identical 8-byte prefix
/// (`"c{:07x}_suffix_{i:08x}"`), forcing the last 4 binary-search steps into the String Arena suffix fallback.
#[inline]
fn format_clustered_key(i: usize, out: &mut [u8; KEY_LEN]) {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let cluster = (i >> 4) as u32;
    let v = i as u32;
    out[0] = b'c';
    for (k, slot) in out[1..8].iter_mut().enumerate() {
        let nib = ((cluster >> ((6 - k) * 4)) & 0xF) as usize;
        *slot = HEX[nib];
    }
    out[8..16].copy_from_slice(b"_suffix_");
    for (k, slot) in out[16..24].iter_mut().enumerate() {
        let nib = ((v >> ((7 - k) * 4)) & 0xF) as usize;
        *slot = HEX[nib];
    }
}

fn build_1m_snapshot(clustered: bool) -> Vec<u8> {
    let index_bytes = NODE_COUNT * AURSNAP1_NODE_ENTRY_LEN;
    let arena_off = AURSNAP1_HEADER_LEN + index_bytes;
    let arena_bytes = NODE_COUNT * (KEY_LEN + PAYLOAD_LEN);
    let total_size = arena_off + arena_bytes;

    let mut buf = vec![0u8; total_size];
    let hdr = AurSnapHeader64 {
        version: 1,
        flags: 0x62,
        epoch: 2026,
        node_count: NODE_COUNT as u64,
        expected_file_size: total_size as u64,
        string_arena_offset: arena_off as u64,
    };
    wupz_encode_aursnap1_header_64_into(&hdr, &mut buf[..AURSNAP1_HEADER_LEN]).unwrap();

    let payload_template = [b'x'; PAYLOAD_LEN];
    let mut key_buf = [0u8; KEY_LEN];

    for i in 0..NODE_COUNT {
        if clustered {
            format_clustered_key(i, &mut key_buf);
        } else {
            format_distinct_key(i, &mut key_buf);
        }
        let id_off = i * (KEY_LEN + PAYLOAD_LEN);
        let pay_off = id_off + KEY_LEN;
        buf[arena_off + id_off..arena_off + id_off + KEY_LEN].copy_from_slice(&key_buf);
        buf[arena_off + pay_off..arena_off + pay_off + PAYLOAD_LEN]
            .copy_from_slice(&payload_template);

        let entry = AurSnapNodeEntry32 {
            salience_p: 850_000,
            salience_q: 1_000_000,
            payload_off: pay_off as u32,
            payload_len: PAYLOAD_LEN as u32,
            id_off: id_off as u32,
            id_len: KEY_LEN as u16,
            scope_id: (i & 3) as u8,
            pinned: u8::from((i & 15) == 0),
            digest_prefix: wupz_key_prefix_be_u64(&key_buf),
        };
        let entry_off = AURSNAP1_HEADER_LEN + i * AURSNAP1_NODE_ENTRY_LEN;
        wupz_encode_aursnap1_node_32_into(
            &entry,
            &mut buf[entry_off..entry_off + AURSNAP1_NODE_ENTRY_LEN],
        )
        .unwrap();
    }
    buf
}

fn main() {
    println!(
        "AURSNAP1 1,000,000-Node Zero-Copy Binary Search Benchmark (32 MB Index + 88 MB Arena = 120 MB)"
    );

    // Pre-generate deterministic pseudo-random query indices across [0, 1_000_000)
    let mut rng = 0x2026_1029_u64;
    let mut query_indices = Vec::with_capacity(QUERY_BATCH);
    for _ in 0..QUERY_BATCH {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        query_indices.push(((rng >> 32) as usize) % NODE_COUNT);
    }

    // ── Workload 1: Distinct 8-Byte Prefixes (1,000,000 nodes) ───────────────
    let snap_distinct = build_1m_snapshot(false);
    let view_distinct = AurSnapIndexView::open(&snap_distinct).unwrap();
    let queries_distinct: Vec<String> = query_indices
        .iter()
        .map(|&idx| {
            let mut kb = [0u8; KEY_LEN];
            format_distinct_key(idx, &mut kb);
            String::from_utf8(kb.to_vec()).unwrap()
        })
        .collect();

    // Warmup & equivalence check
    for q in &queries_distinct {
        let a = view_distinct.get_by_id(q).unwrap();
        let b = view_distinct.get_by_id_arena_only(q).unwrap();
        assert_eq!(a.node_id, b.node_id);
    }

    let rounds = 8usize;
    let t0 = Instant::now();
    for _ in 0..rounds {
        for q in &queries_distinct {
            black_box(view_distinct.get_by_id(black_box(q)).unwrap());
        }
    }
    let inline_distinct_ns = t0.elapsed().as_nanos() / ((rounds * QUERY_BATCH) as u128);

    let t1 = Instant::now();
    for _ in 0..rounds {
        for q in &queries_distinct {
            black_box(view_distinct.get_by_id_arena_only(black_box(q)).unwrap());
        }
    }
    let arena_distinct_ns = t1.elapsed().as_nanos() / ((rounds * QUERY_BATCH) as u128);

    // ── Workload 2: Clustered 8-Byte Prefixes (16 keys/cluster, 1,000,000 nodes) ──
    let snap_clustered = build_1m_snapshot(true);
    let view_clustered = AurSnapIndexView::open(&snap_clustered).unwrap();
    let queries_clustered: Vec<String> = query_indices
        .iter()
        .map(|&idx| {
            let mut kb = [0u8; KEY_LEN];
            format_clustered_key(idx, &mut kb);
            String::from_utf8(kb.to_vec()).unwrap()
        })
        .collect();

    for q in &queries_clustered {
        let a = view_clustered.get_by_id(q).unwrap();
        let b = view_clustered.get_by_id_arena_only(q).unwrap();
        assert_eq!(a.node_id, b.node_id);
    }

    let t2 = Instant::now();
    for _ in 0..rounds {
        for q in &queries_clustered {
            black_box(view_clustered.get_by_id(black_box(q)).unwrap());
        }
    }
    let inline_clustered_ns = t2.elapsed().as_nanos() / ((rounds * QUERY_BATCH) as u128);

    let t3 = Instant::now();
    for _ in 0..rounds {
        for q in &queries_clustered {
            black_box(view_clustered.get_by_id_arena_only(black_box(q)).unwrap());
        }
    }
    let arena_clustered_ns = t3.elapsed().as_nanos() / ((rounds * QUERY_BATCH) as u128);

    // ── Workload 3: Prefix Scan (`scan_prefix_into` over 16-key clusters) ────
    let dummy = AurSnapDecodedNode {
        node_id: "",
        payload: "",
        scope_id: 0,
        pinned: false,
        salience_p: 0,
        salience_q: 1,
        key_prefix: 0,
    };
    let mut out_scan = [dummy; 32];
    let prefixes_clustered: Vec<String> = query_indices
        .iter()
        .map(|&idx| {
            let mut kb = [0u8; KEY_LEN];
            format_clustered_key(idx, &mut kb);
            String::from_utf8(kb[..8].to_vec()).unwrap()
        })
        .collect();

    let t4 = Instant::now();
    for _ in 0..rounds {
        for pfx in &prefixes_clustered {
            let n = view_clustered
                .scan_prefix_into(black_box(pfx), black_box(&mut out_scan))
                .unwrap();
            assert_eq!(n, 16);
        }
    }
    let scan_16_ns = t4.elapsed().as_nanos() / ((rounds * QUERY_BATCH) as u128);

    let speedup_distinct = (arena_distinct_ns as f64) / (inline_distinct_ns.max(1) as f64);
    let speedup_clustered = (arena_clustered_ns as f64) / (inline_clustered_ns.max(1) as f64);

    println!("  workload                             inline_be_u64   arena_pointer   speedup");
    println!(
        "  1M distinct 8B prefixes (get_by_id)     {:>7} ns      {:>7} ns     {:>5.2}x",
        inline_distinct_ns, arena_distinct_ns, speedup_distinct
    );
    println!(
        "  1M 16-key 8B collisions (get_by_id)     {:>7} ns      {:>7} ns     {:>5.2}x",
        inline_clustered_ns, arena_clustered_ns, speedup_clustered
    );
    println!(
        "  1M prefix scan (16-node cluster)        {:>7} ns/scan (bitmask termination)",
        scan_16_ns
    );
}

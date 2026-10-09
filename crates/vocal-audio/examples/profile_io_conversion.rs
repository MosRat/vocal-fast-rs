//! Reproducible worker-side converter latency/allocation profile (no devices).
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;
use vocal_audio::StreamingFormatConverter;
use vocal_types::AudioFormat;

struct CountAlloc;
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
unsafe impl GlobalAlloc for CountAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) };
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.realloc(pointer, layout, size) }
    }
}
#[global_allocator]
static ALLOCATOR: CountAlloc = CountAlloc;

fn main() {
    println!(
        "input_hz,input_channels,output_hz,output_channels,p50_us,p95_us,p99_us,filter_delay_us,steady_allocations"
    );
    for (source_rate, source_channels, target_rate, target_channels) in [
        (16000, 2, 48000, 1),
        (48000, 2, 16000, 1),
        (44100, 1, 48000, 2),
        (48000, 1, 48000, 2),
    ] {
        let input = AudioFormat {
            sample_rate: source_rate,
            channels: source_channels,
        };
        let output = AudioFormat {
            sample_rate: target_rate,
            channels: target_channels,
        };
        let frames = source_rate as usize / 100;
        let samples = vec![0.25; frames * usize::from(source_channels)];
        let mut converter = StreamingFormatConverter::new(input, output, frames).unwrap();
        let mut scratch = Vec::with_capacity(converter.output_capacity_samples());
        let mut durations = Vec::with_capacity(4000);
        for _ in 0..100 {
            converter.process(&samples, &mut scratch).unwrap();
        }
        ALLOCATIONS.store(0, Ordering::Relaxed);
        for _ in 0..4000 {
            let started = Instant::now();
            converter
                .process(std::hint::black_box(&samples), &mut scratch)
                .unwrap();
            std::hint::black_box(&scratch);
            durations.push(started.elapsed().as_nanos() as f64 / 1000.);
        }
        let allocations = ALLOCATIONS.load(Ordering::Relaxed);
        assert_eq!(
            allocations, 0,
            "converter allocated during steady processing"
        );
        durations.sort_by(f64::total_cmp);
        let delay = converter.delay_frames() as f64 * 1_000_000. / f64::from(source_rate);
        println!(
            "{source_rate},{source_channels},{target_rate},{target_channels},{:.3},{:.3},{:.3},{delay:.3},{allocations}",
            durations[2000], durations[3800], durations[3960]
        );
    }
}

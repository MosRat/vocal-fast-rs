//! Native STFT bridge. Each worker exclusively owns its buffers; FFT plans are shared.
use rustfft::{Fft, FftPlanner, num_complex::Complex32};
use std::{
    ffi::c_void,
    panic::catch_unwind,
    sync::{Arc, OnceLock},
};

struct Context {
    plan: Arc<dyn Fft<f32>>,
    window: Vec<f32>,
    buffer: Vec<Complex32>,
    scratch: Vec<Complex32>,
}

#[unsafe(no_mangle)]
extern "C" fn vocal_cpu_fft_create() -> *mut c_void {
    catch_unwind(|| {
        static PLAN: OnceLock<Arc<dyn Fft<f32>>> = OnceLock::new();
        let plan = PLAN
            .get_or_init(|| FftPlanner::new().plan_fft_forward(1024))
            .clone();
        let context = Context {
            scratch: vec![Complex32::default(); plan.get_inplace_scratch_len()],
            plan,
            window: (0..1024)
                .map(|i| 0.5 - 0.5 * (std::f32::consts::TAU * i as f32 / 1024.).cos())
                .collect(),
            buffer: vec![Complex32::default(); 1024],
        };
        Box::into_raw(Box::new(context)).cast()
    })
    .unwrap_or(std::ptr::null_mut())
}

// C++ partitions output columns, so workers write disjoint elements. The ABI
// accepts a raw output pointer rather than creating overlapping &mut slices.
#[unsafe(no_mangle)]
unsafe extern "C" fn vocal_cpu_fft_run(
    context: *mut c_void,
    wave: *const f32,
    samples: usize,
    begin: usize,
    end: usize,
    magnitude: *mut f32,
) -> i32 {
    if context.is_null()
        || wave.is_null()
        || magnitude.is_null()
        || samples <= 512
        || begin > end
        || end > samples / 160 + 1
    {
        return -1;
    }
    catch_unwind(std::panic::AssertUnwindSafe(|| {
        // SAFETY: Native caller retains valid allocations for this synchronous run.
        let ctx = unsafe { &mut *context.cast::<Context>() };
        let wave = unsafe { std::slice::from_raw_parts(wave, samples) };
        let frames = samples / 160 + 1;
        for frame in begin..end {
            for i in 0..1024 {
                let position = (frame * 160 + i) as isize - 512;
                let reflected = if position < 0 {
                    -position
                } else if position >= samples as isize {
                    2 * samples as isize - 2 - position
                } else {
                    position
                } as usize;
                ctx.buffer[i] = Complex32::new(wave[reflected] * ctx.window[i], 0.);
            }
            ctx.plan
                .process_with_scratch(&mut ctx.buffer, &mut ctx.scratch);
            for bin in 0..513 {
                let c = ctx.buffer[bin];
                unsafe {
                    magnitude
                        .add(bin * frames + frame)
                        .write((c.re * c.re + c.im * c.im).sqrt());
                }
            }
        }
    }))
    .map_or(-1, |_| 0)
}

#[unsafe(no_mangle)]
unsafe extern "C" fn vocal_cpu_fft_destroy(context: *mut c_void) {
    if !context.is_null() {
        drop(unsafe { Box::from_raw(context.cast::<Context>()) });
    }
}

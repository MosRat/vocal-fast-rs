//! Exact interpolated-median register matching, entirely inside the GPU graph.
use crate::{
    Precision,
    graph::{DType, GraphBuilder, Op, Tensor},
};

pub(crate) fn build(g: &GraphBuilder, raw: &Tensor, manual_factor: &Tensor, debug: bool) -> Tensor {
    let n = raw.shape[1];
    let scalar = |x| g.scalar(x, Precision::Fp32);
    let enabled = g.input("auto_pitch_enabled", &[1], DType::F32);
    let target = g.input("auto_pitch_target", &[1], DType::F32);
    let locked = g.input("pitch_match_locked", &[1, 1], DType::F32);
    let offset = g.input("pitch_match_offset", &[1, 1], DType::F32);
    let voiced = g.binary(Op::Greater, raw, &scalar(0.));
    let mask = g.cast(&voiced, DType::F32);
    let count = g.reduce(Op::Sum, &mask, &[1]);
    let prefix = g.node(
        Op::Cumsum,
        &[&mask],
        mask.shape.clone(),
        DType::F32,
        vec![1, 0],
        [0.; 4],
        vec![],
        String::new(),
    );
    let positions = g.constant(
        &(0..n).map(|i| i as f32).collect::<Vec<_>>(),
        &[1, n],
        Precision::Fp32,
    );
    // Compact valid positions with a sorted TopK; subsequent indexed gathers
    // fill gaps in linear space rather than an N x N distance/mask matrix.
    let valid_positions = g.topk(
        &g.select(&voiced, &positions, &scalar(n as f32)),
        n,
        false,
        false,
    );
    let left_rank = g.binary(Op::Max, &g.sub(&prefix, &scalar(1.)), &scalar(0.));
    let right_rank = g.binary(
        Op::Max,
        &g.binary(Op::Min, &prefix, &g.sub(&count, &scalar(1.))),
        &scalar(0.),
    );
    let left = g.gather_elements(&valid_positions, &g.cast(&left_rank, DType::I32), 1);
    let right = g.gather_elements(&valid_positions, &g.cast(&right_rank, DType::I32), 1);
    let sample = |indices: &Tensor| {
        g.gather_elements(
            raw,
            &g.cast(
                &g.binary(Op::Min, indices, &scalar((n - 1) as f32)),
                DType::I32,
            ),
            1,
        )
    };
    let a = sample(&left);
    let b = sample(&right);
    let fraction = g.div(
        &g.sub(&positions, &left),
        &g.binary(Op::Max, &g.sub(&right, &left), &scalar(1.)),
    );
    let interpolated = g.add(&a, &g.mul(&g.sub(&b, &a), &fraction));
    let mid = n / 2;
    let sorted = g.topk(&interpolated, mid + 1, false, false);
    let upper = g.slice(&sorted, 1, mid, 1, 1);
    let median = if n.is_multiple_of(2) {
        g.scale(&g.add(&upper, &g.slice(&sorted, 1, mid - 1, 1, 1)), 0.5)
    } else {
        upper
    };
    let semitones = g.unary(
        Op::Round,
        &g.scale(
            &g.unary(
                Op::Log,
                &g.div(&target, &g.binary(Op::Max, &median, &scalar(1.))),
            ),
            12. / std::f32::consts::LN_2,
        ),
    );
    let candidate = g.binary(
        Op::Min,
        &g.binary(Op::Max, &semitones, &scalar(-12.)),
        &scalar(12.),
    );
    let enough = g.binary(
        Op::Greater,
        &count,
        &scalar((vocal_types::pitch_calibration_frames(n) - 1) as f32),
    );
    let was_locked = g.binary(Op::Greater, &locked, &scalar(0.));
    let active = g.binary(Op::Greater, &enabled, &scalar(0.));
    let chosen = g.select(
        &was_locked,
        &offset,
        &g.select(&enough, &candidate, &scalar(0.)),
    );
    let next_offset = g.select(&active, &chosen, &scalar(0.));
    let next_locked = g.select(
        &active,
        &g.binary(Op::Max, &locked, &g.cast(&enough, DType::F32)),
        &scalar(0.),
    );
    g.feedback(&locked, &next_locked);
    g.feedback(&offset, &next_offset);
    if debug {
        g.mark("auto_pitch_offset", &next_offset);
        g.mark("auto_pitch_median", &median);
        g.mark("auto_pitch_locked", &next_locked);
    }
    g.mul(
        manual_factor,
        &g.unary(
            Op::Exp,
            &g.scale(&next_offset, std::f32::consts::LN_2 / 12.),
        ),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DirectMlDevice;
    use vocal_types::{AutoPitchConfig, PitchMatcher};

    fn bytes(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    #[test]
    #[ignore = "requires DirectML GPU"]
    fn gpu_interpolation_calibration_freeze_reset_and_manual_addition() {
        let device = DirectMlDevice::new(0).unwrap();
        for n in [6, 7, 281, 282] {
            let g = GraphBuilder::default();
            let raw = g.input("raw", &[1, n], DType::F32);
            let manual = g.input("manual", &[1], DType::F32);
            let factor = build(&g, &raw, &manual, true);
            g.mark("scaled", &g.mul(&raw, &factor));
            let mut graph = device.compile(&g).unwrap();
            let mut reference = PitchMatcher::default();
            let mut times = Vec::new();
            for (step, (frequency, enabled, manual, target)) in [
                (0., true, 0., 255.),
                (140., true, 0., 255.),
                (280., true, 2., 255.),
                (0., true, 0., 255.),
                (140., false, 0., 255.),
                (140., true, -2., 155.),
            ]
            .into_iter()
            .enumerate()
            {
                if step == 4 || step == 5 {
                    graph.reset_states().unwrap();
                }
                let mut values = vec![frequency; n];
                if frequency > 0. {
                    values[0] = 0.;
                    values[n - 1] = 0.;
                    values[n / 2] = 0.;
                }
                let policy = AutoPitchConfig {
                    enabled,
                    target_hz: target,
                };
                let semitones = reference.resolve(&values, policy, manual, true);
                let raw_bytes = bytes(&values);
                let en = f32::from(enabled).to_le_bytes();
                let target = target.to_le_bytes();
                let manual = 2f32.powf(manual / 12.).to_le_bytes();
                let output = graph
                    .run(&[
                        ("raw", &raw_bytes),
                        ("manual", &manual),
                        ("auto_pitch_enabled", &en),
                        ("auto_pitch_target", &target),
                    ])
                    .unwrap();
                let actual = output.f32("scaled").unwrap();
                for (&x, &value) in actual.iter().zip(&values) {
                    assert!(
                        (x - value * 2f32.powf(semitones / 12.)).abs() < 0.001,
                        "n={n} step={step} actual={x} semitones={semitones}"
                    );
                }
                times.push(output.gpu_ms);
            }
            graph.reset_states().unwrap();
            let values = (0..n)
                .map(|i| if i % 3 == 0 { 0. } else { 100. + i as f32 * 2. })
                .collect::<Vec<_>>();
            let raw_bytes = bytes(&values);
            let out = graph
                .run(&[
                    ("raw", &raw_bytes),
                    ("manual", &1f32.to_le_bytes()),
                    ("auto_pitch_enabled", &1f32.to_le_bytes()),
                    ("auto_pitch_target", &255f32.to_le_bytes()),
                ])
                .unwrap();
            reference.reset();
            let expected = reference.resolve(&values, AutoPitchConfig::default(), 0., true);
            assert_eq!(out.f32("auto_pitch_offset").unwrap(), vec![expected]);
            eprintln!("DML matcher n={n} GPU ms={times:?}");
        }
    }
}

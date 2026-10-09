//! Register matching shared by CPU execution and GPU parity checks.
use crate::AutoPitchConfig;

/// Require 500 ms of voiced evidence in a normal realtime context. Short
/// profiles use half their context; silence never locks the calibration.
pub fn pitch_calibration_frames(frames: usize) -> usize {
    (frames / 2).clamp(2, 50)
}

/// Applio's median of the linearly interpolated F0 contour. The workspace is
/// retained by its owner; selection is linear time and does not sort all F0.
pub fn estimate_pitch_offset(raw: &[f32], target_hz: f32, scratch: &mut Vec<f32>) -> Option<f32> {
    if !target_hz.is_finite() || target_hz <= 0.0 {
        return None;
    }
    let valid = |x: f32| x.is_finite() && x > 0.0;
    let first = raw.iter().position(|&x| valid(x))?;
    let second = raw[first + 1..].iter().position(|&x| valid(x))? + first + 1;
    scratch.clear();
    scratch.resize(raw.len(), raw[first]);
    let mut left = first;
    for right in (second..raw.len()).filter(|&i| valid(raw[i])) {
        let step = (raw[right] - raw[left]) / (right - left) as f32;
        for (i, value) in scratch[left..right].iter_mut().enumerate() {
            *value = raw[left] + step * i as f32;
        }
        left = right;
    }
    scratch[left..].fill(raw[left]);
    let mid = scratch.len() / 2;
    let (lower, value, _) = scratch.select_nth_unstable_by(mid, f32::total_cmp);
    let median = if raw.len().is_multiple_of(2) {
        (*value
            + lower
                .iter()
                .copied()
                .max_by(f32::total_cmp)
                .unwrap_or(*value))
            * 0.5
    } else {
        *value
    };
    Some(
        (12.0 * (target_hz / median).log2())
            .round_ties_even()
            .clamp(-12.0, 12.0),
    )
}

#[derive(Default)]
pub struct PitchMatcher {
    scratch: Vec<f32>,
    config: Option<AutoPitchConfig>,
    offset: Option<f32>,
}
impl PitchMatcher {
    pub fn reset(&mut self) {
        self.offset = None;
    }
    pub fn offset(&self) -> Option<f32> {
        self.offset
    }
    pub fn resolve(
        &mut self,
        raw: &[f32],
        config: AutoPitchConfig,
        manual: f32,
        streaming: bool,
    ) -> f32 {
        if self.config != Some(config) {
            self.reset();
            self.config = Some(config);
        }
        if !config.enabled {
            return manual;
        }
        if !streaming || self.offset.is_none() {
            if !streaming {
                self.offset = None;
            }
            let voiced = raw.iter().filter(|&&x| x.is_finite() && x > 0.0).count();
            let minimum = if streaming {
                pitch_calibration_frames(raw.len())
            } else {
                2
            };
            if voiced >= minimum {
                self.offset = estimate_pitch_offset(raw, config.target_hz, &mut self.scratch);
            }
        }
        manual + self.offset.unwrap_or(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn interpolation_and_median_match_reference_and_do_not_change_raw() {
        let mut scratch = vec![];
        let raw = [0., 100., 0., 0., 200., 0.];
        assert_eq!(estimate_pitch_offset(&raw, 300., &mut scratch), Some(12.));
        assert_eq!(raw, [0., 100., 0., 0., 200., 0.]);
        assert_eq!(
            estimate_pitch_offset(&[0., 120., 0.], 255., &mut scratch),
            None
        );
        assert_eq!(estimate_pitch_offset(&[0.; 10], 255., &mut scratch), None);
        assert_eq!(
            estimate_pitch_offset(&[140.; 281], 255., &mut scratch),
            Some(10.)
        );
    }
    #[test]
    fn calibration_survives_silence_and_freezes_until_reset_or_policy_change() {
        let mut matcher = PitchMatcher::default();
        let config = AutoPitchConfig::default();
        assert_eq!(matcher.resolve(&[0.; 281], config, 0., true), 0.);
        assert_eq!(matcher.offset(), None);
        assert_eq!(matcher.resolve(&[140.; 281], config, 0., true), 10.);
        assert_eq!(matcher.resolve(&[280.; 281], config, 2., true), 12.);
        assert_eq!(matcher.resolve(&[0.; 281], config, 0., true), 10.);
        matcher.reset();
        assert_eq!(matcher.resolve(&[280.; 281], config, 0., true), -2.);
        assert_eq!(
            matcher.resolve(
                &[140.; 281],
                AutoPitchConfig {
                    target_hz: 155.,
                    ..config
                },
                0.,
                true
            ),
            2.
        );
        assert_eq!(
            matcher.resolve(&[140.; 281], AutoPitchConfig::disabled(), 3., true),
            3.
        );
    }
    #[test]
    fn old_configuration_enables_matching_and_invalid_targets_fail() {
        let config: crate::ConversionConfig = serde_json::from_str(
            r#"{"pitchSemitones":0,"indexRatio":0,"protect":0.33,"rmsMixRate":1}"#,
        )
        .unwrap();
        assert!(config.auto_pitch.enabled);
        let disabled: crate::ConversionConfig =
            serde_json::from_str(r#"{"autoPitch":{"enabled":false}}"#).unwrap();
        assert!(!disabled.auto_pitch.enabled);
        assert_eq!(disabled.auto_pitch.target_hz, 255.);
        for target_hz in [0., 49., 1201., f32::NAN] {
            assert!(
                AutoPitchConfig {
                    target_hz,
                    ..Default::default()
                }
                .validate()
                .is_err()
            );
        }
    }

    #[test]
    fn offline_silence_does_not_reuse_a_previous_segment_offset() {
        let mut matcher = PitchMatcher::default();
        assert_eq!(
            matcher.resolve(&[140.; 281], AutoPitchConfig::default(), 0., false),
            10.
        );
        assert_eq!(
            matcher.resolve(&[0.; 281], AutoPitchConfig::default(), 2., false),
            2.
        );
        assert_eq!(matcher.offset(), None);
    }
}

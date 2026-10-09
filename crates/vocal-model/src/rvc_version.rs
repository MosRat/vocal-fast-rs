//! RVC model versions and their learned acoustic feature contract.
//! These versions are independent of the `rvc-synth-v1` tensor ABI.
use crate::{ModelError, ModelPackage, RvcModelManifest, TensorDType};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RvcVersion {
    V1,
    #[default]
    V2,
}

/// The immutable feature contract shared by all graph backends. Keeping this
/// in the model crate prevents each backend from reinterpreting a version
/// string independently.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RvcFeatureSpec {
    pub version: RvcVersion,
    pub feature_width: usize,
    pub hubert_output_layer: usize,
    pub projection: Option<&'static str>,
}

impl RvcVersion {
    pub fn from_labels(version: &str, architecture: &str) -> Result<Self, ModelError> {
        let parsed = match version.trim().to_ascii_lowercase().as_str() {
            "v1" | "1" => Self::V1,
            "v2" | "2" => Self::V2,
            other => {
                return Err(ModelError::UnsupportedModel(format!(
                    "unsupported RVC version {other}"
                )));
            }
        };
        let architecture_version = architecture
            .split(|character: char| !character.is_ascii_alphanumeric())
            .filter_map(|part| match part.to_ascii_lowercase().as_str() {
                "v1" | "1" => Some(Self::V1),
                "v2" | "2" => Some(Self::V2),
                _ => None,
            })
            .try_fold(None, |found, candidate| {
                if found.is_some_and(|value| value != candidate) {
                    Err(())
                } else {
                    Ok(Some(candidate))
                }
            })
            .map_err(|_| {
                ModelError::UnsupportedModel(
                    "RVC architecture contains conflicting version labels".into(),
                )
            })?;
        if architecture_version.is_some_and(|version| version != parsed) {
            return Err(ModelError::UnsupportedModel(
                "RVC architecture disagrees with model version".into(),
            ));
        }
        Ok(parsed)
    }

    pub const fn feature_spec(self) -> RvcFeatureSpec {
        match self {
            Self::V1 => RvcFeatureSpec {
                version: Self::V1,
                feature_width: 256,
                hubert_output_layer: 9,
                projection: Some("hubert.model.final_proj"),
            },
            Self::V2 => RvcFeatureSpec {
                version: Self::V2,
                feature_width: 768,
                hubert_output_layer: 12,
                projection: None,
            },
        }
    }

    pub fn from_feature_width(width: u64) -> Result<Self, ModelError> {
        match width {
            256 => Ok(Self::V1),
            768 => Ok(Self::V2),
            _ => Err(ModelError::UnsupportedModel(format!(
                "unsupported RVC feature width {width}"
            ))),
        }
    }
    pub fn feature_width(self) -> usize {
        self.feature_spec().feature_width
    }
    pub fn hubert_output_layer(self) -> usize {
        self.feature_spec().hubert_output_layer
    }
    pub fn as_str(self) -> &'static str {
        match self {
            Self::V1 => "v1",
            Self::V2 => "v2",
        }
    }
}

impl RvcModelManifest {
    /// Reject conflicting/unknown versions instead of silently choosing v1.
    pub fn rvc_version(&self) -> Result<RvcVersion, ModelError> {
        RvcVersion::from_labels(&self.version, &self.architecture)
    }
}

/// Older canonical packages lack this field; declared feature contracts must
/// agree with the manifest and the learned Synth input width.
pub(crate) fn validate_feature_metadata(
    metadata: &serde_json::Value,
    version: RvcVersion,
) -> Result<(), ModelError> {
    #[derive(Deserialize)]
    struct Contract {
        version: RvcVersion,
        width: usize,
        hubert_output_layer: usize,
        hubert_projection: Option<String>,
    }
    let Some(value) = metadata.get("rvc_features") else {
        return Ok(());
    };
    let contract: Contract = serde_json::from_value(value.clone())?;
    let projection = (version == RvcVersion::V1).then_some("hubert.model.final_proj");
    if contract.version != version
        || contract.width != version.feature_width()
        || contract.hubert_output_layer != version.hubert_output_layer()
        || contract.hubert_projection.as_deref() != projection
    {
        return Err(ModelError::InvalidCache(
            "RVC feature metadata disagrees with model version".into(),
        ));
    }
    Ok(())
}

/// Check canonical layouts without reading tensor bytes; used by bundle inspection too.
pub(crate) fn validate_hubert_feature_tensors<'a>(
    version: RvcVersion,
    tensors: impl IntoIterator<Item = (&'a str, &'a [u64], TensorDType)>,
) -> Result<(), ModelError> {
    if version == RvcVersion::V2 {
        return Ok(());
    }
    let mut weight = false;
    let mut bias = false;
    for (name, shape, dtype) in tensors {
        let expected: &[u64] = match name {
            "hubert.model.final_proj.weight" => {
                weight = true;
                &[768, 256]
            }
            "hubert.model.final_proj.bias" => {
                bias = true;
                &[256]
            }
            _ => continue,
        };
        if shape != expected
            || !matches!(
                dtype,
                TensorDType::F16 | TensorDType::F32 | TensorDType::BF16
            )
        {
            return Err(ModelError::InvalidCache(format!(
                "invalid RVC v1 projection {name}: expected {expected:?} floating-point canonical layout"
            )));
        }
    }
    if !weight || !bias {
        return Err(ModelError::UnsupportedModel("RVC v1 requires learned HuBERT final_proj weights; reconvert HuBERT with --hubert-checkpoint (256 dimensions cannot be obtained by slicing v2 features)".into()));
    }
    Ok(())
}

impl ModelPackage {
    pub fn validate_hubert_for_rvc(&self, version: RvcVersion) -> Result<(), ModelError> {
        let config = self.validate_acoustic(Some(crate::AcousticKind::Hubert))?;
        config.validate_rvc_version(version)?;
        validate_hubert_feature_tensors(
            version,
            self.tensors
                .iter()
                .map(|t| (t.name.as_str(), t.shape.as_slice(), t.dtype)),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn versions_select_distinct_layers_and_dimensions() {
        assert_eq!(
            (
                RvcVersion::V1.hubert_output_layer(),
                RvcVersion::V1.feature_width()
            ),
            (9, 256)
        );
        assert_eq!(
            (
                RvcVersion::V2.hubert_output_layer(),
                RvcVersion::V2.feature_width()
            ),
            (12, 768)
        );
        assert!(RvcVersion::from_feature_width(512).is_err());
    }
    #[test]
    fn projection_requires_both_weights_in_canonical_layout() {
        let weight = (
            "hubert.model.final_proj.weight",
            &[768, 256][..],
            TensorDType::F32,
        );
        let bias = ("hubert.model.final_proj.bias", &[256][..], TensorDType::F16);
        assert!(validate_hubert_feature_tensors(RvcVersion::V1, [weight, bias]).is_ok());
        assert!(validate_hubert_feature_tensors(RvcVersion::V1, [weight]).is_err());
        assert!(
            validate_hubert_feature_tensors(
                RvcVersion::V1,
                [(weight.0, &[256, 768][..], weight.2), bias]
            )
            .is_err()
        );
        assert!(validate_hubert_feature_tensors(RvcVersion::V2, []).is_ok());
    }

    #[test]
    fn declared_feature_metadata_must_match_version() {
        let mut metadata = serde_json::json!({"rvc_features": {
            "version":"v1", "width":256, "hubert_output_layer":9,
            "hubert_projection":"hubert.model.final_proj"
        }});
        assert!(validate_feature_metadata(&metadata, RvcVersion::V1).is_ok());
        assert!(validate_feature_metadata(&metadata, RvcVersion::V2).is_err());
        metadata["rvc_features"]["hubert_output_layer"] = 12.into();
        assert!(validate_feature_metadata(&metadata, RvcVersion::V1).is_err());
        assert!(validate_feature_metadata(&serde_json::json!({}), RvcVersion::V1).is_ok());
    }

    #[test]
    fn labels_accept_common_separators_and_reject_conflicts() {
        assert_eq!(
            RvcVersion::from_labels("1", "RVC_v1_f0").unwrap(),
            RvcVersion::V1
        );
        assert_eq!(
            RvcVersion::from_labels("v2", "rvc-v2-nono").unwrap(),
            RvcVersion::V2
        );
        assert!(RvcVersion::from_labels("v1", "rvc-v2").is_err());
        assert!(RvcVersion::from_labels("v1", "rvc-v1-v2").is_err());
    }
}

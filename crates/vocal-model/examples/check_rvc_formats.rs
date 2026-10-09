//! Verify that real PTH and ONNX sources normalize to the same tensor ABI.
use vocal_model::{Tensor, TensorDType};
fn floats(t: &Tensor) -> Vec<f32> {
    match t.dtype {
        TensorDType::F32 => t
            .data
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect(),
        TensorDType::F16 => t
            .data
            .chunks_exact(2)
            .map(|b| half::f16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).to_f32())
            .collect(),
        TensorDType::BF16 => t
            .data
            .chunks_exact(2)
            .map(|b| half::bf16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).to_f32())
            .collect(),
        _ => panic!("non-floating-point model parameter"),
    }
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.len() != 3 {
        return Err("check_rvc_formats checkpoint.pth synth.onnx report.json".into());
    }
    let pth = vocal_model::load_rvc_pth_auto(&args[0], "Format parity", "validation")?;
    let onnx = vocal_model::load_rvc_onnx(&args[1], pth.manifest.clone())?;
    let pth_config = pth.validate_rvc_synth()?;
    let onnx_config = onnx.validate_rvc_synth()?;
    if serde_json::to_value(&pth_config)? != serde_json::to_value(&onnx_config)? {
        return Err("source configs differ".into());
    }
    if pth.metadata["graph_schema"] != onnx.metadata["graph_schema"]
        || pth.tensors.len() != onnx.tensors.len()
    {
        return Err("source graph schema or tensor count differs".into());
    }
    let mut rows = Vec::new();
    let mut max = 0f32;
    for a in &pth.tensors {
        let b = onnx
            .tensor(&a.name)
            .ok_or("canonical tensor missing from ONNX")?;
        if a.shape != b.shape {
            return Err(format!("tensor layout differs: {}", a.name).into());
        }
        let lhs = floats(a);
        let rhs = floats(b);
        let error = lhs
            .iter()
            .zip(&rhs)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        max = max.max(error);
        rows.push(
            serde_json::json!({"name":a.name, "shape":a.shape, "max_abs":error,
            "pth_dtype":format!("{:?}",a.dtype), "onnx_dtype":format!("{:?}",b.dtype)}),
        );
    }
    let report = serde_json::json!({"version":pth.manifest.rvc_version()?, "feature_width":pth_config.feature_width,
        "graph_schema":pth.metadata["graph_schema"], "tensor_count":rows.len(), "max_weight_abs":max, "tensors":rows});
    std::fs::write(&args[2], serde_json::to_vec_pretty(&report)?)?;
    vocal_model::write_cache(std::path::Path::new(&args[1]).with_extension("vfrm"), &onnx)?;
    println!(
        "{} tensors share canonical names, layouts and configuration; max weight difference {max}",
        pth.tensors.len()
    );
    Ok(())
}

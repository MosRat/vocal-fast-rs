use vocal_graph::{Precision, graph::GraphBuilder};
use vocal_model::{ModelPackage, Tensor, TensorDType};

#[test]
fn converted_cache_parameters_keep_canonical_recipe_names() {
    for dtype in [TensorDType::F16, TensorDType::BF16] {
        let package = ModelPackage {
            manifest: serde_json::from_value(serde_json::json!({
                "id":"test", "display_name":"test", "architecture":"rvc-v2",
                "version":"v2", "sample_rate":40000, "model_file":"test.vfrm",
                "index_file":null, "source":"test", "notes":null
            }))
            .unwrap(),
            metadata: serde_json::Value::Null,
            tensors: vec![Tensor {
                name: "enc_p.emb_phone.weight".into(),
                dtype,
                shape: vec![2, 2],
                data: vec![0; 8],
            }],
        };
        let graph = GraphBuilder::default();
        let tensor = graph
            .parameter(&package, "enc_p.emb_phone.weight", Precision::Fp32)
            .unwrap();
        assert_eq!(
            graph.nodes.borrow()[tensor.id as usize].name,
            "enc_p.emb_phone.weight"
        );
        assert_eq!(
            graph
                .parameter(&package, "enc_p.emb_phone.weight", Precision::Fp32)
                .unwrap()
                .id,
            tensor.id
        );
    }
}

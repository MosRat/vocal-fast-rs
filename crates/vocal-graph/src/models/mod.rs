//! Handwritten inference architectures shared by native execution backends.
mod hubert;
mod retrieval;
mod rmvpe;
mod synth;
pub use hubert::{hubert, hubert_for_rvc};
pub use retrieval::retrieval;
pub use rmvpe::{rmvpe, rmvpe_from_mel};
pub use synth::synth;

use crate::{
    GraphError, Precision,
    graph::{DType, GraphBuilder, Op, Tensor},
};
use vocal_model::ModelPackage;
pub struct Layers<'a> {
    pub g: &'a GraphBuilder,
    pub package: &'a ModelPackage,
}
impl Layers<'_> {
    pub fn parameter(&self, name: &str, p: Precision) -> Result<Tensor, GraphError> {
        self.g.parameter(self.package, name, p)
    }
    pub fn linear(&self, x: &Tensor, prefix: &str, p: Precision) -> Result<Tensor, GraphError> {
        let w = self.parameter(&format!("{prefix}.weight"), p)?;
        let b = self.parameter(&format!("{prefix}.bias"), p)?;
        Ok(self
            .g
            .add(&self.g.matmul(&self.g.cast(x, p.into()), &w, false), &b))
    }
    pub fn conv1(
        &self,
        x: &Tensor,
        prefix: &str,
        p: Precision,
        stride: usize,
        dilation: usize,
        padding: Option<usize>,
        deconv: bool,
        bias: bool,
        weight_scale: f32,
    ) -> Result<Tensor, GraphError> {
        let w = self.parameter(&format!("{prefix}.weight"), p)?;
        let w = if weight_scale == 1. {
            w
        } else {
            let scaled = crate::graph::weight_f32(self.package, &format!("{prefix}.weight"))?
                .into_iter()
                .map(|v| v * weight_scale)
                .collect::<Vec<_>>();
            self.g.constant(&scaled, &w.shape, p)
        };
        let rank = w.shape.len();
        let k = w.shape[rank - 1];
        let pad = padding.unwrap_or((k * dilation - dilation) / 2);
        let w4 = self.g.reshape(&w, &[w.shape[0], w.shape[1], 1, k]);
        let x = self.g.cast(x, p.into());
        let x4 = self.g.reshape(&x, &[1, x.shape[1], 1, x.shape[2]]);
        let b = if bias {
            let b = self.parameter(&format!("{prefix}.bias"), p)?;
            Some(self.g.reshape(&b, &[1, b.elements(), 1, 1]))
        } else {
            None
        };
        let y = self.g.conv(
            &x4,
            &w4,
            b.as_ref(),
            [1, stride],
            [1, dilation],
            [0, pad],
            1,
            deconv,
            [0, 0],
        );
        self.g.nodes.borrow_mut()[y.id as usize].name = prefix.into();
        Ok(self.g.reshape(&y, &[1, y.shape[1], y.shape[3]]))
    }
    pub fn conv2(
        &self,
        x: &Tensor,
        prefix: &str,
        p: Precision,
        stride: usize,
        padding: usize,
        deconv: bool,
        output_pad: usize,
        bias: bool,
    ) -> Result<Tensor, GraphError> {
        let w = self.parameter(&format!("{prefix}.weight"), p)?;
        let b = if bias {
            let b = self.parameter(&format!("{prefix}.bias"), p)?;
            Some(self.g.reshape(&b, &[1, b.elements(), 1, 1]))
        } else {
            None
        };
        let y = self.g.conv(
            &self.g.cast(x, p.into()),
            &w,
            b.as_ref(),
            [stride; 2],
            [1; 2],
            [padding; 2],
            1,
            deconv,
            [output_pad; 2],
        );
        self.g.nodes.borrow_mut()[y.id as usize].name = prefix.into();
        Ok(y)
    }
    pub fn norm(
        &self,
        x: &Tensor,
        prefix: &str,
        axis: usize,
        epsilon: f32,
        p: Precision,
        gamma: &str,
        beta: &str,
    ) -> Result<Tensor, GraphError> {
        let fp = self.g.cast(x, DType::F32);
        let center = self.g.sub(&fp, &self.g.reduce(Op::Mean, &fp, &[axis]));
        let variance = self
            .g
            .reduce(Op::Mean, &self.g.mul(&center, &center), &[axis]);
        let y = self.g.div(
            &center,
            &self.g.unary(
                Op::Sqrt,
                &self
                    .g
                    .add(&variance, &self.g.scalar(epsilon, Precision::Fp32)),
            ),
        );
        let mut shape = vec![1; x.shape.len()];
        shape[axis] = x.shape[axis];
        let w = self.g.reshape(
            &self.parameter(&format!("{prefix}.{gamma}"), Precision::Fp32)?,
            &shape,
        );
        let b = self.g.reshape(
            &self.parameter(&format!("{prefix}.{beta}"), Precision::Fp32)?,
            &shape,
        );
        Ok(self.g.cast(&self.g.add(&self.g.mul(&y, &w), &b), p.into()))
    }
}

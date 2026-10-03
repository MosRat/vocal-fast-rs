//! A non-executing, bounded pickle VM for PyTorch RVC state dictionaries.
//! GLOBAL/REDUCE are symbolic descriptions; no module or callable is invoked.
use crate::{ModelError, TensorDType};
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use zip::ZipArchive;

pub(crate) fn bad(message: impl Into<String>) -> ModelError {
    ModelError::UnsupportedModel(message.into())
}
pub(crate) type Id = usize;

#[derive(Clone, Debug)]
pub(crate) struct Storage {
    pub key: String,
    pub dtype: TensorDType,
    pub elements: u64,
}
#[derive(Clone, Debug)]
pub(crate) struct View {
    pub storage: Storage,
    pub offset: u64,
    pub shape: Vec<u64>,
    pub stride: Vec<u64>,
}
#[derive(Clone, Debug)]
pub(crate) enum Value {
    Null,
    Bool(bool),
    Int(i128),
    Float(f64),
    String(String),
    Global(String),
    List(Vec<Id>),
    Tuple(Vec<Id>),
    Dict(Vec<(Id, Id)>),
    Storage(Storage),
    Tensor(View),
}

pub(crate) fn load_rvc_pth(path: &Path) -> Result<(JsonValue, Vec<crate::Tensor>), ModelError> {
    let file = File::open(path)?;
    let mut archive =
        ZipArchive::new(file).map_err(|e| bad(format!("invalid PyTorch ZIP: {e}")))?;
    let pickle_name = (0..archive.len())
        .find_map(|i| {
            archive.by_index(i).ok().and_then(|entry| {
                entry
                    .name()
                    .ends_with("/data.pkl")
                    .then(|| entry.name().to_owned())
            })
        })
        .ok_or_else(|| bad("RVC checkpoint is not a modern PyTorch ZIP (data.pkl missing)"))?;
    let mut pickle_bytes = Vec::new();
    archive
        .by_name(&pickle_name)
        .map_err(|e| bad(format!("cannot read data.pkl: {e}")))?
        .read_to_end(&mut pickle_bytes)?;
    let parsed = parse(&pickle_bytes)?;
    let root = parsed.dict(parsed.root)?;
    let weight_id = *root
        .get("weight")
        .ok_or_else(|| bad("RVC checkpoint has no weight state_dict"))?;
    let weight = parsed.dict(weight_id)?;
    let mut tensors = Vec::with_capacity(weight.len());
    for (name, value_id) in weight {
        let Value::Tensor(view) = &parsed.arena[value_id] else {
            return Err(bad(format!("RVC weight {name} is not a tensor")));
        };
        tensors.push(materialize_tensor(name, view, &mut archive)?);
    }
    let mut metadata = serde_json::Map::new();
    for (key, value_id) in root {
        if key == "weight" {
            continue;
        }
        metadata.insert(key, parsed.json(value_id, 0)?);
    }
    metadata.insert("source_format".into(), "rvc-pytorch-checkpoint".into());
    metadata.insert("tensor_count".into(), tensors.len().into());
    Ok((JsonValue::Object(metadata), tensors))
}

fn materialize_tensor(
    name: String,
    view: &View,
    archive: &mut ZipArchive<File>,
) -> Result<crate::Tensor, ModelError> {
    let count = view
        .shape
        .iter()
        .try_fold(1_u64, |a, &b| a.checked_mul(b))
        .ok_or_else(|| bad("RVC tensor shape overflows"))?;
    if count > 4 * 1024 * 1024 * 1024 {
        return Err(bad("RVC tensor is too large"));
    }
    let expected_stride = contiguous_stride(&view.shape);
    if view.stride != expected_stride {
        return Err(bad(format!("RVC tensor {name} is a non-contiguous view")));
    }
    let storage_name = (0..archive.len())
        .find_map(|i| {
            archive.by_index(i).ok().and_then(|entry| {
                let suffix = format!("/data/{}", view.storage.key);
                (entry.name().ends_with(&suffix)
                    || entry.name() == format!("data/{}", view.storage.key))
                .then(|| entry.name().to_owned())
            })
        })
        .ok_or_else(|| bad(format!("storage {} is missing", view.storage.key)))?;
    let mut storage = Vec::new();
    archive
        .by_name(&storage_name)
        .map_err(|e| bad(format!("cannot read storage: {e}")))?
        .read_to_end(&mut storage)?;
    let element_size = view.storage.dtype.element_size();
    let expected_storage = view
        .storage
        .elements
        .checked_mul(element_size as u64)
        .ok_or_else(|| bad("RVC storage size overflows"))?;
    if storage.len() as u64 != expected_storage {
        return Err(bad(format!(
            "storage {} has {} bytes, expected {}",
            view.storage.key,
            storage.len(),
            expected_storage
        )));
    }
    let end = view
        .offset
        .checked_add(count)
        .ok_or_else(|| bad("RVC tensor offset overflows"))?;
    if end > view.storage.elements {
        return Err(bad(format!("RVC tensor {name} exceeds its storage")));
    }
    let start = view.offset as usize * element_size;
    let len = count as usize * element_size;
    Ok(crate::Tensor {
        name,
        dtype: view.storage.dtype,
        shape: view.shape.clone(),
        data: storage[start..start + len].to_vec(),
    })
}

fn contiguous_stride(shape: &[u64]) -> Vec<u64> {
    let mut stride = vec![1_u64; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        stride[i] = stride[i + 1].saturating_mul(shape[i + 1]);
    }
    stride
}

pub(crate) struct Pickle {
    pub arena: Vec<Value>,
    pub root: Id,
}

impl Pickle {
    pub fn dict(&self, id: Id) -> Result<HashMap<String, Id>, ModelError> {
        let Value::Dict(pairs) = &self.arena[id] else {
            return Err(bad("expected dictionary"));
        };
        let mut out = HashMap::new();
        for &(key, value) in pairs {
            if out.insert(self.string(key)?.to_owned(), value).is_some() {
                return Err(bad("duplicate dictionary key"));
            }
        }
        Ok(out)
    }
    pub fn string(&self, id: Id) -> Result<&str, ModelError> {
        match &self.arena[id] {
            Value::String(s) => Ok(s),
            _ => Err(bad("expected string")),
        }
    }
    pub fn json(&self, id: Id, depth: usize) -> Result<serde_json::Value, ModelError> {
        if depth > 32 {
            return Err(bad("recursive or excessively nested pickle metadata"));
        }
        Ok(match &self.arena[id] {
            Value::Null => serde_json::Value::Null,
            Value::Bool(v) => (*v).into(),
            Value::Int(v) => i64::try_from(*v)
                .map_err(|_| bad("metadata integer out of range"))?
                .into(),
            Value::Float(v) => serde_json::Number::from_f64(*v)
                .ok_or_else(|| bad("non-finite metadata"))?
                .into(),
            Value::String(s) => s.clone().into(),
            Value::List(ids) | Value::Tuple(ids) => {
                if ids.len() > 4096 {
                    return Err(bad("metadata list too long"));
                }
                serde_json::Value::Array(
                    ids.iter()
                        .map(|&v| self.json(v, depth + 1))
                        .collect::<Result<_, _>>()?,
                )
            }
            Value::Dict(_) => {
                let dict = self.dict(id)?;
                if dict.len() > 256 {
                    return Err(bad("metadata dictionary too large"));
                }
                let mut out = serde_json::Map::new();
                for (key, value) in dict {
                    out.insert(key, self.json(value, depth + 1)?);
                }
                out.into()
            }
            _ => return Err(bad("unsupported metadata object")),
        })
    }
}

struct Vm<'a> {
    bytes: &'a [u8],
    pos: usize,
    arena: Vec<Value>,
    stack: Vec<Id>,
    marks: Vec<usize>,
    memo: HashMap<u32, Id>,
}

pub(crate) fn parse(bytes: &[u8]) -> Result<Pickle, ModelError> {
    let mut vm = Vm {
        bytes,
        pos: 0,
        arena: vec![],
        stack: vec![],
        marks: vec![],
        memo: HashMap::new(),
    };
    for _ in 0..2_000_000 {
        if vm.arena.len() > 200_000 || vm.stack.len() > 100_000 || vm.marks.len() > 64 {
            return Err(bad("pickle resource limit"));
        }
        let opcode = vm.byte()?;
        match opcode {
            0x80 => {
                let version = vm.byte()?;
                if !(2..=5).contains(&version) {
                    return Err(bad("unsupported pickle protocol"));
                }
            }
            0x95 => {
                let size = vm.uint(8)?;
                if size > (bytes.len() - vm.pos) as u64 {
                    return Err(bad("invalid pickle frame length"));
                }
            }
            b'.' => {
                if vm.stack.len() != 1 || !vm.marks.is_empty() {
                    return Err(bad("invalid pickle STOP stack"));
                }
                return Ok(Pickle {
                    root: vm.stack[0],
                    arena: vm.arena,
                });
            }
            b'(' => vm.marks.push(vm.stack.len()),
            b'N' => vm.push(Value::Null),
            0x88 => vm.push(Value::Bool(true)),
            0x89 => vm.push(Value::Bool(false)),
            b'K' => {
                let n = vm.uint(1)?;
                vm.push(Value::Int(n as i128));
            }
            b'M' => {
                let n = vm.uint(2)?;
                vm.push(Value::Int(n as i128));
            }
            b'J' => {
                let n = vm.uint(4)? as u32 as i32;
                vm.push(Value::Int(n as i128));
            }
            0x8a | 0x8b => {
                let len = vm.uint(if opcode == 0x8a { 1 } else { 4 })? as usize;
                if len > 16 {
                    return Err(bad("pickle integer exceeds i128"));
                }
                let raw = vm.take(len)?;
                let mut n = [if raw.last().is_some_and(|b| b & 128 != 0) {
                    255
                } else {
                    0
                }; 16];
                n[..len].copy_from_slice(raw);
                vm.push(Value::Int(i128::from_le_bytes(n)));
            }
            b'G' => {
                let raw: [u8; 8] = vm.take(8)?.try_into().unwrap();
                vm.push(Value::Float(f64::from_be_bytes(raw)));
            }
            b'X' | 0x8c | 0x8d => {
                let len = vm.uint(match opcode {
                    b'X' => 4,
                    0x8c => 1,
                    _ => 8,
                })?;
                if len > 65535 {
                    return Err(bad("pickle string too long"));
                }
                let s = std::str::from_utf8(vm.take(len as usize)?)
                    .map_err(|_| bad("pickle string is not UTF-8"))?
                    .to_owned();
                vm.push(Value::String(s));
            }
            b'c' => {
                let module = vm.line()?;
                let name = vm.line()?;
                vm.global(format!("{module}.{name}"))?;
            }
            0x93 => {
                let name = vm.pop_string()?;
                let module = vm.pop_string()?;
                vm.global(format!("{module}.{name}"))?;
            }
            b']' => vm.push(Value::List(vec![])),
            b')' => vm.push(Value::Tuple(vec![])),
            b'}' => vm.push(Value::Dict(vec![])),
            b't' | b'l' => {
                let ids = vm.marked()?;
                vm.push(if opcode == b't' {
                    Value::Tuple(ids)
                } else {
                    Value::List(ids)
                });
            }
            0x85..=0x87 => {
                let count = (opcode - 0x84) as usize;
                if vm.stack.len() < count {
                    return Err(bad("pickle stack underflow"));
                }
                let ids = vm.stack.split_off(vm.stack.len() - count);
                vm.push(Value::Tuple(ids));
            }
            b'a' | b'e' => {
                let values = if opcode == b'a' {
                    vec![vm.pop()?]
                } else {
                    vm.marked()?
                };
                let id = vm.top()?;
                let Value::List(list) = &mut vm.arena[id] else {
                    return Err(bad("APPEND target is not list"));
                };
                list.extend(values);
            }
            b's' | b'u' | b'd' => {
                let values = if opcode == b's' {
                    let v = vm.pop()?;
                    let k = vm.pop()?;
                    vec![k, v]
                } else {
                    vm.marked()?
                };
                if values.len() % 2 != 0 {
                    return Err(bad("odd pickle dictionary items"));
                }
                if opcode == b'd' {
                    vm.push(Value::Dict(vec![]));
                }
                let id = vm.top()?;
                let Value::Dict(dict) = &mut vm.arena[id] else {
                    return Err(bad("SETITEM target is not dictionary"));
                };
                dict.extend(values.chunks(2).map(|p| (p[0], p[1])));
            }
            b'q' | b'r' => {
                let index = vm.uint(if opcode == b'q' { 1 } else { 4 })? as u32;
                let top = vm.top()?;
                if vm.memo.insert(index, top).is_some() {
                    return Err(bad("pickle memo overwrite"));
                }
            }
            0x94 => {
                let top = vm.top()?;
                let index = vm.memo.len() as u32;
                if vm.memo.insert(index, top).is_some() {
                    return Err(bad("pickle memo collision"));
                }
            }
            b'h' | b'j' => {
                let index = vm.uint(if opcode == b'h' { 1 } else { 4 })? as u32;
                let id = *vm
                    .memo
                    .get(&index)
                    .ok_or_else(|| bad("invalid pickle memo reference"))?;
                vm.stack.push(id);
            }
            b'Q' => {
                let id = vm.pop()?;
                let storage = vm.storage(id)?;
                vm.push(Value::Storage(storage));
            }
            b'R' => vm.reduce()?,
            b'b' => {
                // Only harmless OrderedDict state_dict metadata is accepted.
                let state = vm.pop()?;
                let id = vm.top()?;
                if !matches!(vm.arena[id], Value::Dict(_))
                    || !matches!(vm.arena[state], Value::Dict(_))
                {
                    return Err(bad("unsupported pickle BUILD"));
                }
                let p = Pickle {
                    arena: vm.arena.clone(),
                    root: state,
                };
                let dict = p.dict(state)?;
                if dict.keys().any(|k| k != "_metadata") {
                    return Err(bad("unsupported OrderedDict attributes"));
                }
                for (_, v) in dict {
                    p.json(v, 0)?;
                }
            }
            _ => {
                return Err(bad(format!(
                    "unsupported pickle opcode 0x{opcode:02x} at {}",
                    vm.pos - 1
                )));
            }
        }
    }
    Err(bad("pickle instruction limit"))
}

impl Vm<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8], ModelError> {
        let end = self
            .pos
            .checked_add(n)
            .filter(|&e| e <= self.bytes.len())
            .ok_or_else(|| bad("truncated pickle"))?;
        let slice = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(slice)
    }
    fn byte(&mut self) -> Result<u8, ModelError> {
        Ok(self.take(1)?[0])
    }
    fn uint(&mut self, n: usize) -> Result<u64, ModelError> {
        let mut b = [0; 8];
        b[..n].copy_from_slice(self.take(n)?);
        Ok(u64::from_le_bytes(b))
    }
    fn line(&mut self) -> Result<String, ModelError> {
        let n = self.bytes[self.pos..]
            .iter()
            .position(|b| *b == b'\n')
            .filter(|&n| n < 256)
            .ok_or_else(|| bad("invalid GLOBAL string"))?;
        let s = std::str::from_utf8(self.take(n)?)
            .map_err(|_| bad("invalid GLOBAL UTF-8"))?
            .to_owned();
        self.byte()?;
        Ok(s)
    }
    fn push(&mut self, v: Value) {
        self.stack.push(self.arena.len());
        self.arena.push(v);
    }
    fn pop(&mut self) -> Result<Id, ModelError> {
        self.stack
            .pop()
            .ok_or_else(|| bad("pickle stack underflow"))
    }
    fn top(&self) -> Result<Id, ModelError> {
        self.stack
            .last()
            .copied()
            .ok_or_else(|| bad("pickle stack underflow"))
    }
    fn pop_string(&mut self) -> Result<String, ModelError> {
        let id = self.pop()?;
        self.string(id).map(str::to_owned)
    }
    fn string(&self, id: Id) -> Result<&str, ModelError> {
        match &self.arena[id] {
            Value::String(s) => Ok(s),
            _ => Err(bad("expected string")),
        }
    }
    fn integer(&self, id: Id) -> Result<u64, ModelError> {
        match self.arena[id] {
            Value::Int(n) => u64::try_from(n).map_err(|_| bad("negative/oversized tensor index")),
            _ => Err(bad("expected integer")),
        }
    }
    fn tuple(&self, id: Id) -> Result<&[Id], ModelError> {
        match &self.arena[id] {
            Value::Tuple(ids) => Ok(ids),
            _ => Err(bad("expected tuple")),
        }
    }
    fn dimensions(&self, id: Id) -> Result<Vec<u64>, ModelError> {
        let ids = self.tuple(id)?;
        if ids.len() > 8 {
            return Err(bad("tensor rank exceeds 8"));
        }
        ids.iter().map(|&i| self.integer(i)).collect()
    }
    fn marked(&mut self) -> Result<Vec<Id>, ModelError> {
        let start = self
            .marks
            .pop()
            .filter(|&s| s <= self.stack.len())
            .ok_or_else(|| bad("invalid pickle MARK"))?;
        Ok(self.stack.split_off(start))
    }
    fn global(&mut self, name: String) -> Result<(), ModelError> {
        if !matches!(
            name.as_str(),
            "collections.OrderedDict"
                | "torch._utils._rebuild_tensor_v2"
                | "torch._utils._rebuild_tensor"
                | "torch.HalfStorage"
                | "torch.FloatStorage"
                | "torch.BFloat16Storage"
                | "torch.LongStorage"
                | "torch.IntStorage"
                | "torch.ByteStorage"
        ) {
            return Err(bad(format!("pickle GLOBAL {name} is not allowed")));
        }
        self.push(Value::Global(name));
        Ok(())
    }
    fn storage(&self, id: Id) -> Result<Storage, ModelError> {
        let ids = self.tuple(id)?;
        if !(ids.len() == 5 || ids.len() == 6) || self.string(ids[0])? != "storage" {
            return Err(bad("unsupported persistent storage ID"));
        }
        if ids.len() == 6 && !matches!(self.arena[ids[5]], Value::Null) {
            return Err(bad("legacy storage views are unsupported"));
        }
        let Value::Global(kind) = &self.arena[ids[1]] else {
            return Err(bad("unsupported storage class"));
        };
        let dtype = match kind.as_str() {
            "torch.HalfStorage" => TensorDType::F16,
            "torch.FloatStorage" => TensorDType::F32,
            "torch.BFloat16Storage" => TensorDType::BF16,
            "torch.LongStorage" => TensorDType::I64,
            "torch.IntStorage" => TensorDType::I32,
            "torch.ByteStorage" => TensorDType::U8,
            _ => return Err(bad("unsupported storage dtype")),
        };
        let key = self.string(ids[2])?.to_owned();
        if key.is_empty() || !key.bytes().all(|c| c.is_ascii_digit()) {
            return Err(bad("invalid storage key"));
        }
        let device = self.string(ids[3])?;
        if device != "cpu"
            && !device
                .strip_prefix("cuda:")
                .is_some_and(|s| !s.is_empty() && s.bytes().all(|c| c.is_ascii_digit()))
        {
            return Err(bad("unsupported storage location"));
        }
        Ok(Storage {
            key,
            dtype,
            elements: self.integer(ids[4])?,
        })
    }
    fn reduce(&mut self) -> Result<(), ModelError> {
        let args_id = self.pop()?;
        let callable = self.pop()?;
        let Value::Global(name) = &self.arena[callable] else {
            return Err(bad("non-GLOBAL pickle reducer"));
        };
        let args = self.tuple(args_id)?.to_vec();
        match name.as_str() {
            "collections.OrderedDict" if args.is_empty() => self.push(Value::Dict(vec![])),
            "torch._utils._rebuild_tensor" | "torch._utils._rebuild_tensor_v2" => {
                let count = if name.ends_with("_v2") { 6 } else { 4 };
                if args.len() != count {
                    return Err(bad("unexpected tensor rebuild arguments"));
                }
                if count == 6
                    && (!matches!(self.arena[args[4]], Value::Bool(_))
                        || !matches!(&self.arena[args[5]], Value::Dict(v) if v.is_empty()))
                {
                    return Err(bad("tensor hooks/attributes are unsupported"));
                }
                let Value::Storage(storage) = &self.arena[args[0]] else {
                    return Err(bad("tensor has no storage"));
                };
                let v = View {
                    storage: storage.clone(),
                    offset: self.integer(args[1])?,
                    shape: self.dimensions(args[2])?,
                    stride: self.dimensions(args[3])?,
                };
                if v.shape.len() != v.stride.len() {
                    return Err(bad("tensor shape/stride rank mismatch"));
                }
                self.push(Value::Tensor(v));
            }
            _ => return Err(bad(format!("unsupported reducer {name}"))),
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_executable_global() {
        assert!(parse(b"\x80\x02cos\nsystem\n.").is_err());
    }
    #[test]
    fn rejects_bad_memo_and_truncation() {
        assert!(parse(b"\x80\x02h\x01.").is_err());
        assert!(parse(b"\x80\x02X\xff\xff\xff\x7f").is_err());
    }
}

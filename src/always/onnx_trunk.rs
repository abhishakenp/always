//! Splits the pinned WeSpeaker ResNet34 graph into its convolutional trunk
//! and its pooling tail, in memory, at load (no protobuf dependency: the
//! few ONNX fields involved are walked by hand).
//!
//! Why: the trunk runs as a CoreML ML Program on the GPU in fp32 — the same
//! scores as the CPU session (max |Δscore| 8e-7 over 547 ladder windows,
//! 2026-10-19) for an eighth of its CPU — but the graph's tail (std pooling
//! via `Shape → Gather → ReduceProd → Cast`) comes out NaN on that path.
//! So the session gets the trunk only and `speaker_embed::PoolingTail` does
//! the tail in Rust, op for op.

use anyhow::{Context, Result, bail};

/// Output of the last residual block (`Relu`), `[1, 256, 10, T']`: the
/// input of the tail's first `ReduceMean`. Names belong to the model pinned
/// by `speaker_embed::MODEL_SHA256`.
pub(crate) const TRUNK_OUTPUT: &str = "onnx::ReduceMean_343";
const TAIL_WEIGHT: &str = "model.seg_1.weight";
const TAIL_BIAS: &str = "model.seg_1.bias";
const TAIL_MEAN: &str = "mean_vec";

/// Trunk-only model bytes and the tail's weights as little-endian f32:
/// `weight [256 * 5120] ++ bias [256] ++ mean_vec [256]`.
pub(crate) struct Split {
    pub model: Vec<u8>,
    pub tail: Vec<u8>,
}

struct Field<'a> {
    num: u64,
    /// The whole field, tag included (copied through unchanged).
    raw: &'a [u8],
    /// Length-delimited payload (empty for other wire types).
    payload: &'a [u8],
    varint: u64,
}

fn varint(b: &[u8], i: &mut usize) -> Result<u64> {
    let mut v = 0u64;
    for shift in (0..64).step_by(7) {
        let byte = *b.get(*i).context("truncated varint")?;
        *i += 1;
        v |= u64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            return Ok(v);
        }
    }
    bail!("varint too long")
}

fn fields(b: &[u8]) -> Result<Vec<Field<'_>>> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        let start = i;
        let tag = varint(b, &mut i)?;
        let (mut payload, mut value): (&[u8], u64) = (&[], 0);
        match tag & 7 {
            0 => value = varint(b, &mut i)?,
            1 => i += 8,
            5 => i += 4,
            2 => {
                let len = varint(b, &mut i)? as usize;
                payload = b.get(i..i + len).context("truncated field")?;
                i += len;
            }
            w => bail!("unsupported wire type {w}"),
        }
        anyhow::ensure!(i <= b.len(), "truncated field");
        out.push(Field { num: tag >> 3, raw: &b[start..i], payload, varint: value });
    }
    Ok(out)
}

fn put_varint(out: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        out.push((v as u8) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

fn put_bytes(out: &mut Vec<u8>, num: u64, payload: &[u8]) {
    put_varint(out, num << 3 | 2);
    put_varint(out, payload.len() as u64);
    out.extend_from_slice(payload);
}

fn str_field<'a>(f: &Field<'a>) -> &'a str {
    std::str::from_utf8(f.payload).unwrap_or("")
}

/// TensorProto → its f32 values (`raw_data` or `float_data`).
fn tensor_f32(t: &[u8]) -> Result<Vec<f32>> {
    let fs = fields(t)?;
    if let Some(f) = fs.iter().find(|f| f.num == 2) {
        anyhow::ensure!(f.varint == 1, "tail tensor is not float");
    }
    let bytes = match fs.iter().find(|f| f.num == 9) {
        Some(raw) => raw.payload.to_vec(),
        None => fs.iter().filter(|f| f.num == 4).flat_map(|f| f.payload.to_vec()).collect(),
    };
    Ok(bytes.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect())
}

/// `model`: the full ONNX model. Keeps the graph up to the node producing
/// [`TRUNK_OUTPUT`], makes that its only output, and lifts the tail's
/// initializers out.
pub(crate) fn split(model: &[u8]) -> Result<Split> {
    let mut out = Vec::with_capacity(model.len());
    let mut tail = None;
    for f in fields(model)? {
        if f.num == 7 {
            let (graph, t) = split_graph(f.payload)?;
            put_bytes(&mut out, 7, &graph);
            tail = Some(t);
        } else {
            out.extend_from_slice(f.raw);
        }
    }
    Ok(Split { model: out, tail: tail.context("model has no graph")? })
}

fn split_graph(g: &[u8]) -> Result<(Vec<u8>, Vec<u8>)> {
    let fs = fields(g)?;
    let node_outputs = |n: &Field<'_>| -> Result<Vec<String>> {
        Ok(fields(n.payload)?.iter().filter(|o| o.num == 2).map(|o| str_field(o).to_owned()).collect())
    };
    let mut nodes = fs.iter().filter(|f| f.num == 1);
    let mut last = None;
    for (k, n) in nodes.by_ref().enumerate() {
        if node_outputs(n)?.iter().any(|o| o == TRUNK_OUTPUT) {
            last = Some(k);
            break;
        }
    }
    let last = last.context("trunk output node not found")?;
    let (mut w, mut b, mut m) = (None, None, None);
    let mut out = Vec::with_capacity(g.len());
    let mut node_k = 0usize;
    for f in &fs {
        match f.num {
            1 => {
                if node_k <= last {
                    out.extend_from_slice(f.raw);
                }
                node_k += 1;
            }
            5 => {
                let name = fields(f.payload)?
                    .iter()
                    .find(|x| x.num == 8)
                    .map(|x| str_field(x).to_owned())
                    .unwrap_or_default();
                match name.as_str() {
                    TAIL_WEIGHT => w = Some(tensor_f32(f.payload)?),
                    TAIL_BIAS => b = Some(tensor_f32(f.payload)?),
                    TAIL_MEAN => m = Some(tensor_f32(f.payload)?),
                    _ => out.extend_from_slice(f.raw),
                }
            }
            12 => {} // outputs: replaced below
            _ => out.extend_from_slice(f.raw),
        }
    }
    // ValueInfoProto { name, type: TypeProto { tensor_type { elem_type: FLOAT } } }
    let mut vi = Vec::new();
    put_bytes(&mut vi, 1, TRUNK_OUTPUT.as_bytes());
    put_bytes(&mut vi, 2, &[0x0a, 0x02, 0x08, 0x01]);
    put_bytes(&mut out, 12, &vi);
    let (w, b, m) = (
        w.context("tail weight missing")?,
        b.context("tail bias missing")?,
        m.context("mean_vec missing")?,
    );
    let tail: Vec<u8> = w.iter().chain(&b).chain(&m).flat_map(|v| v.to_le_bytes()).collect();
    Ok((out, tail))
}

//! Pure-Rust Li-GRU implementation for streaming inference.
//!
//! Instead of running Li-GRU as ~80 primitive ONNX ops inside tract, this module
//! implements the full SqueezedLiGRU_S cell as native ndarray matrix-vector operations.
//!
//! # Li-GRU cell equation (matches Python LiGRU_Layer.forward_single_step)
//!
//! ```text
//! w_t   = W  × x_t                     [2H]  (no bias, W shape [2H, I])
//! w_t   = layernorm(w_t)                [2H]  (if normalize=true)
//! gates = w_t + U × h_{t-1}            [2H]  (U shape [2H, H])
//! at    = gates[0..H]                   [H]
//! zt    = sigmoid(gates[H..2H])         [H]   (update gate)
//! h_t   = zt ⊙ h_{t-1} + (1−zt) ⊙ relu(at)
//! ```
//!
//! # SqueezedLiGRU_S step (matches Python SqueezedLiGRU_S.step)
//!
//! ```text
//! x = linear_in(input)       [H]   grouped linear + relu
//! for each layer i:
//!     x = ligru_cell(x, h[i])  →  new h[i]
//! out = linear_out(x)        [out_dim]  grouped linear + relu  (or identity)
//! if gru_skip:
//!     out += grouped_linear(input, skip_weights)
//! ```

use ndarray::{s, Array1, Array2, Array3};
use serde::Deserialize;

// ---------------------------------------------------------------------------
// JSON deserialisation types
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct GroupedLinearJson {
    /// [G][I/G][H/G]
    weight: Vec<Vec<Vec<f32>>>,
    groups: usize,
    input_size: usize,
    hidden_size: usize,
    #[serde(default)]
    relu: bool,
}

#[derive(Deserialize)]
struct LiGruLayerJson {
    /// [2H][I]
    w: Vec<Vec<f32>>,
    /// [2H][H]
    u: Vec<Vec<f32>>,
    normalize: bool,
    hidden_size: usize,
    input_size: usize,
    /// "layernorm" (default) or "affine" (folded BatchNorm)
    #[serde(default)]
    norm_type: String,
    #[serde(default)]
    ln_weight: Vec<f32>,
    #[serde(default)]
    ln_bias: Vec<f32>,
}

#[derive(Deserialize)]
struct SqueezedLiGruJson {
    linear_in: GroupedLinearJson,
    layers: Vec<LiGruLayerJson>,
    linear_out: Option<GroupedLinearJson>,
    gru_skip: Option<GroupedLinearJson>,
}

#[derive(Deserialize)]
struct LsnrFcJson {
    /// [1][emb_out_dim]
    weight: Vec<Vec<f32>>,
    /// [1]
    bias: Vec<f32>,
}

#[derive(Deserialize)]
struct EncGruJson {
    gru: SqueezedLiGruJson,
    lsnr_fc: LsnrFcJson,
    lsnr_scale: f32,
    lsnr_offset: f32,
}

#[derive(Deserialize)]
struct DfGruJson {
    gru: SqueezedLiGruJson,
    df_skip: Option<GroupedLinearJson>,
}

#[derive(Deserialize)]
struct AllWeightsJson {
    enc: EncGruJson,
    erb_dec: SqueezedLiGruJson,
    df_dec: DfGruJson,
}

// ---------------------------------------------------------------------------
// Compute types
// ---------------------------------------------------------------------------

/// Grouped linear (no bias). Matches Python `GroupedLinearEinsum`.
///
/// weight shape: [G, I/G, H/G]
///
/// Forward: for each group g,  out[g*H/G..(g+1)*H/G] = x[g*I/G..(g+1)*I/G] @ weight[g]
pub struct GroupedLinear {
    weight_3d: ndarray::Array3<f32>,
    groups: usize,
    input_size: usize,
    pub hidden_size: usize,
    relu: bool,
}

impl GroupedLinear {
    fn from_json(j: GroupedLinearJson) -> Self {
        let g = j.groups;
        let ig = j.input_size / g;
        let hg = j.hidden_size / g;

        let flat: Vec<f32> = j
            .weight
            .iter()
            .flat_map(|gi| gi.iter().flat_map(|row| row.iter().copied()))
            .collect();
        let weight_3d =
            ndarray::Array3::from_shape_vec((g, ig, hg), flat).expect("GroupedLinear weight shape mismatch");

        Self {
            weight_3d,
            groups: g,
            input_size: j.input_size,
            hidden_size: j.hidden_size,
            relu: j.relu,
        }
    }

    /// Forward pass: x [input_size] → [hidden_size].
    pub fn forward(&self, x: &Array1<f32>) -> Array1<f32> {
        let g  = self.groups;
        let ig = self.input_size / g;
        let hg = self.hidden_size / g;

        let mut out = Array1::<f32>::zeros(self.hidden_size);
        for gi in 0..g {
            let x_g = x.slice(s![gi * ig..(gi + 1) * ig]);        // [I/G]
            let w_g = self.weight_3d.slice(s![gi, .., ..]);        // [I/G, H/G]
            // out_g = x_g @ w_g  →  [H/G]
            let out_g = w_g.t().dot(&x_g);
            out.slice_mut(s![gi * hg..(gi + 1) * hg]).assign(&out_g);
        }

        if self.relu {
            out.mapv_inplace(|v| v.max(0.0));
        }
        out
    }
}

/// Input normalisation for Li-GRU — either LayerNorm or folded affine (BatchNorm in eval).
///
/// - `LayerNorm`: normalises `w_t` by its own mean/var, then affine.
/// - `Affine`: per-element `w_t * scale + offset` (BatchNorm with folded running stats).
enum InputNorm {
    LayerNorm { weight: Array1<f32>, bias: Array1<f32> },
    Affine    { scale: Array1<f32>, offset: Array1<f32> },
}

impl InputNorm {
    fn from_json(norm_type: &str, w: Vec<f32>, b: Vec<f32>) -> Self {
        if norm_type == "affine" {
            Self::Affine { scale: Array1::from_vec(w), offset: Array1::from_vec(b) }
        } else {
            // "layernorm" or legacy empty string
            Self::LayerNorm { weight: Array1::from_vec(w), bias: Array1::from_vec(b) }
        }
    }

    #[allow(dead_code)]
    fn forward(&self, x: &Array1<f32>) -> Array1<f32> {
        match self {
            Self::LayerNorm { weight, bias } => {
                let n    = x.len() as f32;
                let mean = x.sum() / n;
                let var  = x.mapv(|v| (v - mean) * (v - mean)).sum() / n;
                let std  = (var + 1e-5_f32).sqrt();
                x.mapv(|v| (v - mean) / std) * weight + bias
            }
            Self::Affine { scale, offset } => {
                x * scale + offset
            }
        }
    }
}

/// Single Li-GRU layer.
///
/// w: [2H, I]  (input projection, no bias)
/// u: [2H, H]  (recurrent projection, no bias)
struct LiGruLayer {
    w: Array2<f32>,            // [2H, I]
    u: Array2<f32>,            // [2H, H]
    norm: Option<InputNorm>,
    pub hidden_size: usize,
    scratch: std::cell::RefCell<Vec<f32>>, // reusable [2H] gate buffer
}

impl LiGruLayer {
    fn from_json(j: LiGruLayerJson) -> Self {
        let two_h = j.hidden_size * 2;
        let i     = j.input_size;

        let w_flat: Vec<f32> = j.w.iter().flat_map(|row| row.iter().copied()).collect();
        let u_flat: Vec<f32> = j.u.iter().flat_map(|row| row.iter().copied()).collect();

        let w = Array2::from_shape_vec((two_h, i), w_flat)
            .expect("LiGruLayer w shape mismatch");
        let u = Array2::from_shape_vec((two_h, j.hidden_size), u_flat)
            .expect("LiGruLayer u shape mismatch");

        let norm = if j.normalize {
            assert!(
                !j.ln_weight.is_empty() && !j.ln_bias.is_empty(),
                "normalize=true but ln_weight/ln_bias missing"
            );
            Some(InputNorm::from_json(&j.norm_type, j.ln_weight, j.ln_bias))
        } else {
            None
        };

        let two_h = j.hidden_size * 2;
        Self {
            w,
            u,
            norm,
            hidden_size: j.hidden_size,
            scratch: std::cell::RefCell::new(vec![0.0_f32; two_h]),
        }
    }

    /// Single recurrent step — fused, zero heap allocation.
    ///
    /// x_t:   [I]   — input to this layer (output of previous layer or linear_in)
    /// h_prev:[H]   — hidden state from previous frame (mutated in-place → new h)
    ///
    /// The return value IS the same buffer as `h` after update; the caller just
    /// holds a reference. We mutate `h` in-place and return it for chaining.
    ///
    /// Inner loop avoids all ndarray temporaries (no `to_owned`, no `mapv`):
    ///   1. w_t = W @ x_t               [computed via raw slice loop]
    ///   2. optional LayerNorm(w_t)
    ///   3. gates[i] = w_t[i] + U[i] · h   (one dot per row of U)
    ///   4. for each unit i in [0..H]:
    ///        at[i]  = gates[i]           (activation input = first H rows)
    ///        zt[i]  = σ(gates[H+i])      (update gate   = last  H rows)
    ///        h[i]   = zt[i]*h[i] + (1−zt[i])*relu(at[i])
    pub fn step_inplace(&self, x_t: &[f32], h: &mut Vec<f32>) {
        let hidden = self.hidden_size;
        let input  = x_t.len();
        let two_h  = hidden * 2;

        // ── Step 1: gates = W @ x_t  (shape [2H]) ────────────────────────────
        // W is stored row-major [2H, I], so row i = &w[[i, 0..I]]
        let w_slice = self.w.as_slice().expect("w not contiguous");
        let u_slice = self.u.as_slice().expect("u not contiguous");

        // Borrow preallocated scratch buffer — zero allocation per call.
        let mut gates = self.scratch.borrow_mut();
        gates.iter_mut().for_each(|v| *v = 0.0);

        for i in 0..two_h {
            let row = &w_slice[i * input..(i + 1) * input];
            let mut acc = 0.0_f32;
            for (w_val, x_val) in row.iter().zip(x_t.iter()) {
                acc += w_val * x_val;
            }
            gates[i] = acc;
        }

        // ── Step 2: optional LayerNorm on gates ───────────────────────────────
        if let Some(ref ln) = self.norm {
            match ln {
                InputNorm::LayerNorm { weight, bias } => {
                    let n = two_h as f32;
                    let mean = gates.iter().sum::<f32>() / n;
                    let var: f32 =
                        gates.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / n;
                    let inv_std = 1.0 / (var + 1e-5_f32).sqrt();
                    let w = weight.as_slice().unwrap();
                    let b = bias.as_slice().unwrap();
                    for i in 0..two_h {
                        gates[i] = (gates[i] - mean) * inv_std * w[i] + b[i];
                    }
                }
                InputNorm::Affine { scale, offset } => {
                    let s = scale.as_slice().unwrap();
                    let o = offset.as_slice().unwrap();
                    for i in 0..two_h {
                        gates[i] = gates[i] * s[i] + o[i];
                    }
                }
            }
        }

        // ── Step 3: gates += U @ h  (add recurrent contribution in-place) ─────
        for i in 0..two_h {
            let row = &u_slice[i * hidden..(i + 1) * hidden];
            let mut acc = 0.0_f32;
            for (u_val, h_val) in row.iter().zip(h.iter()) {
                acc += u_val * h_val;
            }
            gates[i] += acc;
        }

        // ── Step 4: fused gated recurrence (in-place update of h) ────────────
        // gates[0..H]  = activation input (at)
        // gates[H..2H] = update gate logit (zt_raw)
        for i in 0..hidden {
            let at = gates[i];
            let zt = 1.0_f32 / (1.0 + (-gates[hidden + i]).exp()); // sigmoid
            let hcand = at.max(0.0_f32);                            // relu
            h[i] = zt * h[i] + (1.0 - zt) * hcand;
        }
    }

    /// Single recurrent step (ndarray API, allocates — kept for external use).
    ///
    /// Prefer `step_inplace` in hot paths.
    #[allow(dead_code)]
    pub fn step(&self, x_t: &Array1<f32>, h_prev: &Array1<f32>) -> Array1<f32> {
        let mut h = h_prev.to_vec();
        let x_slice = x_t.as_slice().expect("x_t not contiguous");
        self.step_inplace(x_slice, &mut h);
        Array1::from_vec(h)
    }
}

/// Full `SqueezedLiGRU_S` — linear_in + Li-GRU layers + linear_out + optional skip.
pub struct SqueezedLiGru {
    linear_in:  GroupedLinear,
    layers:     Vec<LiGruLayer>,
    linear_out: Option<GroupedLinear>,
    gru_skip:   Option<GroupedLinear>,
    /// Hidden size of each Li-GRU layer.
    pub hidden_size: usize,
    /// Number of layers.
    pub num_layers: usize,
    /// Output size after linear_out (or hidden_size if no linear_out).
    pub output_size: usize,
}

impl SqueezedLiGru {
    fn from_json(j: SqueezedLiGruJson) -> Self {
        let hidden_size = j.layers.first().map(|l| l.hidden_size).unwrap_or(256);
        let num_layers  = j.layers.len();

        let linear_out_opt = j.linear_out.map(GroupedLinear::from_json);
        let output_size = linear_out_opt
            .as_ref()
            .map(|lo| lo.hidden_size)
            .unwrap_or(hidden_size);

        Self {
            linear_in:  GroupedLinear::from_json(j.linear_in),
            layers:     j.layers.into_iter().map(LiGruLayer::from_json).collect(),
            linear_out: linear_out_opt,
            gru_skip:   j.gru_skip.map(GroupedLinear::from_json),
            hidden_size,
            num_layers,
            output_size,
        }
    }

    /// Streaming single-frame step — zero heap allocation in the GRU layers.
    ///
    /// input: [input_size]
    /// h:     slice of length `num_layers`, each `Vec<f32>` of size `hidden_size`
    ///
    /// Returns (output [output_size], new_h Vec of length num_layers).
    ///
    /// The hidden states are updated in-place inside the `h` vecs; new_h holds the
    /// same updated values for the caller to store.
    pub fn step(
        &self,
        input: &Array1<f32>,
        h: &[Array1<f32>],
    ) -> (Array1<f32>, Vec<Array1<f32>>) {
        debug_assert_eq!(h.len(), self.num_layers);

        // linear_in + relu → [hidden_size]
        let x_arr = self.linear_in.forward(input);

        // Li-GRU layers — use step_inplace to avoid per-layer allocations
        let mut x_slice: Vec<f32> = x_arr.to_vec();
        let mut new_h: Vec<Array1<f32>> = Vec::with_capacity(self.num_layers);

        for (i, layer) in self.layers.iter().enumerate() {
            // Convert h[i] to mutable Vec for in-place update
            let mut h_i: Vec<f32> = h[i].to_vec();
            layer.step_inplace(&x_slice, &mut h_i);
            // h_i is now the new hidden state; it is also the layer output
            x_slice = h_i.clone();
            new_h.push(Array1::from_vec(h_i));
        }

        let x_out = Array1::from_vec(x_slice);

        // linear_out (or identity)
        let mut out = match &self.linear_out {
            Some(lo) => lo.forward(&x_out),
            None => x_out,
        };

        // optional gru_skip: add skip(input) to output
        if let Some(ref skip) = self.gru_skip {
            out = out + skip.forward(input);
        }

        (out, new_h)
    }
}

// ---------------------------------------------------------------------------
// Encoder GRU + lsnr_fc
// ---------------------------------------------------------------------------

/// Encoder Li-GRU wrapper — runs SqueezedLiGRU_S then computes lsnr.
pub struct EncGru {
    pub gru: SqueezedLiGru,
    lsnr_w: Array1<f32>,  // [emb_out_dim]  — flattened lsnr_fc.weight[0]
    lsnr_b: f32,
    lsnr_scale: f32,
    lsnr_offset: f32,
}

impl EncGru {
    fn from_json(j: EncGruJson) -> Self {
        let lsnr_w = Array1::from_vec(
            j.lsnr_fc.weight.into_iter().flatten().collect(),
        );
        let lsnr_b = j.lsnr_fc.bias.first().copied().unwrap_or(0.0);
        Self {
            gru:         SqueezedLiGru::from_json(j.gru),
            lsnr_w,
            lsnr_b,
            lsnr_scale:  j.lsnr_scale,
            lsnr_offset: j.lsnr_offset,
        }
    }

    /// Step: emb_raw [emb_in_dim], h slice → (emb [emb_out_dim], lsnr f32, new_h)
    pub fn step(
        &self,
        emb_raw: &Array1<f32>,
        h: &[Array1<f32>],
    ) -> (Array1<f32>, f32, Vec<Array1<f32>>) {
        let (emb, new_h) = self.gru.step(emb_raw, h);

        // lsnr_fc: Linear → Sigmoid → scale+offset
        let logit = self.lsnr_w.dot(&emb) + self.lsnr_b;
        let prob  = 1.0_f32 / (1.0 + (-logit).exp());
        let lsnr  = prob * self.lsnr_scale + self.lsnr_offset;

        (emb, lsnr, new_h)
    }
}

// ---------------------------------------------------------------------------
// DF decoder GRU + df_skip
// ---------------------------------------------------------------------------

/// DF decoder Li-GRU + optional df_skip GroupedLinear.
///
/// df_skip applies to the *encoder* emb (same input as the GRU), not the GRU output.
pub struct DfGru {
    pub gru: SqueezedLiGru,
    df_skip: Option<GroupedLinear>,
}

impl DfGru {
    fn from_json(j: DfGruJson) -> Self {
        Self {
            gru:     SqueezedLiGru::from_json(j.gru),
            df_skip: j.df_skip.map(GroupedLinear::from_json),
        }
    }

    /// Step: enc_emb [emb_out_dim], h slice → (c_combined [df_hidden], new_h)
    pub fn step(
        &self,
        enc_emb: &Array1<f32>,
        h: &[Array1<f32>],
    ) -> (Array1<f32>, Vec<Array1<f32>>) {
        let (c, new_h) = self.gru.step(enc_emb, h);

        let c = if let Some(ref skip) = self.df_skip {
            c + skip.forward(enc_emb)
        } else {
            c
        };

        (c, new_h)
    }
}

// ---------------------------------------------------------------------------
// Top-level container
// ---------------------------------------------------------------------------

/// All three Li-GRU modules loaded from `ligru_weights.json`.
pub struct NativeGrus {
    pub enc:     EncGru,
    pub erb_dec: SqueezedLiGru,
    pub df_dec:  DfGru,
}

impl NativeGrus {
    /// Parse from the JSON string produced by `export_native_onnx.py`.
    pub fn from_json(json_str: &str) -> Result<Self, String> {
        let j: AllWeightsJson =
            serde_json::from_str(json_str).map_err(|e| format!("ligru JSON parse error: {e}"))?;
        Ok(Self {
            enc:     EncGru::from_json(j.enc),
            erb_dec: SqueezedLiGru::from_json(j.erb_dec),
            df_dec:  DfGru::from_json(j.df_dec),
        })
    }
}

// ---------------------------------------------------------------------------
// State helpers — convert between Array3<f32> [L,1,H] and Vec<Array1<f32>>
// ---------------------------------------------------------------------------

/// [L, 1, H] → Vec of length L, each [H].
pub fn array3_to_vec(h: &Array3<f32>) -> Vec<Array1<f32>> {
    let layers = h.shape()[0];
    (0..layers)
        .map(|i| h.slice(s![i, 0, ..]).to_owned())
        .collect()
}

/// Vec of length L (each [H]) → [L, 1, H].
pub fn vec_to_array3(h: &[Array1<f32>]) -> Array3<f32> {
    let layers = h.len();
    let hidden = h[0].len();
    let mut arr = Array3::<f32>::zeros((layers, 1, hidden));
    for (i, hi) in h.iter().enumerate() {
        arr.slice_mut(s![i, 0, ..]).assign(hi);
    }
    arr
}

/// Build a zeroed Vec hidden state for `num_layers` layers of size `hidden_size`.
pub fn zero_hidden(num_layers: usize, hidden_size: usize) -> Vec<Array1<f32>> {
    (0..num_layers)
        .map(|_| Array1::<f32>::zeros(hidden_size))
        .collect()
}

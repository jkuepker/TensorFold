# Qwen3.6 MoE engine against the Python engine

`capture.py` runs the Python CUDA engine on stock Qwen3.6-35B-A3B MLX 4-bit and dumps, per prompt: every
norm's `h`, `y` and `xs`, every MoE layer's router logits, picks, weights and output, each step's logits and the
state a prefill commits (the layout is under Dump layout below). Two runs of it hash equal.

1. Dumps and the Triton launches, in the Python engine's container, on a copy of its tree:
   `TRITON_CACHE_DIR=<cache> PYTHONPATH=<copy>/src python3 capture.py --model <checkpoint> --out <dumps> --tools <folder of triton_aot_manifest.py>`
2. The kernel set, as `nemotron/box/README.md` steps 2 and 3 (`triton_aot_manifest.py --launches <dumps>/launches.json`, then `nemotron/aot_pack.py --jit <dumps>/jit.json`).
3. The Zig engine against the dumps, byte for byte:
   `tf-cuda-test qwen36-layers <checkpoint> <kernel set> <dumps> [story,code,facts,long] [--steps N] [--teacher]`

The test runs the whole 40-layer engine (prefill, one-row steps from an empty state, steps after the prefill)
and compares at every norm and every MoE layer, so the first differing op is the first line it prints.
`--teacher` puts the dump's bytes in place of each compared value, so one layer's fault does not flow on.

Served: `tensorfold-native serve MODEL --no-drafts` (on a GB10 with `TENSORFOLD_CUDA_KERNELS` unset: the built-in kernels alone; a captured kernel set is optional there)
answers like the Python 0.6.6 engine run serial. `serve_check.py record URL FIXTURE` records a Python server's prompt
ids (/tokenize), greedy reply ids (`return_token_ids`, temperature 0, 256 tokens, thinking off) and parsed tool calls;
`serve_check.py check URL FIXTURE` reruns the fixture against any server and compares the chat template's ids (a), the
reply ids (b) and the tool calls (c). Stdlib Python only.

## Dump layout

Prompts (`prompts.json`): story, code, facts (chat turns of 25 to 35 tokens) and long (about 5,000 tokens of module
source, which prefill splits evenly into two chunks of about 2,520 rows: `chunks()` divides a prompt into the fewest
4,096-row passes of equal size). For each prompt the engine's own passes are dumped as follows.

```text
<prompt>/cold/stepSSS/      step SSS of the one-row steps from an empty state (up to 32; the chat prompts
                            are shorter), forced token prompt[SSS], every step dumped
<prompt>/prefill/           the whole prompt's prefill pass (long: tensor digests only, digests.json, one chunk
                            after the other with call numbers running on)
<prompt>/state/             the committed state after prefill (see below)
<prompt>/warm/stepSSS/      32 teacher-forced one-row steps after prefill, forced token: the prompt's first
                            greedy token, then each step's greedy argmax (the serial reply; tokens.json)
<prompt>/tokens.json        the prompt's hash, forced and sampled tokens, each step's logits sha256, and the
                            engine's own generate(draft=False) tokens over the same count (must equal)
```

A pass directory holds raw little-endian tensors (shape and dtype in `dump.json`: name -> [dtype, shape, sha256]):

```text
NNN_h.bin NNN_y.bin NNN_xs.bin   the NNN-th add_rmsnorm call: its residual sum h (bf16 [rows, hidden]), the
                                 normed rows y (bf16), and the y group sums xs (the matmul's activation sums,
                                 fp32). Calls run in order, two a layer (input norm, then post norm), then the
                                 final norm of the pass: 81 calls with 40 layers
mNN_logits.bin                   MoE call NN (one a layer, the layer's index in a one-chunk pass): router logits,
                                 fp32 [rows, 257] (256 experts, the shared expert's gate last)
mNN_pick.bin  mNN_wts.bin        its picks (int32 [rows, 9]: 8 experts by logit, the lower id on a tie, then
                                 the shared expert 256) and weights (fp32 [rows, 9])
mNN_out.bin                      the layer's MoE output (bf16 [rows, hidden])
logits.bin                       a decode step's logits (bf16 [1, vocab])
```

The state after prefill (`<prompt>/state/`):

```text
LNN_rec.bin LNN_conv.bin   delta-net layer NN: the recurrent state (fp32 [32, 128, 128] value heads, value rows,
                           key columns) and the conv window (bf16 [3, 8192])
LNN_k.bin LNN_v.bin        attention layer NN: the cache rows past (bf16 [pos, 2, 256])
```

`weights.txt` holds "name sha256 bytes" lines for the engine's loaded weights (tiled 4-bit projections, norms,
conv, router rows, expert stacks), the lines the Zig upload prints for its own device bytes. `manifest.json` holds
every dump file's sha256 and the tree's; `timing.json` (outside the manifest) holds wall times.

## MTP drafts

The checkpoint's MTP layer drafts through the lane core (`families/qwen3_5/cuda/mtp*.zig`): the kept rows' final normed
states are absorbed into the head, which then chains up to three drafts, ending a chain after a draft it gives under 0.3
probability (the Python engine's rule); the target verifies them in one pass, the
DeltaNet state replays to the accepted path, and every row is drawn by the target's keyed rule, so a drafted reply equals
the `"draft": false` one, sampled requests included. `"draft": false` on a request turns its drafts off.

1. The Python engine's head calls and drafted rounds, in the container as above: `capture.py --mtp --model <checkpoint> --out <dumps>`
   (`capture_mtp.py` is its body; `mtp_weights.txt` holds the head's tensor digests, `<prompt>/greedy/cNNN` each call's
   states, norms, normed rows and draft logits, `tokens.json` the drafted and serial replies, greedy and one seeded sampled
   request each). It also sweeps every verify width, so the Triton set packed from its launches covers drafted rounds.
2. `tf-qwen35-upload MODEL --mtp --digests` prints the head's tensors as `mtp_weights.txt` holds them.
3. `tf-cuda-test qwen36-mtp MODEL <kernel set> <dumps> [PROMPTS]` replays the head calls in order over its own cache:
   every norm, normed row and draft logit byte for byte.
4. `mtp_check.py URL FIXTURE DUMPS` against a native server with drafts on: the fixture's chat requests and the dumps'
   four prompts (greedy, and the seeded sampled request) drafted, with `"draft": false`, and the Python engine's serial
   reply must all match, with acceptance beside the Python engine's.

## Prompt cache

The native CUDA server keeps a stream's state (each DeltaNet layer's recurrence and conv window, each
attention layer's key and value rows) at the cuts the lane core names, 62 MiB and 20 KiB a prompt token (the startup
line says so). `cache_check.py record URL FIXTURE` against a server run with `--prompt-cache-gib 0` records three
conversations turn by turn (a short chat, a three-turn chat, a ~5k-token history, fixed assistant texts between the
turns); `cache_check.py check URL FIXTURE` against a server with the cache on must give the same reply ids token for token,
a hit (usage's `cached_tokens`) on every turn after a conversation's first, and `/metrics`' `prefix_cache_hits_total`
equal to the hits. `serve_check.py check` against a cache-on server is the same check against the Python engine.

With drafts on, a drafting stream's kept state also holds its prompt's final normed rows (4 KiB a token more), so the
draft head reads the whole prompt after a restore; a state kept by a `"draft": false` stream has no rows and a drafting
stream runs its prompt whole instead (the entry stays). `cache_check.py check` passes against a drafted server too.

## Concurrent streams

`--parallel N` (up to 16) serves N streams: every stream's window goes through one shared round (the 27B's
multi_tree_forward with the MoE block on the round's rows), drafted or not, each keeping its own state, head cache and
keyed draws, so a reply equals the same request alone. Prompts still run one at a time through the chunked prefill.

1. The glue kernels' MULTI variants and the row-count alignments of shared rounds come from the Python engine run with
   N streams: `capture.py --lanes 8 --model <checkpoint> --out <dir> --tools <triton tools>` (`capture_multi.py` is its
   body; it sends N requests at once, greedy and seeded sampled, drafted and not). Pack its manifest with the MTP run's
   (`aot_pack.py` takes `--manifest` and `--cache` in pairs) into the kernel set the server loads.
2. `lanes_check.py URL FIXTURE ORACLE [1,4,8]` against a server with drafts on and `--parallel` of at least the largest
   count: each request alone first, then N at once (the 5k-token prompt and the seeded sampled request among them),
   then N at once with every second one sent with `"draft": false`; every reply must equal its solo reply token for
   token (the solo ones also equal the Python engine's serial replies).

## MiniCPM-V 4.7 (text)

`../minicpm/fixture.json` holds `serve_check.py record` output from a Python 0.6.6 MiniCPM-V 4.7 server (serial, drafts
off, thinking off): six chat requests with their prompt ids, greedy reply ids and parsed tool calls. The native server
(`minicpmv4_7`) answers it token for token, drafted and with `--draft-off`, alone and with two clients at once:
`serve_check.py check URL ../minicpm/fixture.json [--draft-off]`. It uses the same kernels as Qwen3.6.

## Native prep kernels

`qwen_prep.cu` runs `glue.gdn_pre` and `glue.attn_prep` (the conv, SiLU, q/k norms, g and beta; the q/k norms and rotary) without the
captured cubins, on GB10 and for the shapes in `prep_native.zig` (`gdnFits`, `attnFits`). `tf-cuda-test qwen-prep <kernel set>` runs every
`_gdn_pre` and `_attn_prep` entry of the set (each pointer alignment and MULTI form) against the native kernel on synthetic rows of
1 to 4096 (saturating sigmoid and softplus inputs, positions up to 2^30 for the sin and cos argument reduction), whole outputs and guard
bytes byte for byte. `qwen36-layers`, `qwen36-mtp` and the serve checks must pass on a kernel set with those entries removed from `aot.json`.

## Float64 references

The byte-equality tests above show the native kernels equal the Python engine's; these show how far both sit from the exact answer.
Each reference is computed in float64 on the host from the same input bytes (bf16 and fp32 decoded exactly, nothing taken from a
native or Triton output), element by element, with a per-element bound built from the rounding points the format forces and the
condition of the sum. A column's worst absolute error, worst relative error and worst error over its bound are kept over every
element of that column, and printed as one `F64 <output>.<phase> ...` line beside the aggregate over every element of the output: the
mean absolute error and the RMS error (`mean_abs`, `rms`), so an average never stands in for a worse column. The fields come for native
and, where a kernel set is given, for Triton. Decode row counts (1 to 16 rows: decode steps, lane rounds, verify windows) and prompt row
counts (more than 16 rows) are separate lines (`.decode`, `.prompt`), not pooled; where one kernel takes either, a prompt-sized call is
relaunched on the first 16 rows of the same device buffers, so both are measured on identical rows. The relative error is over the larger
of the reference and one bf16 ulp of the vector's largest value (prep, attention) or of the row norm (GDN), so a cancelling element does
not read as 100%.

- `tf-cuda-test qwen-prep-f64 [kernel set]`: `gdn_pre` (q, k, v, g, beta) and `attn_prep` (q, k, rotary) over serving magnitudes
  and heavy tails, bound asserted; with a kernel set, every captured entry also runs on the same inputs, the bound-free extremes of
  `qwen-prep` are compared, and native must be no further from float64 than Triton in every column. Positions past 262143 are a
  separate `_stress` output: the fp32 angle `pos * inv` rounds there, so its bound is the angle's own and is loose by design.
- `tf-cuda-test qwen-attention-f64 [kernel set] [rounds]`: the attention output, and every partial slot's `PO / PL` and `PM + ln PL`
  (the quantities that do not depend on the kernel's own running maximum), against a float64 softmax attention over the stream's
  committed keys and the row's root path, the mask built from the window's parents; plans come from `round.stage`. Slots past a row's
  last key must hold `l = 0`, `m = -inf`. The phase is the call's total window rows (up to 16 decode, more prompt-sized); a round is staged
  once, so there is no identical-row relaunch. Committed lengths run to 8191 on the rig of the byte-equality test, then a second rig of
  262400 positions checks 16383 to 262143 (windows of 1 to 8 rows, and 64 rows at 32768; chains and trees; folded groups of 2048
  keys and unfolded chunks: a stream folds when `committed / 2048 * rows >= 64`, so 16383 against 16384 with 8 rows, 32768 with 4 rows
  and 131072 with 1 row sit on the threshold, and 131071 is the longest unfolded stream), so long folded groups are checked in float64 as
  well. With a kernel set the captured Triton launches run those lengths too and fill the Triton columns. A slot's exp sum is bounded by
  (54 + 6 (chunks - 1)) 2^-24, relative, for the chunks it folds (the ex2.approx, the tile and chain sums, and the rescales of the running
  maximum), and the merge adds 12 units of 2^-24 per fold.
- `tf-cuda-test qwen-gdn-f64`: the tree kernel's chain (decode) and every schedule-slot class, one and several streams, the prompt
  chain kernel's output and final state, and the in-place replay of accepted rows, against a float64 gated delta rule. The phase is the
  kernel: the tree kernel's chain, tree and replay are decode, the chunked prompt kernel is prompt. The same rows go through both chains
  and are reported separately (`gdn.identical_rows_y.decode`, `.prompt`). A step's fp32 error is bounded by 32 units of 2^-24 of the
  state row norm, decayed by `g` down the path, and reaches `y` through `|q|`.

None of the three needs a captured set (`qwen-prep-f64` and `qwen-attention-f64` take one to add the Triton columns; `qwen-gdn-f64` has no
Triton counterpart). Negative controls: `TF_QWEN_F64_PERTURB=<column>` moves that column's native elements 1.25 bounds off the
reference (each command must then fail), and `TF_QWEN_PREP_FATBIN` runs `qwen-prep-f64` on another build of `qwen_prep.cu`.

## Programmatic dependent launch

Each plain launch on the stream leaves the GPU about 2 us idle before the next kernel starts; a kernel launched with
the PDL attribute is staged while the one before it still runs, and the gap closes. The native kernels that follow
another grid (`qwen_route.cu`'s three, `gdn.cu`'s tree and replay) begin with `griddepcontrol.wait` and are launched
that way on GB10. The captured Triton kernels have no wait, so `aot_pdl.py --set <kernel set> --cache <Triton cache>
[--cache ...] --ptxas <ptxas> --out <new set>` rebuilds a set: it finds each kernel's PTX by hash in the Triton
caches, builds it with Triton's own ptxas command and requires the captured cubin back byte for byte (use the ptxas of
the container that ran the capture; run the script inside it), then builds it again with `griddepcontrol.wait` as the
first instruction and marks the kernel `pdl` in `aot.json`. The loader launches `pdl` kernels with the attribute. The
rebuilt SASS has the same instructions plus the wait (`ACQBULK`, ahead of every memory access); on some kernels ptxas
also renames registers or reorders independent instructions around it, so the checks above must pass on the new set,
as they do on the captured one. Point `TENSORFOLD_CUDA_KERNELS` at it.

## Without a captured set

`qwen36-layers` and `qwen36-mtp` take `native` in place of the kernel set: the engine then runs on the built-in kernels alone, and a launch none of them covers stops with `QwenNeedsCapturedKernels`. Both pass byte-equal that way on a GB10, as do the serve, MTP, lanes and cache checks with `TENSORFOLD_CUDA_KERNELS` unset.

## Built-in row glue

`qwen_glue.cu` (add_rmsnorm, gated_norm, gate_mul, group_sums, paths) and `embed_rows.cu`'s 4-bit rows replace those Triton launches on a GB10, with the cubins' element-to-thread order, sum trees and `sqrt.approx` / `div.full` / `ex2.approx` primitives. `tf-cuda-test qwen-glue <kernel set>` runs each of the set's entries (pointers an entry lists as unaligned sit 4 bytes off a boundary) beside the native kernel on identical buffers, zero rows, inf and NaN rows, subnormal and wide-range rows, and every row count the engine serves, and compares every output byte. `exact` in `families/qwen27/cuda/glue_native.zig` names the kernels that pass on every shape; a shape outside the tested constants (hidden 2048, 32 value heads of 128, 16 query heads of 256) still replays. `qwen36-layers` and `qwen36-mtp` then run with the native kernels in.

`tf-cuda-test qwen-glue-f64 [<kernel set>|-] [oracle]` checks the same kernels against an independent float64 reference, on top of the byte
comparison and without changing a kernel output. The reference is computed on the host in float64 from the same input bytes (`core.norms_ref`
for the norm, `core.act_ref` for the sigmoid and the 4-bit dequantisation), never from a native or Triton output, on the differential test's
rows without NaN or inf; a row whose fp32 sum of squares overflows is counted as excluded.

- Passes: `realistic:` (normal, outlier and wide-range rows; `embed` and `paths` run here only) and `stress:` (every other kind of the rig:
  zeros, subnormal, tiny, huge and spread rows). The run ends with a negative control: every native element of one column of every
  output is moved 1.25 bounds off its reference and each output must fail, and a perturbed `paths` entry must differ.
- Each output prints one `F64` line per phase with the worst per-column absolute and relative error (with the column), the mean absolute
  and RMS error, and the largest error as a fraction of its bound, for native and, with a kernel set, for the captured Triton launch; native
  must be no worse than Triton in any column. Rows of 1 to 16 are `.decode`, more are `.prompt`, and each prompt-sized call is relaunched on
  its first 16 rows of the same buffers, so decode and prompt are measured on identical rows.
- The relative error is `|err| / |reference|`, except for a group sum (`XS`), where it is divided by the sum of the terms' magnitudes, and for
  `gate_mul` and `gated_norm`, where the divisor is at least the fp32 sigmoid's flush floor (below 2^-126 the division flushes). Every
  element must sit inside its bound, half a bf16 ulp of the reference plus the fp32 arithmetic's slack.
- `add_rmsnorm` with a residual rounds `h = bf16(fp32(x) + fp32(r))` before it normalises, so `H` is checked against that sum rounded on
  the host (exactly), and `Y` against the norm of the rounded `h` with the norm's own slack only; a kernel that rounded `h` toward zero
  would put `Y` over its bound without Triton. `H` is also checked against the float64 sum x + r (half a bf16 ulp).
- Group sums are checked twice: against the float64 sum of the reference rows, and against the float64 sum of the rows the kernel stored.
  Their bound is the fp32 reduction's own: the number of adds on the deepest path times 2^-24 times sum |terms| (10 for `group_sums` and
  `add_rmsnorm`: a chain of 8 and the 4-2-1 butterfly; 7 for `gated_norm` and `gate_mul`: a chain of 4 and the 8-4-2-1 butterfly).
- With an oracle directory the real dumped `y` rows go through `group_sums` (Triton is the dump's own sums), and add_rmsnorm sees the `h`
  of every eighth dumped `y` file, counted in visiting order over prompts, phases and steps (not every eighth row). Every fourth
  input-norm call also goes through the residual path as `x` = the previous call's dumped `h` and `r` = the layer's dumped MoE output;
  the run reports how many of those pairs reproduce the dumped `h` bit for bit.
- Limit of the group-sum bound: every summation order of the same depth meets the same worst-case bound, so on arbitrary rows no sound
  bound against float64 can tell a reversed chain, another butterfly order or a pairwise chain from the kernel's order, and the fused
  `XS` of `add_rmsnorm`, `gated_norm` and `gate_mul` sums the kernel's own outputs, which the test cannot choose. `group_sums` takes any row,
  so it also runs ladder rows (`group_sums[ladder rows].XS`): a group holding B and seven values just above half an fp32 ulp of B,
  bounded per element (sum |x_i| gamma(d_i), d_i the adds on x_i's path). The kernel's order errs 0.95 units of 2^-24 B against a
  bound of 4.0; a chain summed the other way, or a sequential sum, errs 6.9 and fails. A different butterfly order, a pairwise chain
  and orders no deeper than the kernel's at every element err as the kernel does and pass; only a bit-exact fp32 emulation of the
  order would tell them apart, and the byte comparison with the captured launch (`qwen-glue`) does.

## Native tree attention

`kernels/cuda/qwen_attention.cu` (`tree_native.zig`) is attention.py's `_shared`, `_tail` and `_merge` in native CUDA: both dots are
`mma.m16n8k16` chains in key and head-dim order onto `o * alpha`, the softmax is `ex2.approx` of `x * log2(e)`, `l` is `fma(l, alpha, row sum)`
with the row sum taken as the Triton program reduces it (per thread in register order, `xor 2`, `xor 1`, then the four warps as
`(w0 + w2) + (w1 + w3)`), the fold is `fma(old, a, new * b)` and the output is `div.full.f32`. `tf-cuda-test qwen-attention <kernel set> [rounds]`
runs it against the captured launches on identical buffers (plans from `round.stage`; committed lengths 0 to 8191, windows of 1 to 128
rows as chains and random trees, 2 to 16 streams, normal and heavy-tail keys, NaN words past each stream's committed rows, every captured
`_shared`/`_tail`/`_merge` variant with its pointer alignments and `W`); every partial slot (0xff-filled first) and the bf16 output must be
byte-equal. The engine takes the native kernels on sm_121 only (`tree_native.verified_capability`); `tree_native.exact` turns one kernel
back to the replay. A kernel set without those three entries (`aot.json` stripped of them) runs the whole engine on the native ones.

## FP64 references

`qwen-router-f64`, `qwen-experts-f64` and `qwen-proj-f64` hold the MoE and projection kernels to CONTRIBUTING.md's precision rule, beside the
byte-equality differentials above and without changing a kernel output. Each output is compared with a host float64 reference computed from
the same input bytes (bf16 rows, bf16 router rows, MLX 4-bit words with bf16 scales and biases, all dequantized in float64), never from a
native or Triton output. Every check prints, per output and phase (`.decode` for 1 to 16 rows, `.prompt` for more), the worst per-column
absolute and relative error with the column and the mean absolute and RMS error, native and, with a kernel set, Triton (`RESULT` per
case, `TABLE` per output over the run), and fails past its bound or where native is worse than Triton in a column; relative error is
against max(|reference|, the column's largest |reference| / 1024). The prompt batches of the experts and the projections also run their
first 16 rows through the decode kernels, so decode and prompt see identical rows. `CONTROL` lines show the check refusing an output moved
four bounds away (and, for the router, two picks swapped); `TF_QWEN_F64_PERTURB=<column>` moves that column of every native output.

- `tf-cuda-test qwen-router-f64 <kernel set|native> [fixtures]` (fixtures: `qwen35/capture_router.py`'s dump; the kernel set holds the
  captured `_router`, `_topk_rows`, `_combine`): router logits (bound 2^-16 of sum |x w|, one fp32 rounding per k16 step), the picks
  (a top-8 up to the logits' own errors against float64, equal to the strict float64 ranking or one transposition from it, adjacent
  picks or the last pick against the best unpicked expert; each such near-tie row is printed with its margin), the weights
  (softmax of the picks' float64 logits, the shared slot a sigmoid of the bf16-rounded gate logit) and the combine (fp32 and bf16 `y`),
  on the real captured rows and a seeded sweep. With a kernel set the Triton launch gets the same check and native must be no worse.
- `tf-cuda-test qwen-experts-f64`: the shared expert and 256 routed ones (MLX 4-bit, SwiGLU, hidden 2048, width 512, top-8), decode at
  1, 4 and 16 rows and the prompt kernels at 96 rows and 128 rows on one hot set. Gate-up is compared after SwiGLU with a bound built from
  the kernels' bf16 roundings of gate, up, silu and the product; down reads the kernel's own activations (fp32 rows in decode, bf16 in
  the prompt). The prompt reference rounds the weights to bf16 once, as that kernel does (quant-ref's convention).
- `tf-cuda-test qwen-proj-f64 <kernel set|native>`: GDN qkv, z, b, a; attention q, k, v; o_proj/out_proj; lm_head at the served shapes,
  `dense`, `denseGroup` (fused 4 and 3 parts) at 1 to 80 rows and `prefillDense` at 1 to 130 rows, on identical input rows with their group
  sums. Bound: 2^-8 |y| (the bf16 output) plus (k/16 + 16) 2^-23 of sum |x| (|q s| + |b|).

The Python engine runs the experts and projections as CUDA kernels these are compiled from, so there is no separate captured reference
for them: the printed bound is the format's.

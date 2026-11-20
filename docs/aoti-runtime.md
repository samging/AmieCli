# The AOTI runtime

`initMod(name)` now defaults to the AOTInductor runtime of onnx2pt, which is faster than ONNX Runtime for the models it
can take. A numeric-vector ONNX model opens on it with no change to the script. Any other model opens on ONNX Runtime
as it always did. `initMod(name, "ort")` forces ONNX Runtime for a model that needs it, such as an int64 token-id
model, and `initMod(name, "aoti")` names the default explicitly. This page covers what the runtime takes, which
runtime a model gets, what the runtime needs on your machine, how it was measured, and what the onnx2pt side of it
does.

## Which runtime a model gets

| Call | What opens |
|---|---|
| `initMod(name)`, `initMod(name, "aoti")` | A numeric-vector ONNX model, from `AMP/models/imported/<name>` or a path, on AOTI. Any other ONNX model, and every other kind (intent, text, speech, package jar, group), on ONNX Runtime, with a console line saying why the model was not taken by AOTI |
| `initMod(name, "ort")` | Exactly what `initMod(name)` opened before this default existed, with AOTI never tried |

A model that AOTI can't take is opened on ONNX Runtime and the console says why, for example
`initMod("…/ids.onnx") opened on ort: aoti can't run it, its input is INT64, not float32 …`. The same fallback covers a
machine where the AOTI native library or onnx2pt's compiler isn't set up. Passing `"ort"` keeps the console quiet.

## What it runs

A model that the AOTI runtime can take has **one float32 input and one float32 output**, with a static shape. amie
takes the numbers of a text as that input, comma or space separated:

```kotlin
val m = initMod("models/vecnet.onnx")             // AOTI by default; initMod("name") for AMP/models/imported/<name>
log(m.predict("0.1, 0.2, 0.3, 0.4, -0.1, -0.2, -0.3, -0.4"))   // green
log(m.answer("0.1, 0.2, 0.3, 0.4, -0.1, -0.2, -0.3, -0.4"))    // green:0.4808192
log(m.scores("0.1, 0.2, 0.3, 0.4, -0.1, -0.2, -0.3, -0.4"))    // every label with its score
m.close()
```

The labels come from a `config.json` beside the model (`id2label`), and the answers are formatted the way the
ONNX Runtime vector path formats them, so `predict`, `answer` and `scores` read the same on either runtime. The
`inputs` and `outputs` of the model are reported as `input` and `output`, because the compiled package keeps the
shapes but not the ONNX tensor names.

A model that doesn't fit is not refused: it opens on ONNX Runtime. What AOTI checks, through ONNX Runtime's metadata
before anything is compiled, is:

- one input and one output, both float32, with the input `[1, n]` (n known) and the output `[1, classes]`;
- so an int64 model with token ids, several inputs, or an image-shaped output stays on ONNX Runtime.

A check that fails costs a few milliseconds and is remembered for the process, so opening the same file again doesn't
repeat it. A compile or load that fails is remembered the same way.

Once a model is open, a call with a count of numbers that doesn't match the input still throws, as it does on ONNX
Runtime: `takes 8 values (its input is [1, 8]), got 3`. `initMod(name, runtime)` takes any other runtime name as an
error: `runtime is "ort" or "aoti", not "gpu"`.

## What it needs

The runtime has three pieces. The Kotlin side is in amie itself; the other two come from onnx2pt.

| Variable | What it names |
|---|---|
| `AOTI_FFI_LIB` | The native library, `libaoti_ffi.dylib` on macOS or `libaoti_ffi.so` on Linux. Built by onnx2pt's `onnx_aoti/build.sh`. amie doesn't bundle it. `-Daoti.ffi.lib=<path>` in `AMIE_JVM_ARGS` does the same |
| `AOTI_PYTHON` | The Python of onnx2pt's environment with torch and onnx, such as `<onnx2pt>/env/bin/python`. Without it amie looks for `env/bin/python` from the working directory up, then `python3` |
| `AOTI_COMPILER_SCRIPT` | onnx2pt's compiler, `<onnx2pt>/onnx_aoti/onnx_aoti.py`. Without it amie looks for `onnx_aoti/onnx_aoti.py` from the working directory up |

```sh
export AOTI_FFI_LIB=~/GitHubRepository/onnx2pt/onnx_aoti/rust/target/release/libaoti_ffi.dylib
export AOTI_PYTHON=~/GitHubRepository/onnx2pt/env/bin/python
export AOTI_COMPILER_SCRIPT=~/GitHubRepository/onnx2pt/onnx_aoti/onnx_aoti.py
```

amie itself runs on **JDK 25**, with Kotlin 2.3.21 for its scripts. The launcher passes
`--enable-native-access=ALL-UNNAMED`, which the runtime's `java.lang.foreign` calls need. Earlier builds of amie ran on
JDK 21 and can't open this runtime.

### The first open compiles

The first `initMod(name)` of a numeric-vector ONNX file runs onnx2pt's compiler once: it turns the ONNX file into a
`.pt2` package, which takes a while (the 512 by 512 model below takes minutes). The package is cached, by a hash of
the ONNX content, the input shape, the dtype and the compiler's sources. Later opens, in new processes too, load the
cached package without Python. Compiles give up after 30 minutes. The cache is `~/.cache/onnx_aoti`, or
`AOTI_CACHE_DIR`.

### Sessions

An AOTI model is held like any other session: it is listed on the Sessions tab as `aoti <name> #<id>`, it is closed
by the block around the call (`ortClosable`, a script's end), and `Terminate` on the Sessions tab closes it. A call on
a closed model throws `'<name>' is closed; initMod it again`. Calls on one model are serialized, and one session
object is reused for every call, so the per-call work is writing the input and running the package.

## How it was measured

Two comparisons were run. They measure different things, so they are reported separately.

### Through amie: a small MLP

`examples/aoti-vector/compare.kts` opens the same model twice, with `initMod(model, "ort")` and `initMod(model, "aoti")`,
and times `predict(text)` and `answer(text)` (median of five rounds of 2000 calls, after 300 warm-up calls). The
model is an 8 to 16 to 3 MLP that comes with the example.

| Call | ONNX Runtime | AOTI | AOTI / ORT |
|---|---|---|---|
| `predict(text)` | 50 to 59 µs | 5.8 to 6.5 µs | about 0.11 |
| `answer(text)` | 42 to 50 µs | 3.3 to 5.3 µs | about 0.10 |

Both runtimes gave the same answer, `green:0.4808192`. Two runs agreed. The times include parsing the text, because
that is part of the public call. The model is tiny, so most of the gap is per-call overhead in ONNX Runtime's Java
binding rather than compute.

Run it with the three variables above set:

```sh
bin/amie examples/aoti-vector/compare.kts
```

`VEC_MODEL`, `VEC_INPUT` and `VEC_CALLS` change the model, the text and the number of calls.

### On the same CPU: a 512 by 512 model

This comparison is outside amie. It is a 145 MB fp16 model (promoted to float32 for compilation), run at
`1×3×512×512` with the same input on both runtimes, on a 10-core CPU:

| Runtime | Runs | Mean | Median | p90 |
|---|---|---|---|---|
| ONNX Runtime 1.32.0, CPU | 100, after 20 warm-up | 2414.5 ms | 2405.0 ms | 2486.2 ms |
| AOTI, Kotlin `session.run()` | 1000, after 20 warm-up | 399.2 ms | 397.9 ms | 400.5 ms |

The AOTI package runs about six times faster on the mean. The ONNX Runtime run is 100 iterations rather than 1000
because 1000 would have taken about 40 minutes.

## The onnx2pt side

The compiler is `onnx_aoti/onnx_aoti.py` in onnx2pt. It takes an ONNX file, optionally with a fixed input shape, and
writes a `.pt2`. Four things had to change to get the 512 by 512 model through it.

**Folding the shape graph.** The model computes its Resize sizes from a chain of Shape, Gather, Cast and Concat
nodes. When the input shape is pinned, `specialize_static_shape` runs ONNX shape inference and onnxslim in turns,
until no Shape node is left. One pass left 21, two passes left none.

**Scalar Gathers.** onnx2pytorch indexes with a 0-d index tensor, which torch.export records as a data-dependent
`.item()` that Inductor can't lower. `scalar_gathers_to_vectors` rewrites each such Gather as a one-element index
followed by a Squeeze on the same axis, which gives the same values.

**A Gather converter.** `_gather` in `onnx_op_handler.py` replaces onnx2pytorch's Gather with `index_select` on a
flat, normalized index, reshaped to ONNX's output shape. It handles scalar, 1-D, N-D and negative indices, and it
matches ONNX Runtime exactly on those cases. It also removes the PyTorch "non-tuple sequence" deprecation warning
that onnx2pytorch's Gather raised.

**DeformConv in plain ATen.** The model's 20 DeformConv layers were converted to `torchvision.deform_conv2d`. Inductor
has no C shim for that op, so the package called back into Python for each of them. The Rust and Kotlin hosts have no
such callback, so the package could not run there. `_deform_conv` now expresses DeformConv as `grid_sample` for the
sampling, the optional mask applied per offset group, and one grouped `conv2d` with kernel-sized stride. The package
compiles with zero proxy-executor calls. The converter matches ONNX Runtime to about 1e-7 on five cases: bias, mask,
stride and dilation, `group` and `offset_group`, and the 1×1 kernel the model uses. The base sampling positions are
computed from the traced static shapes, not at conversion time, because the converter sees only the node.

### Results on the 512 by 512 model

- **Parity** against ONNX Runtime with `parity_check.py --atol 1e-2 --rtol 1e-2`: 0 of 262,144 elements outside the
  tolerance. The largest absolute difference is 5.1e-3, so the margin is about 2×. Mean absolute difference is 7.2e-4.
- **Kotlin hot loop**: a 3.6 s load, then 399 ms mean per `session.run()`.
- **Cross-check**: the Kotlin and Python runners give the same first five outputs for the same input.

## Limits

- **One float32 input and one float32 output**, with a static shape. That is what the compiler and the Rust
  binding take. Token ids and multi-input models open on ONNX Runtime, by default and with `"ort"`.
- **Packages must have no fallbacks.** The Rust and Kotlin hosts can't call back into Python, so an op that Inductor
  can't lower to C ends the run with `AOTInductorModelContainerRun failed`. The DeformConv rewrite exists for this.
- **The measurements are CPU only**, on one 10-core Mac.
- **Recompiling is needed for the newest converter changes.** The Gather override and the scalar-Gather
  de-duplication were added after the 512 package was built, so that package keeps the older graph. It gives the same
  output. A fresh compile picks up both.

## Building the pieces

```sh
# onnx2pt: the native library and the Kotlin jar that bundles it
cd ~/GitHubRepository/onnx2pt/onnx_aoti
LIBTORCH=~/GitHubRepository/onnx2pt/env/lib/python3.14/site-packages/torch ./build.sh

# amie: the runtime and the interpreter, on JDK 25
cd ~/GitHubRepository/amieCli
./gradlew :amie:installDist
```

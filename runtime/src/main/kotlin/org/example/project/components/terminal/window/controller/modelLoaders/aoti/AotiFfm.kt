package org.example.project.components.terminal.window.controller.modelLoaders.aoti

import java.lang.foreign.Arena
import java.lang.foreign.FunctionDescriptor
import java.lang.foreign.Linker
import java.lang.foreign.MemorySegment
import java.lang.foreign.SymbolLookup
import java.lang.foreign.ValueLayout.ADDRESS
import java.lang.foreign.ValueLayout.JAVA_BYTE
import java.lang.foreign.ValueLayout.JAVA_FLOAT
import java.lang.foreign.ValueLayout.JAVA_INT
import java.lang.foreign.ValueLayout.JAVA_LONG
import java.lang.invoke.MethodHandle
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.StandardCopyOption
import java.util.zip.ZipFile
import java.util.concurrent.locks.ReentrantReadWriteLock
import kotlin.concurrent.read
import kotlin.concurrent.write

class AotiException(message: String) : RuntimeException(message)

/**
 * Binding to the Rust `aoti_ffi` library, using the Java Foreign Function & Memory API.
 */
class AotiLibrary(libPath: Path) {
    private val linker = Linker.nativeLinker()
    private val lookup: SymbolLookup = SymbolLookup.libraryLookup(libPath, Arena.global())

    private val loadHandle = handle("aoti_load", FunctionDescriptor.of(ADDRESS, ADDRESS, JAVA_LONG))
    private val freeHandle = handle("aoti_free", FunctionDescriptor.ofVoid(ADDRESS))
    private val runIntoHandle = handle(
        "aoti_run_into",
        FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG),
    )
    private val runViewHandle = handle(
        "aoti_run_view",
        FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS),
    )
    private val resultFreeHandle = handle("aoti_result_free", FunctionDescriptor.ofVoid(ADDRESS))
    private val lastErrorHandle = handle("aoti_last_error", FunctionDescriptor.of(JAVA_LONG, ADDRESS, JAVA_LONG))

    private fun handle(name: String, descriptor: FunctionDescriptor): MethodHandle {
        val address = lookup.find(name).orElseThrow { AotiException("missing symbol $name") }
        return linker.downcallHandle(address, descriptor)
    }

    /**
     * Loads a `.pt2` package and reads its shape metadata. [numRunners] is the
     * number of instances that can run concurrently. Close the returned model when done.
     */
    fun load(modelPath: Path, numRunners: Int = 1): Model {
        require(numRunners >= 1) { "numRunners must be at least 1" }
        val meta = readMeta(modelPath)
        Arena.ofConfined().use { arena ->
            val path = arena.allocateFrom(modelPath.toAbsolutePath().toString())
            val model = loadHandle.invokeWithArguments(path, numRunners.toLong()) as MemorySegment
            if (model.address() == 0L) throw AotiException(lastError())
            return Model(model, meta.inputShape, meta.outputShape)
        }
    }

    /**
     * Loads an ONNX model through the content-addressed `.pt2` cache. On a cache
     * miss, runs the Python compiler once. Later calls, including in new processes,
     * load the cached package without Python.
     *
     * Defaults: cache in `~/.cache/onnx_aoti` (`AOTI_CACHE_DIR`), Python from
     * `env/bin/python` (`AOTI_PYTHON`), compiler `onnx_aoti/onnx_aoti.py`
     * (`AOTI_COMPILER_SCRIPT`). Defaults are searched from the working directory upward.
     */
    fun loadOnnx(
        onnxPath: Path,
        inputShape: LongArray? = null,
        dtype: AotiDtype = AotiDtype.FLOAT32,
        numRunners: Int = 1,
        cacheDir: Path = OnnxCache.defaultCacheDir(),
        pythonBin: Path = OnnxCache.defaultPythonBin(),
        compilerScript: Path = OnnxCache.defaultCompilerScript(),
    ): Model {
        val onnx = onnxPath.toAbsolutePath().normalize()
        require(Files.isRegularFile(onnx)) { "ONNX model not found: $onnx" }
        Files.createDirectories(cacheDir)

        val key = OnnxCache.cacheKey(onnx, inputShape, dtype, compilerScript, cacheDir)
        val cached = cacheDir.resolve("$key.pt2")
        if (!Files.isRegularFile(cached)) {
            OnnxCache.ensureCompiled(onnx, cached, key, inputShape, dtype, pythonBin, compilerScript)
        }
        return load(cached, numRunners)
    }

    private fun lastError(): String = Arena.ofConfined().use { arena ->
        val size = 4096L
        val buf = arena.allocate(JAVA_BYTE, size)
        val length = (lastErrorHandle.invokeWithArguments(buf, size) as Long).coerceAtMost(size - 1)
        val bytes = ByteArray(length.toInt())
        MemorySegment.copy(buf, JAVA_BYTE, 0L, bytes, 0, bytes.size)
        String(bytes, Charsets.UTF_8)
    }

    private class Meta(val inputShape: LongArray, val outputShape: LongArray)

    private fun readMeta(modelPath: Path): Meta {
        val text = ZipFile(modelPath.toFile()).use { zip ->
            val entry = zip.getEntry(META_NAME)
                ?: throw AotiException("$META_NAME missing in $modelPath; recompile with onnx_aoti.compile_onnx_to_pt2")
            zip.getInputStream(entry).use { it.readBytes().toString(Charsets.UTF_8) }
        }
        return Meta(jsonLongArray(text, "input_shape"), jsonLongArray(text, "output_shape"))
    }

    /** Reads a flat integer array field from the metadata JSON. */
    private fun jsonLongArray(json: String, key: String): LongArray {
        val match = Regex("\"$key\"\\s*:\\s*\\[([^\\]]*)\\]").find(json)
            ?: throw AotiException("$META_NAME has no $key")
        val body = match.groupValues[1].trim()
        if (body.isEmpty()) return LongArray(0)
        return body.split(',').map { it.trim().toLong() }.toLongArray()
    }

    /**
     * A dynamic-shape output in native memory. [data] is valid only until [close] is called.
     */
    private inner class View(
        private var raw: MemorySegment?,
        val shape: LongArray,
        val data: MemorySegment,
    ) : AutoCloseable {
        override fun close() {
            val handle = raw ?: return
            resultFreeHandle.invokeWithArguments(handle)
            raw = null
        }
    }

    /**
     * A loaded model. Calls can run concurrently from several threads.
     * [close] waits for in-flight calls to finish before freeing the model.
     */
    inner class Model internal constructor(
        private var handle: MemorySegment?,
        val inputShape: LongArray,
        val outputShape: LongArray,
    ) : AutoCloseable {
        private val lock = ReentrantReadWriteLock()

        val inputNumel: Long = inputShape.fold(1L) { acc, d -> acc * d }
        val outputNumel: Long = outputShape.fold(1L) { acc, d -> acc * d }

        fun allocateInput(): FloatTensor = FloatTensor.zeros(*inputShape)

        fun allocateOutput(): FloatTensor = FloatTensor.zeros(*outputShape)

        /** Runs the model on [input] and returns a new tensor with the output. */
        fun predict(input: FloatTensor): FloatTensor {
            if (input.shape[0] == 0L) {
                val out = FloatTensor.zeros(*outputShape)
                return out
            }
            requireShape(input, inputShape, "input")
            return withModel { model ->
                Arena.ofConfined().use { arena ->
                    val view = runInternal(model, input.segment, input.shape, arena)
                    try {
                        val out = FloatTensor.zeros(*view.shape)
                        MemorySegment.copy(view.data, 0L, out.segment, 0L, view.data.byteSize())
                        out
                    } finally {
                        view.close()
                    }
                }
            }
        }

        /** A session with its own pre-allocated input and output tensors. */
        fun newSession(): InferenceSession {
            if (outputNumel == 0L) {
                // Zero outputs: run is a no-op with nothing to read back.
                return InferenceSession(this, allocateInput(), FloatTensor.zeros(0))
            }
            return InferenceSession(this, allocateInput(), allocateOutput())
        }

        /** Zero-copy inference into [output], which must have the model's output shape. */
        internal fun runInto(input: FloatTensor, output: FloatTensor) {
            if (input.shape[0] == 0L) return   // AOTInductor allocates outputs; skip a no-op run.
            requireShape(input, inputShape, "input")
            requireShape(output, outputShape, "output")
            withModel { model ->
                Arena.ofConfined().use { arena ->
                    val shapeSeg = copyShape(arena, input.shape)
                    val status = runIntoHandle.invokeWithArguments(
                        model,
                        input.segment,
                        floatCount(input.segment),
                        shapeSeg,
                        input.shape.size.toLong(),
                        output.segment,
                        floatCount(output.segment),
                    ) as Int
                    if (status != 0) throw AotiException(lastError())
                }
            }
        }

        /** Runs the model, returns its output. Zero-copy where the engine provides it. */
        private fun runInternal(model: MemorySegment, input: MemorySegment, shape: LongArray, arena: Arena): View {
            val shapeSeg = copyShape(arena, shape)
            // struct AotiOutput: data_ptr, data_len, shape_ptr, shape_len, raw_handle
            val out = arena.allocate(40L, 8L)
            val status = runViewHandle.invokeWithArguments(
                model,
                input,
                floatCount(input),
                shapeSeg,
                shape.size.toLong(),
                out,
            ) as Int
            if (status != 0) throw AotiException(lastError())

            val dataPtr = out.get(ADDRESS, 0L).address()
            val dataLen = out.get(JAVA_LONG, 8L)
            val shapePtr = out.get(ADDRESS, 16L).address()
            val shapeLen = out.get(JAVA_LONG, 24L)
            val raw = out.get(ADDRESS, 32L)

            val outShape = MemorySegment.ofAddress(shapePtr).reinterpret(shapeLen * 8).toArray(JAVA_LONG)
            val data = MemorySegment.ofAddress(dataPtr).reinterpret(dataLen * 4)
            return View(raw, outShape, data)
        }

        private inline fun <T> withModel(block: (MemorySegment) -> T): T = lock.read {
            block(handle ?: throw IllegalStateException("model is closed"))
        }

        override fun close() {
            lock.write {
                val model = handle ?: return
                freeHandle.invokeWithArguments(model)
                handle = null
            }
        }

        private fun requireShape(tensor: FloatTensor, expected: LongArray, name: String) {
            require(tensor.shape.contentEquals(expected)) {
                "$name shape ${tensor.shape.contentToString()} does not match model $name shape ${expected.contentToString()}"
            }
        }
    }

    companion object {
        /**
         * Loads the bundled native library, or the one named by the system property
         * `aoti.ffi.lib` or the environment variable `AOTI_FFI_LIB`.
         */
        fun default(): AotiLibrary = AotiLibrary(resolveLibrary())

        private fun resolveLibrary(): Path {
            System.getProperty("aoti.ffi.lib")?.let { return Path.of(it) }
            System.getenv("AOTI_FFI_LIB")?.let { return Path.of(it) }

            val name = System.mapLibraryName("aoti_ffi")
            val resource = "/native/${platformDir()}/$name"
            val stream = AotiLibrary::class.java.getResourceAsStream(resource)
                ?: throw AotiException("native library not bundled at $resource; set -Daoti.ffi.lib=<path>")
            val dir = Files.createTempDirectory("aoti_ffi")
            val file = dir.resolve(name)
            stream.use { Files.copy(it, file, StandardCopyOption.REPLACE_EXISTING) }
            file.toFile().deleteOnExit()
            dir.toFile().deleteOnExit()
            return file
        }

        private fun platformDir(): String {
            val os = System.getProperty("os.name").lowercase()
            val arch = System.getProperty("os.arch").lowercase()
            val osName = when {
                "mac" in os -> "macos"
                "linux" in os -> "linux"
                "windows" in os -> "windows"
                else -> throw AotiException("unsupported OS: $os")
            }
            val archName = when (arch) {
                "aarch64", "arm64" -> "aarch64"
                "amd64", "x86_64" -> "x86_64"
                else -> throw AotiException("unsupported architecture: $arch")
            }
            return "$osName-$archName"
        }

        private const val META_NAME = "model_meta.json"
    }

    private fun copyShape(arena: Arena, shape: LongArray): MemorySegment {
        val seg = arena.allocate(JAVA_LONG, shape.size.toLong())
        MemorySegment.copy(shape, 0, seg, JAVA_LONG, 0L, shape.size)
        return seg
    }

    private fun floatCount(seg: MemorySegment): Long = seg.byteSize() / JAVA_FLOAT.byteSize()
}

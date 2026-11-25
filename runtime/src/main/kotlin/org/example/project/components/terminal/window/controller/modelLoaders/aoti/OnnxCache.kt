package org.example.project.components.terminal.window.controller.modelLoaders.aoti

import java.io.IOException
import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.StandardCopyOption
import java.security.MessageDigest
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.TimeUnit

enum class AotiDtype(val cliName: String) {
    FLOAT32("float32"),
    FLOAT16("float16"),
    BFLOAT16("bfloat16"),
}

/**
 * Content-addressed cache of compiled `.pt2` packages.
 *
 * The cache key covers the ONNX content (plus the `.onnx.data` file if present),
 * the input shape, the dtype, and the Python compiler sources. Python runs only on
 * a cache miss.
 *
 * Hashing a large model is slow, so the content hash is also stored in `index.tsv`
 * in the cache directory, keyed by canonical path, size, and modification time.
 * A file whose size and mtime are unchanged is not read again.
 */
internal object OnnxCache {
    private val compileLocks = ConcurrentHashMap<String, Any>()
    private val indexLock = Any()
    private const val COMPILE_TIMEOUT_MINUTES = 30L
    private const val INDEX_NAME = "index.tsv"

    fun defaultCacheDir(): Path =
        System.getenv("AOTI_CACHE_DIR")?.let(Path::of)
            ?: Path.of(System.getProperty("user.home"), ".cache", "onnx_aoti")

    fun defaultPythonBin(): Path =
        System.getenv("AOTI_PYTHON")?.let(Path::of)
            ?: findUp("env/bin/python")
            ?: Path.of("python3")

    fun defaultCompilerScript(): Path =
        System.getenv("AOTI_COMPILER_SCRIPT")?.let(Path::of)
            ?: findUp("onnx_aoti/onnx_aoti.py")
            ?: Path.of("onnx_aoti/onnx_aoti.py")

    /** Looks for [relative] in the working directory and its parents. */
    private fun findUp(relative: String): Path? {
        var dir: Path? = Path.of("").toAbsolutePath()
        while (dir != null) {
            val candidate = dir.resolve(relative)
            if (Files.exists(candidate)) return candidate
            dir = dir.parent
        }
        return null
    }

    /** Full cache key for one ONNX file, input shape, dtype, and compiler version. */
    fun cacheKey(
        onnxPath: Path,
        inputShape: LongArray?,
        dtype: AotiDtype,
        compilerScript: Path,
        cacheDir: Path,
    ): String {
        val digest = MessageDigest.getInstance("SHA-256")
        fun updateString(s: String) {
            digest.update(s.toByteArray(StandardCharsets.UTF_8))
            digest.update(0)
        }

        updateString("content=${contentHash(onnxPath, cacheDir)}")
        updateString("dtype=${dtype.cliName}")
        updateString("shape=${inputShape?.joinToString(",") ?: "auto"}")
        updateString("compiler=${compilerHash(compilerScript)}")
        return hex(digest.digest())
    }

    /**
     * SHA-256 of the ONNX file and its `.onnx.data` companion. Reads the file
     * only when its canonical path, size, or modification time is not in the index.
     */
    fun contentHash(onnxPath: Path, cacheDir: Path): String {
        val fingerprint = fingerprint(onnxPath)
        val index = cacheDir.resolve(INDEX_NAME)
        lookupIndex(index, fingerprint)?.let { return it }

        val digest = MessageDigest.getInstance("SHA-256")
        updateFile(digest, onnxPath)
        val externalData = dataFile(onnxPath)
        if (Files.isRegularFile(externalData)) {
            digest.update("external_data".toByteArray(StandardCharsets.UTF_8))
            updateFile(digest, externalData)
        }
        val hash = hex(digest.digest())
        appendIndex(index, fingerprint, hash)
        return hash
    }

    private fun fingerprint(onnxPath: Path): String {
        val parts = mutableListOf(onnxPath.toRealPath().toString())
        for (file in listOfNotNull(onnxPath, dataFile(onnxPath).takeIf(Files::isRegularFile))) {
            parts += Files.size(file).toString()
            parts += Files.getLastModifiedTime(file).toMillis().toString()
        }
        return parts.joinToString("|")
    }

    private fun dataFile(onnxPath: Path): Path = onnxPath.resolveSibling("${onnxPath.fileName}.data")

    private fun lookupIndex(index: Path, fingerprint: String): String? {
        if (!Files.isRegularFile(index)) return null
        var found: String? = null
        // The last entry wins, so a rewritten file overrides older entries.
        for (line in Files.readAllLines(index)) {
            val tab = line.lastIndexOf('\t')
            if (tab > 0 && line.substring(0, tab) == fingerprint) found = line.substring(tab + 1)
        }
        return found
    }

    private fun appendIndex(index: Path, fingerprint: String, hash: String) {
        synchronized(indexLock) {
            Files.createDirectories(index.parent)
            Files.writeString(
                index,
                "$fingerprint\t$hash\n",
                StandardCharsets.UTF_8,
                java.nio.file.StandardOpenOption.CREATE,
                java.nio.file.StandardOpenOption.APPEND,
            )
        }
    }

    private fun compilerHash(compilerScript: Path): String {
        val digest = MessageDigest.getInstance("SHA-256")
        Files.list(compilerScript.toAbsolutePath().parent).use { files ->
            files.filter { it.fileName.toString().endsWith(".py") }.sorted().forEach { file ->
                digest.update(file.fileName.toString().toByteArray(StandardCharsets.UTF_8))
                digest.update(0)
                updateFile(digest, file)
            }
        }
        return hex(digest.digest())
    }

    private fun updateFile(digest: MessageDigest, file: Path) {
        Files.newInputStream(file).use { input ->
            val buf = ByteArray(1024 * 1024)
            while (true) {
                val read = input.read(buf)
                if (read <= 0) break
                digest.update(buf, 0, read)
            }
        }
    }

    private fun hex(bytes: ByteArray): String = bytes.joinToString("") { "%02x".format(it) }

    /** The ExportedProgram saved beside a cached package, `<key>.program.pt2`: what the package was compiled from */
    fun programOf(cachedPt2: Path): Path = cachedPt2.resolveSibling(cachedPt2.fileName.toString().removeSuffix(".pt2") + ".program.pt2")

    /** Compiles [onnxPath] into [cachedPt2] and its ExportedProgram unless another caller already did. */
    fun ensureCompiled(
        onnxPath: Path,
        cachedPt2: Path,
        key: String,
        inputShape: LongArray?,
        dtype: AotiDtype,
        pythonBin: Path,
        compilerScript: Path,
    ) {
        val lock = compileLocks.computeIfAbsent(key) { Any() }
        synchronized(lock) {
            if (Files.isRegularFile(cachedPt2)) return
            compile(onnxPath, cachedPt2, key, inputShape, dtype, pythonBin, compilerScript)
        }
    }

    /**
     * Compiles [onnxPath] again into [cachedPt2] and its ExportedProgram, for an export of a model that has a package but no
     * program: a package compiled before programs were saved. The input is the default one the runtime opens with.
     */
    fun compileForExport(
        onnxPath: Path,
        cachedPt2: Path,
        key: String,
        pythonBin: Path = defaultPythonBin(),
        compilerScript: Path = defaultCompilerScript(),
    ) {
        Files.createDirectories(cachedPt2.parent)
        val lock = compileLocks.computeIfAbsent(key) { Any() }
        synchronized(lock) {
            compile(onnxPath, cachedPt2, key, null, AotiDtype.FLOAT32, pythonBin, compilerScript)
        }
    }

    private fun compile(
        onnxPath: Path,
        cachedPt2: Path,
        key: String,
        inputShape: LongArray?,
        dtype: AotiDtype,
        pythonBin: Path,
        compilerScript: Path,
    ) {
        require(Files.isRegularFile(compilerScript)) {
            "Compiler script not found at $compilerScript (set AOTI_COMPILER_SCRIPT)"
        }
        val cacheDir = cachedPt2.parent
        val tempPt2 = Files.createTempFile(cacheDir, "$key.", ".tmp.pt2")
        val tempProgram = Files.createTempFile(cacheDir, "$key.", ".tmp.program.pt2")
        val log = Files.createTempFile(cacheDir, "$key.", ".log")
        try {
            val cmd = mutableListOf(
                pythonBin.toString(),
                compilerScript.toAbsolutePath().toString(),
                "--input", onnxPath.toString(),
                "--output", tempPt2.toAbsolutePath().toString(),
                "--program", tempProgram.toAbsolutePath().toString(),
                "--dtype", dtype.cliName,
            )
            if (inputShape != null) cmd += listOf("--input-shape", inputShape.joinToString(","))

            System.err.println("onnx_aoti: compiling ${onnxPath.fileName} (cache miss, key ${key.take(12)})")
            val process = try {
                ProcessBuilder(cmd)
                    .redirectErrorStream(true)
                    .redirectOutput(ProcessBuilder.Redirect.to(log.toFile()))
                    .start()
            } catch (e: IOException) {
                throw AotiException("could not start Python at $pythonBin: ${e.message}")
            }

            if (!process.waitFor(COMPILE_TIMEOUT_MINUTES, TimeUnit.MINUTES)) {
                process.destroyForcibly()
                throw AotiException("AOTInductor compilation of $onnxPath timed out after $COMPILE_TIMEOUT_MINUTES minutes")
            }
            val exitCode = process.exitValue()
            if (exitCode != 0 || Files.size(tempPt2) == 0L) {
                val output = Files.readString(log)
                throw AotiException("AOTInductor compilation failed for $onnxPath (exit code $exitCode):\n$output")
            }

            Files.move(tempProgram, programOf(cachedPt2), StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING)
            Files.move(tempPt2, cachedPt2, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING)
        } finally {
            Files.deleteIfExists(tempPt2)
            Files.deleteIfExists(tempProgram)
            Files.deleteIfExists(log)
        }
    }
}

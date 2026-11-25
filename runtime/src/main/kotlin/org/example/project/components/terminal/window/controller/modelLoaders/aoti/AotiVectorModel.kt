package org.example.project.components.terminal.window.controller.modelLoaders.aoti

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import org.example.project.components.terminal.integrations.amiePilot.InteractionModel
import org.example.project.components.terminal.integrations.amiePilot.Tensor
import org.example.project.components.terminal.integrations.amiePilot.TensorSpec
import org.example.project.components.terminal.window.controller.modelLoaders.OrtSessions
import org.example.project.components.terminal.window.controller.modelLoaders.imported.ImportedModels
import org.example.project.components.terminal.window.controller.modelLoaders.imported.ModelFormat
import org.example.project.components.terminal.window.controller.modelLoaders.imported.ModelFormats
import org.example.project.components.terminal.window.controller.modelLoaders.imported.ModelText
import java.io.File
import java.nio.file.Path
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicLong

/**
 * A numeric-vector ONNX model on the AOTInductor runtime: one float32 input, one float32 output, compiled once by
 * onnx2pt's onnx_aoti. It answers what onnxModel's vector path answers, the numbers of the text in and the labels
 * with their scores out, through one InferenceSession made at open: a call writes the input and runs the package.
 */
class AotiVectorModel internal constructor(
    val name: String,
    val path: Path,
    private val model: AotiLibrary.Model,
    private val labels: Map<Int, String>,
) : InteractionModel {
    val id: Long = nextId.incrementAndGet()
    private val session = model.newSession()
    private val lock = Any()
    @Volatile var closed = false
        private set
    @Volatile var runs = 0
        private set
    @Volatile var running = 0
        private set

    val busy: Boolean get() = running > 0

    private fun <T> locked(block: () -> T): T = synchronized(lock) {
        check(!closed) { "'$name' is closed; initMod it again" }
        running++
        try {
            runs++
            block()
        } finally {
            running--
        }
    }

    private fun scoresOf(values: FloatArray): FloatArray {
        require(values.size.toLong() == model.inputNumel) {
            "$name takes ${model.inputNumel} values (its input is ${model.inputShape.toList()}), got ${values.size}"
        }
        session.input.copyFrom(values)
        return session.run().toFloatArray()
    }

    override fun inference(text: String): List<String> {
        val values = ModelText.numbers(text, name)
        return locked { ModelText.classScores(scoresOf(values), labels) }
    }

    override fun run(values: FloatArray): Map<String, Tensor> = locked {
        mapOf("output" to Tensor("output", model.outputShape.toList(), "float32", scoresOf(values)))
    }

    override val inputs: List<TensorSpec> get() = listOf(TensorSpec("input", model.inputShape.toList(), "float32"))

    override val outputs: List<TensorSpec> get() = listOf(TensorSpec("output", model.outputShape.toList(), "float32"))

    override fun getStatus(): Any = mapOf(
        "name" to name,
        "path" to path.toString(),
        "runtime" to "aoti",
        "trainable" to false,
        "inputShape" to model.inputShape.toList(),
        "outputShape" to model.outputShape.toList(),
        "labels" to labels.toSortedMap().values.toList(),
    )

    override fun close() {
        synchronized(lock) {
            if (closed) return
            closed = true
            session.close()
            model.close()
        }
    }

    companion object {
        private val nextId = AtomicLong()
    }
}

/** What [AotiModels.tryOpen] made of a name: a model on the AOTI runtime, a name that is no ONNX file, or why it can't run there */
sealed interface AotiAttempt {
    class Opened(val model: AotiVectorModel) : AotiAttempt
    object NotOnnx : AotiAttempt
    class Refused(val reason: String) : AotiAttempt
}

/**
 * Opens the ONNX models of AMP/models/imported (by folder name) or by path on the AOTI runtime. A model is checked
 * through ONNX Runtime's metadata first, so a text model with token ids is refused without a compile; the verdict of
 * each file, and a compile that failed, are kept for the process, so the next open of the same file doesn't repeat them.
 */
object AotiModels {
    private class Verdict(val stamp: String, val reason: String?)

    private val verdicts = ConcurrentHashMap<String, Verdict>()

    fun tryOpen(modelName: String): AotiAttempt {
        val name = modelName.trim()
        BundledAoti.entry(name)?.let { return openBundled(name, it) }
        val (onnx, top) = onnxOf(name) ?: return AotiAttempt.NotOnnx
        val key = onnx.absolutePath
        val stamp = "${onnx.length()}|${onnx.lastModified()}"
        val known = verdicts[key]?.takeIf { it.stamp == stamp }
        val reason = if (known != null) known.reason else inspect(onnx).also { verdicts[key] = Verdict(stamp, it) }
        if (reason != null) return AotiAttempt.Refused(reason)
        return try {
            val labels = ModelText.labels(ImportedModels.homesOf(onnx, top))
            AotiAttempt.Opened(AotiVectorModel(name, onnx.toPath(), AotiLibrary.default().loadOnnx(onnx.toPath()), labels))
        } catch (e: Exception) {
            val why = "the compile or the load failed: ${e.message?.lineSequence()?.firstOrNull()}"
            verdicts[key] = Verdict(stamp, why)
            AotiAttempt.Refused(why)
        }
    }

    /** A model a bundle carries: its package loaded as it is, its labels from the copy of its config beside it */
    private fun openBundled(name: String, entry: BundledAoti.Entry): AotiAttempt = try {
        val labels = entry.labels?.let(ModelText::labelsOf) ?: emptyMap()
        AotiAttempt.Opened(AotiVectorModel(name, entry.pt2.toPath(), AotiLibrary.default().load(entry.pt2.toPath()), labels))
    } catch (e: Exception) {
        AotiAttempt.Refused("the bundled package ${entry.pt2.name} can't be loaded: ${e.message?.lineSequence()?.firstOrNull()}")
    }

    /** The ONNX file a name opens, and the folder its labels and tokenizer are looked for up to; null for a name that isn't one */
    private fun onnxOf(name: String): Pair<File, File?>? {
        val (location, top) = when {
            ImportedModels.has(name) -> ImportedModels.folder(name)!!.let { it to it }
            ImportedModels.looksLikePath(name) -> ImportedModels.expand(name).let { f -> f to f.takeIf { it.isDirectory } }
            else -> return null
        }
        if (!location.exists()) return null
        val scan = if (location.isDirectory) ModelFormats.scan(location) else ImportedModels.scanPath(location)
        val onnx = scan.of(ModelFormat.ONNX).firstOrNull()?.file ?: return null
        return onnx to top
    }

    /** Why [onnx] can't run here, null when it can: one float32 input and one float32 output, both `[1, n]` with n known */
    private fun inspect(onnx: File): String? = try {
        OrtSessions.env.createSession(onnx.absolutePath, OrtSession.SessionOptions()).use { session ->
            val inputs = session.inputInfo.values.map { it.info as? TensorInfo }
            val outputs = session.outputInfo.values.map { it.info as? TensorInfo }
            when {
                inputs.size != 1 || outputs.size != 1 -> "it has ${inputs.size} inputs and ${outputs.size} outputs; aoti takes one of each"
                inputs.single() == null || outputs.single() == null -> "its inputs or outputs are not tensors"
                inputs.single()!!.type != OnnxJavaType.FLOAT -> "its input is ${inputs.single()!!.type}, not float32 (token ids and other integers stay on ort)"
                outputs.single()!!.type != OnnxJavaType.FLOAT -> "its output is ${outputs.single()!!.type}, not float32"
                !vector(inputs.single()!!.shape) -> "its input is ${inputs.single()!!.shape.toList()}; aoti takes a [1, n] vector with n known"
                !vector(outputs.single()!!.shape) -> "its output is ${outputs.single()!!.shape.toList()}; aoti takes a [1, classes] answer"
                else -> null
            }
        }
    } catch (e: Exception) {
        "it can't be read by ONNX Runtime: ${e.message?.lineSequence()?.firstOrNull()}"
    }

    private fun vector(shape: LongArray): Boolean = shape.size == 2 && shape[0] <= 1 && shape[1] > 0
}

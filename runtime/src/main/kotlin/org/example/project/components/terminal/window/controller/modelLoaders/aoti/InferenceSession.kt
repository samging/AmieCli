package org.example.project.components.terminal.window.controller.modelLoaders.aoti

/**
 * Input and output tensors allocated once for a model. Reuse one session
 * across calls to avoid allocating per inference.
 *
 * A session is not thread-safe on its own. Use one session per thread.
 * Closing a session does not close the model.
 */
class InferenceSession internal constructor(
    private val model: AotiLibrary.Model,
    val input: FloatTensor,
    val output: FloatTensor,
) : AutoCloseable {

    /** Runs the model on [input] and writes the result to [output]. */
    fun run(): FloatTensor {
        model.runInto(input, output)
        return output
    }

    override fun close() {
        input.close()
        output.close()
    }
}

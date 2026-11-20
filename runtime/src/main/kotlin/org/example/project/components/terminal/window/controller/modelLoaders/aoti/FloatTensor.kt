package org.example.project.components.terminal.window.controller.modelLoaders.aoti

import java.lang.foreign.Arena
import java.lang.foreign.MemorySegment
import java.lang.foreign.ValueLayout.JAVA_FLOAT

/**
 * A float32 tensor in native memory. It owns its memory and frees it on [close].
 */
class FloatTensor private constructor(
    val shape: LongArray,
    private val arena: Arena,
    internal val segment: MemorySegment,
) : AutoCloseable {

    val numel: Long = shape.fold(1L) { acc, dim -> acc * dim }

    private var closed = false

    fun copyFrom(data: FloatArray) {
        check(!closed) { "tensor is closed" }
        require(data.size.toLong() == numel) {
            "Expected $numel elements for shape ${shape.contentToString()}, got ${data.size}"
        }
        MemorySegment.copy(data, 0, segment, JAVA_FLOAT, 0L, data.size)
    }

    fun toFloatArray(): FloatArray {
        check(!closed) { "tensor is closed" }
        return segment.toArray(JAVA_FLOAT)
    }

    operator fun get(index: Long): Float {
        check(!closed) { "tensor is closed" }
        return segment.getAtIndex(JAVA_FLOAT, index)
    }

    operator fun set(index: Long, value: Float) {
        check(!closed) { "tensor is closed" }
        segment.setAtIndex(JAVA_FLOAT, index, value)
    }

    override fun close() {
        if (closed) return
        closed = true
        arena.close()
    }

    companion object {
        /** Zeroed tensor. Uses a shared arena so it can move between threads. */
        fun zeros(vararg shape: Long): FloatTensor {
            val numel = shape.fold(1L) { acc, dim -> acc * dim }
            val arena = Arena.ofShared()
            val segment = arena.allocate(JAVA_FLOAT, numel)
            return FloatTensor(shape.copyOf(), arena, segment)
        }

        fun of(data: FloatArray, vararg shape: Long): FloatTensor =
            zeros(*shape).apply { copyFrom(data) }
    }
}

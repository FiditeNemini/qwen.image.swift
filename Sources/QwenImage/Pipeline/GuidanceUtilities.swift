import MLX

enum GuidanceUtilities {
  static func applyClassifierFreeGuidance(
    unconditional: MLXArray,
    conditional: MLXArray,
    guidanceScale: Float
  ) -> MLXArray {
    let scale = MLXArray(Float(guidanceScale)).asType(unconditional.dtype)
    let guided = unconditional + (conditional - unconditional) * scale
    return guided.asType(unconditional.dtype)
  }

  static func stackLatentsForGuidance(_ latents: MLXArray) -> MLXArray {
    MLX.concatenated([latents, latents], axis: 0)
  }

  static func splitGuidanceLatents(_ latents: MLXArray) -> (unconditional: MLXArray, conditional: MLXArray) {
    let batch = latents.dim(0)
    precondition(batch >= 2, "Guidance latents require at least two samples.")
    let unconditional = latents[0 ..< 1, 0..., 0..., 0...]
    let conditional = latents[1..., 0..., 0..., 0...]
    return (unconditional, conditional)
  }
}

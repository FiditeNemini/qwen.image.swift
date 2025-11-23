import Foundation
import MLX
import MLXNN

final class QwenTimeTextEmbed: Module {
  @ModuleInfo(key: "time_proj") var timeProjection: QwenTimesteps
  @ModuleInfo(key: "timestep_embedder") var timestepEmbedder: QwenTimestepEmbedding

  init(timestepProjectionDim: Int = 256, innerDim: Int = 3072) {
    self._timeProjection.wrappedValue = QwenTimesteps(projectionDim: timestepProjectionDim)
    self._timestepEmbedder.wrappedValue = QwenTimestepEmbedding(
      projectionDim: timestepProjectionDim,
      innerDim: innerDim
    )
  }

  func callAsFunction(timestep: MLXArray, hiddenStates: MLXArray) -> MLXArray {
    let proj = timeProjection(timestep)
    let embedded = timestepEmbedder(proj.asType(hiddenStates.dtype))
    return embedded
  }
}

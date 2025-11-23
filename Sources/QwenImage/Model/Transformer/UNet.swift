import Foundation
import MLX
import MLXNN

public final class QwenUNet: Module {

  @ModuleInfo(key: "transformer") var transformer: QwenTransformer

  public init(configuration: QwenTransformerConfiguration) {
    self._transformer.wrappedValue = QwenTransformer(configuration: configuration)
    super.init()
  }

  public var configuration: QwenTransformerConfiguration {
    transformer.configuration
  }

  public func callAsFunction(
    timestepIndex: Int,
    runtimeConfig: QwenRuntimeConfig,
    latentTokens: MLXArray,
    encoderHiddenStates: MLXArray,
    encoderHiddenStatesMask: MLXArray,
    imageSegments: [(Int, Int, Int)]? = nil
  ) -> MLXArray {
    forwardTokens(
      timestepIndex: timestepIndex,
      runtimeConfig: runtimeConfig,
      latentTokens: latentTokens,
      encoderHiddenStates: encoderHiddenStates,
      encoderHiddenStatesMask: encoderHiddenStatesMask,
      imageSegments: imageSegments
    )
  }

  public func forwardTokens(
    timestepIndex: Int,
    runtimeConfig: QwenRuntimeConfig,
    latentTokens: MLXArray,
    encoderHiddenStates: MLXArray,
    encoderHiddenStatesMask: MLXArray,
    imageSegments: [(Int, Int, Int)]? = nil,
    precomputedImageRotaryEmbeddings: (MLXArray, MLXArray)? = nil
  ) -> MLXArray {
    return transformer.forward(
      timestepIndex: timestepIndex,
      runtimeConfig: runtimeConfig,
      hiddenStates: latentTokens,
      encoderHiddenStates: encoderHiddenStates,
      encoderHiddenStatesMask: encoderHiddenStatesMask,
      imageSegments: imageSegments,
      precomputedImageRotaryEmbeddings: precomputedImageRotaryEmbeddings
    )
  }

  public func forwardLatents(
    timestepIndex: Int,
    runtimeConfig: QwenRuntimeConfig,
    latentImages: MLXArray,
    encoderHiddenStates: MLXArray,
    encoderHiddenStatesMask: MLXArray
  ) -> MLXArray {
    let packed = LatentUtilities.packLatents(
      latentImages,
      height: runtimeConfig.height,
      width: runtimeConfig.width
    )
    let noiseTokens = forwardTokens(
      timestepIndex: timestepIndex,
      runtimeConfig: runtimeConfig,
      latentTokens: packed,
      encoderHiddenStates: encoderHiddenStates,
      encoderHiddenStatesMask: encoderHiddenStatesMask,
      imageSegments: nil
    )
    return LatentUtilities.unpackLatents(
      noiseTokens,
      height: runtimeConfig.height,
      width: runtimeConfig.width
    )
  }
}

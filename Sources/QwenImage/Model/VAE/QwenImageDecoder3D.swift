import Foundation
import MLX
import MLXNN

final class QwenImageDecoder3D: Module {
  @ModuleInfo(key: "conv_in") var convIn: QwenImageCausalConv3D
  @ModuleInfo(key: "mid_block") var midBlock: QwenImageMidBlock3D
  @ModuleInfo(key: "up_blocks") var upBlocks: [QwenImageUpBlock3D]
  @ModuleInfo(key: "norm_out") var normOut: QwenImageRMSNorm
  @ModuleInfo(key: "conv_out") var convOut: QwenImageCausalConv3D

  override init() {
    self._convIn.wrappedValue = QwenImageCausalConv3D(
      inputChannels: 16,
      outputChannels: 384,
      kernelSize: (3, 3, 3),
      stride: (1, 1, 1),
      padding: (1, 1, 1)
    )
    self._midBlock.wrappedValue = QwenImageMidBlock3D(channels: 384, attentionLayers: 1)
    self._upBlocks.wrappedValue = [
      QwenImageUpBlock3D(inChannels: 384, outChannels: 384, numberOfResBlocks: 2, upsampleMode: .upsample3d),
      QwenImageUpBlock3D(inChannels: 192, outChannels: 384, numberOfResBlocks: 2, upsampleMode: .upsample3d),
      QwenImageUpBlock3D(inChannels: 192, outChannels: 192, numberOfResBlocks: 2, upsampleMode: .upsample2d),
      QwenImageUpBlock3D(inChannels: 96, outChannels: 96, numberOfResBlocks: 2, upsampleMode: nil)
    ]
    self._normOut.wrappedValue = QwenImageRMSNorm(channels: 96, images: false)
    self._convOut.wrappedValue = QwenImageCausalConv3D(
      inputChannels: 96,
      outputChannels: 3,
      kernelSize: (3, 3, 3),
      stride: (1, 1, 1),
      padding: (1, 1, 1)
    )
    super.init()
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    var hidden = convIn(x)
    hidden = midBlock(hidden)
    for block in upBlocks {
      hidden = block(hidden)
    }
    hidden = normOut(hidden).asType(hidden.dtype)
    hidden = MLXNN.silu(hidden)
    hidden = convOut(hidden)
    return hidden
  }
}

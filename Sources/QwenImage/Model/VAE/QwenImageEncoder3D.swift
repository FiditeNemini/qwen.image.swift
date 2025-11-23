import Foundation
import MLX
import MLXNN

final class QwenImageEncoder3D: Module {
  private static let baseChannels = 96
  private static let stageChannelMultipliers = [1, 1, 2, 4, 4]
  private static let downsampleModes: [QwenImageResampleMode?] = [
    .downsample2d,
    .downsample3d,
    .downsample3d,
    nil
  ]

  @ModuleInfo(key: "conv_in") var convIn: QwenImageCausalConv3D
  @ModuleInfo(key: "down_blocks") var downBlocks: [QwenImageDownBlock3D]
  @ModuleInfo(key: "mid_block") var midBlock: QwenImageMidBlock3D
  @ModuleInfo(key: "norm_out") var normOut: QwenImageRMSNorm
  @ModuleInfo(key: "conv_out") var convOut: QwenImageCausalConv3D

  override init() {
    let channels = Self.stageChannelMultipliers.map { $0 * Self.baseChannels }
    self._convIn.wrappedValue = QwenImageCausalConv3D(
      inputChannels: 3,
      outputChannels: channels[0],
      kernelSize: (3, 3, 3),
      stride: (1, 1, 1),
      padding: (1, 1, 1)
    )

    var blocks: [QwenImageDownBlock3D] = []
    for (index, mode) in Self.downsampleModes.enumerated() {
      let inChannels = channels[index]
      let outChannels = channels[index + 1]
      blocks.append(
        QwenImageDownBlock3D(
          inChannels: inChannels,
          outChannels: outChannels,
          numberOfResBlocks: 2,
          downsampleMode: mode
        )
      )
    }
    self._downBlocks.wrappedValue = blocks

    self._midBlock.wrappedValue = QwenImageMidBlock3D(channels: channels.last ?? Self.baseChannels, attentionLayers: 1)
    self._normOut.wrappedValue = QwenImageRMSNorm(channels: channels.last ?? Self.baseChannels, images: false)
    self._convOut.wrappedValue = QwenImageCausalConv3D(
      inputChannels: channels.last ?? Self.baseChannels,
      outputChannels: 32,
      kernelSize: (3, 3, 3),
      stride: (1, 1, 1),
      padding: (1, 1, 1)
    )

    super.init()
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    var hidden = convIn(x)
    for (index, block) in downBlocks.enumerated() {
      hidden = block(hidden)
    }
    hidden = midBlock(hidden)
    hidden = normOut(hidden).asType(hidden.dtype)
    hidden = MLXNN.silu(hidden)
    hidden = convOut(hidden)
    return hidden
  }
}

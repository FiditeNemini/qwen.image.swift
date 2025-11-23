import Foundation
import MLX

enum LatentUtilities {
  static func packLatents(
    _ latents: MLXArray,
    height: Int,
    width: Int,
    latentChannels: Int = 16
  ) -> MLXArray {
    precondition(latents.ndim == 4, "Expected latents in BCHW format.")
    let batch = latents.dim(0)
    let channels = latents.dim(1)
    let latentHeight = latents.dim(2)
    let latentWidth = latents.dim(3)

    let patchHeight = height / 16
    let patchWidth = width / 16

    precondition(channels == latentChannels, "Latent channels (\(channels)) must match expected \(latentChannels).")
    precondition(latentHeight == patchHeight * 2, "Latent height mismatch: expected \(patchHeight * 2), got \(latentHeight).")
    precondition(latentWidth == patchWidth * 2, "Latent width mismatch: expected \(patchWidth * 2), got \(latentWidth).")

    var hidden = latents
    hidden = hidden.reshaped(batch, channels, patchHeight, 2, patchWidth, 2)
    hidden = hidden.transposed(0, 2, 4, 1, 3, 5)
    hidden = hidden.reshaped(batch, patchHeight * patchWidth, channels * 4)
    return hidden
  }

  static func unpackLatents(
    _ tokens: MLXArray,
    height: Int,
    width: Int,
    latentChannels: Int = 16
  ) -> MLXArray {
    precondition(tokens.ndim == 3, "Expected token tensor with shape [batch, tokens, features].")
    let batch = tokens.dim(0)
    let tokenCount = tokens.dim(1)
    let features = tokens.dim(2)

    let patchHeight = height / 16
    let patchWidth = width / 16

    precondition(features == latentChannels * 4, "Token feature size (\(features)) must match \(latentChannels * 4).")
    precondition(tokenCount == patchHeight * patchWidth, "Token count (\(tokenCount)) must match \(patchHeight * patchWidth).")

    var hidden = tokens
    hidden = hidden.reshaped(batch, patchHeight, patchWidth, latentChannels, 2, 2)
    hidden = hidden.transposed(0, 3, 1, 4, 2, 5)
    hidden = hidden.reshaped(batch, latentChannels, patchHeight * 2, patchWidth * 2)
    return hidden
  }
}

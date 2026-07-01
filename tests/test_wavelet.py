import torch

from wavestgate.models.wavelet_encoder import HaarWaveletTransform, WaveletMorphologyEncoder


def test_haar_round_trip_preserves_odd_shape():
    wavelet = HaarWaveletTransform()
    x = torch.randn(2, 3, 31, 32)
    bands = wavelet.dwt(x)
    reconstructed = wavelet.idwt(bands)
    assert reconstructed.shape == x.shape
    assert torch.allclose(reconstructed, x, atol=1e-5)


def test_morphology_encoder_handles_odd_patch():
    encoder = WaveletMorphologyEncoder(in_channels=3, latent_dim=16)
    token, fmap = encoder(torch.randn(2, 3, 33, 35), return_map=True)
    assert token.shape == (2, 16)
    assert fmap.shape[:2] == (2, 16)
    assert fmap.shape[-2:] == (33, 35)

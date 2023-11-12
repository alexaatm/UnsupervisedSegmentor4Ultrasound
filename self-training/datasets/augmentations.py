import numpy as np
import torch
from PIL import Image


class HistogramNormalize:
    """Performs histogram normalization on numpy array and returns 8-bit image.

    Code was taken from lightly, but adpated to work with PIL image as input:
    https://docs.lightly.ai/self-supervised-learning/tutorials/package/tutorial_custom_augmentations.html
    who adapted it from Facebook:
    https://github.com/facebookresearch/CovidPrognosis

    """

    def __init__(self, number_bins: int = 256):
        self.number_bins = number_bins

    def __call__(self, image: np.array) -> Image:
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        # Get the image histogram.
        image_histogram, bins = np.histogram(
            image.flatten(), self.number_bins, density=True
        )
        cdf = image_histogram.cumsum()  # cumulative distribution function
        cdf = 255 * cdf / cdf[-1]  # normalize

        # Use linear interpolation of cdf to find new pixel values.
        image_equalized = np.interp(image.flatten(), bins[:-1], cdf)
        pil_image = Image.fromarray(np.uint8(image_equalized.reshape(image.shape)))
        # return Image.fromarray(image_equalized.reshape(image.shape))
        return pil_image

class GaussianNoise:
    """Applies random Gaussian noise to a tensor.

    Code was taken from lightly tutirials:
    https://docs.lightly.ai/self-supervised-learning/tutorials/package/tutorial_custom_augmentations.html

    The intensity of the noise is dependent on the mean of the pixel values.
    See https://arxiv.org/pdf/2101.04909.pdf for more information.

    """

    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        mu = sample.mean()
        snr = np.random.randint(low=4, high=8)
        sigma = mu / snr
        noise = torch.normal(torch.zeros(sample.shape), sigma)
        return sample + noise
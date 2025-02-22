'''This module handles task-dependent operations (A) and noises (n) to simulate a measurement y=Ax+n.'''

from abc import ABC, abstractmethod
from functools import partial
import yaml
import skimage
from PIL import Image
import numpy as np
from torch.nn import functional as F
from torchvision import torch, transforms
from motionblur.motionblur import Kernel
import torchvision.transforms as transforms

from util.resizer import Resizer
from util.img_utils import Blurkernel, fft2_m, fft2, ifft2


# =================
# Operation classes
# =================

__OPERATOR__ = {}

def register_operator(name: str):
    def wrapper(cls):
        if __OPERATOR__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __OPERATOR__[name] = cls
        return cls
    return wrapper


def get_operator(name: str, **kwargs):
    if __OPERATOR__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __OPERATOR__[name](**kwargs)


class LinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        # calculate A * X
        pass

    @abstractmethod
    def transpose(self, data, **kwargs):
        # calculate A^T * X
        pass
    
    def ortho_project(self, data, **kwargs):
        # calculate (I - A^T * A)X
        return data - self.transpose(self.forward(data, **kwargs), **kwargs)

    def project(self, data, measurement, **kwargs):
        # calculate (I - A^T * A)Y - AX
        return self.ortho_project(measurement, **kwargs) - self.forward(data, **kwargs)


@register_operator(name='noise')
class DenoiseOperator(LinearOperator):
    def __init__(self, device):
        self.device = device
    
    def forward(self, data):
        return data

    def transpose(self, data):
        return data
    
    def ortho_project(self, data):
        return data

    def project(self, data):
        return data


@register_operator(name='super_resolution')
class SuperResolutionOperator(LinearOperator):
    def __init__(self, in_shape, scale_factor, device):
        self.device = device
        self.up_sample = partial(F.interpolate, scale_factor=scale_factor)
        self.down_sample = Resizer(in_shape, 1/scale_factor).to(device)

    def forward(self, data, **kwargs):
        return self.down_sample(data)

    def transpose(self, data, **kwargs):
        return self.up_sample(data)

    def project(self, data, measurement, **kwargs):
        return data - self.transpose(self.forward(data)) + self.transpose(measurement)

@register_operator(name='motion_blur')
class MotionBlurOperator(LinearOperator):
    def __init__(self, kernel_size, intensity, device):
        self.device = device
        self.kernel_size = kernel_size
        self.conv = Blurkernel(blur_type='motion',
                               kernel_size=kernel_size,
                               std=intensity,
                               device=device).to(device)  # should we keep this device term?

        self.kernel = Kernel(size=(kernel_size, kernel_size), intensity=intensity)
        kernel = torch.tensor(self.kernel.kernelMatrix, dtype=torch.float32)
        self.conv.update_weights(kernel)
    
    def forward(self, data, **kwargs):
        # A^T * A 
        return self.conv(data)

    def transpose(self, data, **kwargs):
        return data

    def get_kernel(self):
        kernel = self.kernel.kernelMatrix.type(torch.float32).to(self.device)
        return kernel.view(1, 1, self.kernel_size, self.kernel_size)


@register_operator(name='gaussian_blur')
class GaussialBlurOperator(LinearOperator):
    def __init__(self, kernel_size, intensity, device):
        self.device = device
        self.kernel_size = kernel_size
        self.conv = Blurkernel(blur_type='gaussian',
                               kernel_size=kernel_size,
                               std=intensity,
                               device=device).to(device)
        self.kernel = self.conv.get_kernel()
        self.conv.update_weights(self.kernel.type(torch.float32))

    def forward(self, data, **kwargs):
        return self.conv(data)

    def transpose(self, data, **kwargs):
        return data

    def get_kernel(self):
        return self.kernel.view(1, 1, self.kernel_size, self.kernel_size)

@register_operator(name='inpainting')
class InpaintingOperator(LinearOperator):
    '''This operator get pre-defined mask and return masked image.'''
    def __init__(self, device):
        self.device = device
    
    def forward(self, data, **kwargs):
        try:
            return data * kwargs.get('mask', None).to(self.device)
        except:
            raise ValueError("Require mask")
    
    def transpose(self, data, **kwargs):
        return data
    
    def ortho_project(self, data, **kwargs):
        return data - self.forward(data, **kwargs)
    
@register_operator(name='convolution')
class ConvolutionOperator(LinearOperator):
    def __init__(self, psf_path, psf_size, device):
        psf = np.array(Image.open(psf_path))
        psf_bg = np.mean(psf[0 : 15, 0 : 15])             #102
        h = psf - psf_bg

        h = skimage.transform.resize(h, (270, 480), mode='constant', anti_aliasing=True)
        h = h.transpose((2, 0, 1))
        h = torch.tensor(h, device=device)
        h = transforms.CenterCrop(256)(h) 

        self.channels, self.img_shape = h.shape[0], h.shape[1:]
        self.device = device
        self.padded_shape = [self.nextPow2(2*n - 1) for n in self.img_shape]

        self.starti = (self.padded_shape[0] - self.img_shape[0])//2
        self.endi = self.starti + self.img_shape[0]
        self.startj = self.padded_shape[1]//2 - self.img_shape[1]//2
        self.endj = self.startj + self.img_shape[1]

        hpad = torch.zeros([self.channels] + self.padded_shape, device=self.device)
        hpad[:, self.starti:self.endi, self.startj:self.endj] = h

        self.h = fft2(hpad)

    # Get nearest power of 2 that is larger than input (used for padding)
    def nextPow2(self, n):
        return int(2**torch.ceil(torch.log2(torch.tensor(n))))

    def crop(self, X):
        return X[:, :, self.starti:self.endi, self.startj:self.endj]

    def pad(self, v):
        vpad = torch.zeros([v.shape[0], self.channels] + self.padded_shape, device=self.device)
        vpad[:, :, self.starti:self.endi, self.startj:self.endj] = v
        return vpad

    def forward(self, data, **kwargs):
        temp = fft2(self.pad(data))
        return self.crop(ifft2(temp * self.h))

    def transpose(self, data, **kwargs):
        return data

@register_operator(name='convolution')
class ConvolutionOperator(LinearOperator):
    def __init__(self, psf_path, psf_size, device):
        psf = np.array(Image.open(psf_path))
        psf_bg = np.mean(psf[0 : 15, 0 : 15])             #102
        h = psf - psf_bg

        h = skimage.transform.resize(h, (270, 480), mode='constant', anti_aliasing=True)
        h = h.transpose((2, 0, 1))
        # h = h[:,:-58,62:-18,]
        h = torch.tensor(h, device=device)
        h = transforms.CenterCrop(256)(h) 

        self.channels, self.img_shape = h.shape[0], h.shape[1:]
        self.device = device
        self.padded_shape = [self.nextPow2(2*n - 1) for n in self.img_shape]

        self.starti = (self.padded_shape[0] - self.img_shape[0])//2
        self.endi = self.starti + self.img_shape[0]
        self.startj = self.padded_shape[1]//2 - self.img_shape[1]//2
        self.endj = self.startj + self.img_shape[1]

        hpad = torch.zeros([self.channels] + self.padded_shape, device=self.device)
        hpad[:, self.starti:self.endi, self.startj:self.endj] = h

        self.h = fft2(hpad)

    # Get nearest power of 2 that is larger than input (used for padding)
    def nextPow2(self, n):
        return int(2**torch.ceil(torch.log2(torch.tensor(n))))

    def crop(self, X):
        return X[:, :, self.starti:self.endi, self.startj:self.endj]

    def pad(self, v):
        vpad = torch.zeros([v.shape[0], self.channels] + self.padded_shape, device=self.device)
        vpad[:, :, self.starti:self.endi, self.startj:self.endj] = v
        return vpad

    def forward(self, data, **kwargs):
        temp = fft2(self.pad(data))
        return self.crop(ifft2(temp * self.h))

    def transpose(self, data, **kwargs):
        return data

class NonLinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        pass

    def project(self, data, measurement, **kwargs):
        return data + measurement - self.forward(data) 

@register_operator(name='phase_retrieval')
class PhaseRetrievalOperator(NonLinearOperator):
    def __init__(self, oversample, device):
        self.pad = int((oversample / 8.0) * 256)
        self.device = device
        
    def forward(self, data, **kwargs):
        padded = F.pad(data, (self.pad, self.pad, self.pad, self.pad))
        amplitude = fft2_m(padded).abs()
        return amplitude

@register_operator(name='nonlinear_blur')
class NonlinearBlurOperator(NonLinearOperator):
    def __init__(self, opt_yml_path, device):
        self.device = device
        self.blur_model = self.prepare_nonlinear_blur_model(opt_yml_path)     
         
    def prepare_nonlinear_blur_model(self, opt_yml_path):
        '''
        Nonlinear deblur requires external codes (bkse).
        '''
        from bkse.models.kernel_encoding.kernel_wizard import KernelWizard

        with open(opt_yml_path, "r") as f:
            opt = yaml.safe_load(f)["KernelWizard"]
            model_path = opt["pretrained"]
        blur_model = KernelWizard(opt)
        blur_model.eval()
        blur_model.load_state_dict(torch.load(model_path)) 
        blur_model = blur_model.to(self.device)
        return blur_model
    
    def forward(self, data, **kwargs):
        random_kernel = torch.randn(1, 512, 2, 2).to(self.device) * 1.2
        data = (data + 1.0) / 2.0  #[-1, 1] -> [0, 1]
        blurred = self.blur_model.adaptKernel(data, kernel=random_kernel)
        blurred = (blurred * 2.0 - 1.0).clamp(-1, 1) #[0, 1] -> [-1, 1]
        return blurred

# =============
# Noise classes
# =============


__NOISE__ = {}

def register_noise(name: str):
    def wrapper(cls):
        if __NOISE__.get(name, None):
            raise NameError(f"Name {name} is already defined!")
        __NOISE__[name] = cls
        return cls
    return wrapper

def get_noise(name: str, **kwargs):
    if __NOISE__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    noiser = __NOISE__[name](**kwargs)
    noiser.__name__ = name
    return noiser

class Noise(ABC):
    def __call__(self, data):
        return self.forward(data)
    
    @abstractmethod
    def forward(self, data):
        pass

@register_noise(name='clean')
class Clean(Noise):
    def forward(self, data):
        return data

@register_noise(name='gaussian')
class GaussianNoise(Noise):
    def __init__(self, sigma):
        self.sigma = sigma
    
    def forward(self, data):
        return data + torch.randn_like(data, device=data.device) * self.sigma


@register_noise(name='poisson')
class PoissonNoise(Noise):
    def __init__(self, rate):
        self.rate = rate

    def forward(self, data):
        '''
        Follow skimage.util.random_noise.
        '''

        # TODO: set one version of poisson
       
        # version 3 (stack-overflow)
        import numpy as np
        data = (data + 1.0) / 2.0
        data = data.clamp(0, 1)
        device = data.device
        data = data.detach().cpu()
        data = torch.from_numpy(np.random.poisson(data * 255.0 * self.rate) / 255.0 / self.rate)
        data = data * 2.0 - 1.0
        data = data.clamp(-1, 1)
        return data.to(device)

        # version 2 (skimage)
        # if data.min() < 0:
        #     low_clip = -1
        # else:
        #     low_clip = 0

    
        # # Determine unique values in iamge & calculate the next power of two
        # vals = torch.Tensor([len(torch.unique(data))])
        # vals = 2 ** torch.ceil(torch.log2(vals))
        # vals = vals.to(data.device)

        # if low_clip == -1:
        #     old_max = data.max()
        #     data = (data + 1.0) / (old_max + 1.0)

        # data = torch.poisson(data * vals) / float(vals)

        # if low_clip == -1:
        #     data = data * (old_max + 1.0) - 1.0
       
        # return data.clamp(low_clip, 1.0)

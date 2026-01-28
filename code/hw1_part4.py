"""
Part 4: Differentiable hybrid image optimization.

This file implements differentiable optimization of hybrid image parameters using
perceptual frequency weighting based on viewing distance. There are THREE different
optimization modes implemented, each with different tradeoffs:

================================================================================
MODE 1: Mannos-Sakrison CSF (Full Perceptual Model)
================================================================================
- Uses the psychophysically-validated Mannos-Sakrison Contrast Sensitivity Function
- Optimizes: blur_sigma (blur strength) and alpha (high-freq weight)
- Status: ✓ WORKS WELL - Gives good results with perceptually-motivated frequency weights
- Run with: simple_csf=False
- Best for: Understanding how human perception varies with viewing distance

================================================================================
MODE 2: Simplified Complementary Filters (Learns All Parameters)
================================================================================
- Uses complementary Gaussian filters: low-pass for far view, high-pass for near view
  * Far view (low-freq):  W_far  = exp(-f²/(2σ_p²))
  * Near view (high-freq): W_near = 1 - W_far
- Optimizes: blur_sigma, alpha, AND sigma_p (cutoff frequency)
- Status: ✗ CHEATS - Model finds trivial solution by making sigma_p very large
  * Result: Shows only one of the input images instead of a true hybrid
  * Why: Too many degrees of freedom; blur_sigma and sigma_p interact in a way
    that allows the model to "cheat" by effectively ignoring one image
- Run with: simple_csf=True, learn_sigma_p=True, freeze_blur_sigma=False
- Problem: Unstable optimization landscape with degenerate solutions

================================================================================
MODE 3: Simplified Complementary Filters (Frozen Blur, Learns Cutoff)
================================================================================
- Uses the same complementary Gaussian filters as Mode 2
- Optimizes: alpha and sigma_p ONLY
- Keeps blur_sigma FIXED at a reasonable value (default: 4.0)
- Status: ✓ WORKS WELL - Gives reasonable results similar to Mode 1
- Run with: simple_csf=True, learn_sigma_p=True, freeze_blur_sigma=True
- Why it works: Freezing blur_sigma removes the degeneracy, forcing sigma_p to
  find a meaningful cutoff frequency rather than a trivial solution
- Best for: Faster computation and simpler model with reasonable results

================================================================================
PARAMETER GLOSSARY
================================================================================
blur_sigma (σ_blur):
  - Controls the Gaussian blur applied to create low-frequency images
  - Larger values = more blur = lower spatial frequencies preserved
  - This is the "cutoff_frequency" from parts 1-2 of the assignment

sigma_p (σ_p):
  - Controls the cutoff frequency in the simplified perceptual masks (Modes 2-3)
  - Larger values = more low frequencies weighted, cutoff moves higher
  - Only used when simple_csf=True
  - In cycles per pixel (spatial frequency units)

alpha (α):
  - Weight/gain applied to the high-frequency component
  - Larger values = high frequencies more prominent in the hybrid
  - Controls the balance between low and high frequency images

D_near, D_far:
  - Viewing distances in meters for near/far views
  - Used to convert spatial frequencies to perceptual units (cycles per degree)
  - Only used in Mode 1 (Mannos-Sakrison CSF)

================================================================================
RECOMMENDED USAGE
================================================================================
For teaching/demos: Use Mode 1 (Mannos-Sakrison) for full perceptual model
For speed/simplicity: Use Mode 3 (frozen blur) for faster convergence
Avoid: Mode 2 demonstrates an interesting failure case in optimization
"""
import math
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from skimage import io, img_as_float32


# ============================================================================
# PERCEPTUAL FREQUENCY WEIGHTING FUNCTIONS
# ============================================================================


def csf_mannos_sakrison(f_cpd: torch.Tensor) -> torch.Tensor:
    """
    Mannos-Sakrison Contrast Sensitivity Function (CSF).
    
    This function models human visual system's sensitivity to different spatial
    frequencies. It's band-pass shaped: we're most sensitive to mid-frequencies,
    less to very low or very high frequencies.
    
    Args:
        f_cpd: Spatial frequency in cycles per degree (cpd)
    
    Returns:
        Sensitivity values (arbitrary units, band-pass shaped curve)
    
    Reference: Mannos & Sakrison, "The Effects of a Visual Fidelity Criterion
    on the Encoding of Images" (1974)
    """
    f = torch.clamp(f_cpd, min=1e-6)
    S = 2.6 * (0.0192 + 0.114 * f) * torch.exp(-(0.114 * f) ** 1.1)
    return S


def radial_freq_cycles_per_pixel(H, W, device):
    """
    Create a radial frequency grid in cycles per pixel.
    
    Returns a 2D array where each element contains the radial frequency
    (distance from DC component in frequency space).
    """
    fy = torch.fft.fftfreq(H, d=1.0, device=device)  # cycles/pixel
    fx = torch.fft.fftfreq(W, d=1.0, device=device)
    FY, FX = torch.meshgrid(fy, fx, indexing="ij")
    fr = torch.sqrt(FX**2 + FY**2)
    return fr


def cycles_per_degree(fr_cyc_per_px, D_m, pixel_pitch_m):
    """
    Convert spatial frequency from cycles/pixel to cycles/degree.
    
    Uses small-angle approximation to convert pixel-based frequencies to
    visual angle-based frequencies, which is how human perception works.
    
    Args:
        fr_cyc_per_px: Frequency in cycles per pixel
        D_m: Viewing distance in meters
        pixel_pitch_m: Physical size of one pixel in meters
    
    Returns:
        Frequency in cycles per degree of visual angle
    """
    # small-angle approx: cpd ≈ f_px * (pi*D)/(180*p)
    return fr_cyc_per_px * (math.pi * D_m) / (180.0 * pixel_pitch_m)


# ============================================================================
# DIFFERENTIABLE IMAGE FILTERING
# ============================================================================


def gaussian_kernel2d(ksize, sigma, device):
    """
    Create a 2D Gaussian kernel - DIFFERENTIABLE version.
    
    Args:
        ksize: Kernel size (should be odd)
        sigma: Standard deviation (scalar tensor, requires_grad=True for optimization)
        device: torch device
    
    Returns:
        2D Gaussian kernel of shape (ksize, ksize), normalized to sum to 1
    """
    # sigma is a scalar tensor (requires grad)
    r = ksize // 2
    x = torch.arange(-r, r + 1, device=device).float()
    g1 = torch.exp(-(x**2) / (2.0 * sigma**2))
    g1 = g1 / g1.sum()
    g2 = g1[:, None] * g1[None, :]
    return g2  # (k,k)


def gaussian_blur(image_bchw, sigma, ksize=31):
    """
    Apply Gaussian blur to an image - DIFFERENTIABLE version.
    
    This is similar to student.py's gen_hybrid_image, but implemented in PyTorch
    so gradients can flow through sigma for optimization.
    
    Args:
        image_bchw: Image tensor of shape (B, C, H, W), float in [0,1]
        sigma: Blur strength (standard deviation), scalar tensor > 0
        ksize: Kernel size (default 31)
    
    Returns:
        Blurred image of same shape as input
    """
    B, C, H, W = image_bchw.shape
    device = image_bchw.device
    k2 = gaussian_kernel2d(ksize, sigma, device).to(image_bchw.dtype)
    weight = k2[None, None, :, :].repeat(C, 1, 1, 1)  # (C,1,k,k) depthwise

    pad = ksize // 2
    x = F.pad(image_bchw, (pad, pad, pad, pad), mode="reflect")
    y = F.conv2d(x, weight, groups=C)
    return y


def hybrid(I1, I2, sigma, alpha, ksize=31):
    """
    Create a hybrid image - DIFFERENTIABLE version.
    
    Hybrid image = low_frequencies(I1) + alpha * high_frequencies(I2)
    
    This matches the logic from parts 1-2 of the assignment, but in PyTorch
    so we can optimize sigma and alpha with gradient descent.
    
    Args:
        I1: Low-frequency source image (B, C, H, W)
        I2: High-frequency source image (B, C, H, W)
        sigma: Blur strength (scalar tensor, optimizable)
        alpha: High-frequency weight (scalar tensor, optimizable)
        ksize: Blur kernel size
    
    Returns:
        Hybrid image (B, C, H, W)
    """
    low1 = gaussian_blur(I1, sigma, ksize)
    low2 = gaussian_blur(I2, sigma, ksize)
    high2 = I2 - low2
    H = low1 + alpha * high2
    return H


# ============================================================================
# PERCEPTUAL LOSS FUNCTIONS
# ============================================================================


def csf_weighted_spectral_loss(Himg, ref, W):
    """
    Compute frequency-weighted MSE between two images in Fourier domain.
    
    This loss function compares the magnitude spectra of two images, weighted
    by a perceptual frequency mask W. Higher weights mean those frequencies
    are more important for perception.
    
    Args:
        Himg: Hybrid image (B, C, H, W)
        ref: Reference image (B, C, H, W)
        W: Frequency weight mask (H, W) - higher = more perceptually important
    
    Returns:
        Scalar loss: mean of weighted squared differences in magnitude spectra
    """
    FH = torch.fft.fft2(Himg, norm="ortho")
    FR = torch.fft.fft2(ref, norm="ortho")

    MH = torch.abs(FH)
    MR = torch.abs(FR)

    Wb = W[None, None, :, :]
    return torch.mean((Wb * (MH - MR)) ** 2)


# ============================================================================
# MAIN OPTIMIZATION FUNCTION
# ============================================================================


def optimize_sigma_alpha(
    I1,
    I2,
    D_near,
    D_far,
    pixel_pitch_m=0.000264,
    steps=300,
    lr=0.05,
    ksize=51,
    device="cpu",
    simple_csf=False,
    learn_sigma_p=False,
    freeze_blur_sigma=False,
    blur_sigma_fixed=4.0,
):
    """
    Optimize hybrid image parameters using perceptual frequency weighting.
    
    This function finds optimal blur_sigma and alpha (and optionally sigma_p) to
    create a hybrid image that looks like I1 when viewed from far away and I2
    when viewed close up.
    
    ============================================================================
    THREE OPTIMIZATION MODES (see file header for details):
    ============================================================================
    
    MODE 1 - Mannos-Sakrison CSF (RECOMMENDED):
        simple_csf=False
        Optimizes: blur_sigma, alpha
        Status: ✓ Works well
    
    MODE 2 - Simple CSF, learns everything (DEMONSTRATES FAILURE):
        simple_csf=True, learn_sigma_p=True, freeze_blur_sigma=False
        Optimizes: blur_sigma, alpha, sigma_p
        Status: ✗ Cheats - finds degenerate solution
    
    MODE 3 - Simple CSF, frozen blur (RECOMMENDED ALTERNATIVE):
        simple_csf=True, learn_sigma_p=True, freeze_blur_sigma=True
        Optimizes: alpha, sigma_p (blur_sigma fixed)
        Status: ✓ Works well
    
    ============================================================================
    
    Args:
        I1: Low-frequency source image, shape (B, C, H, W), float32 in [0,1]
            This image should be visible from FAR away
        I2: High-frequency source image, shape (B, C, H, W), float32 in [0,1]
            This image should be visible from NEAR
        D_near: Near viewing distance in meters (e.g., 0.25m = 25cm)
        D_far: Far viewing distance in meters (e.g., 5.0m)
        pixel_pitch_m: Physical size of one pixel in meters (default: 0.264mm)
        steps: Number of optimization steps (default: 300)
        lr: Learning rate for Adam optimizer (default: 0.05)
        ksize: Gaussian kernel size (default: 51, should be odd)
        device: 'cpu' or 'cuda'
        
        MODE CONTROL FLAGS:
        simple_csf: If True, use simplified complementary filters instead of
                    Mannos-Sakrison CSF. If False, use full CSF model.
        learn_sigma_p: If True AND simple_csf=True, optimize sigma_p (cutoff freq).
                       Otherwise sigma_p is fixed at 0.10 cycles/pixel.
        freeze_blur_sigma: If True, keep blur_sigma fixed at blur_sigma_fixed.
                           If False, optimize blur_sigma.
        blur_sigma_fixed: Fixed value for blur_sigma when freeze_blur_sigma=True
                          (default: 4.0)
    
    Returns:
        If learn_sigma_p=True:
            (Himg, sigma, alpha, W_near, W_far, sigma_p)
        Else:
            (Himg, sigma, alpha, W_near, W_far)
        
        Where:
            Himg: Optimized hybrid image (B, C, H, W)
            sigma: Final blur_sigma value (scalar tensor)
            alpha: Final alpha value (scalar tensor)
            W_near: Frequency weights for near view (H, W)
            W_far: Frequency weights for far view (H, W)
            sigma_p: Final sigma_p value if learned (scalar tensor)
    
    Loss Function:
        L = ||W_near ⊙ (FFT(H) - FFT(I2))||² + ||W_far ⊙ (FFT(H) - FFT(I1))||²
        
        Where:
        - H = hybrid(I1, I2, sigma, alpha) is the hybrid image
        - W_near weights frequencies important for near viewing
        - W_far weights frequencies important for far viewing
        - ⊙ denotes element-wise multiplication
        - FFT magnitudes are compared (phase is ignored)
    """
    I1 = I1.to(device)
    I2 = I2.to(device)
    B, C, H, W = I1.shape

    # Create radial frequency grid (in cycles/pixel)
    fr = radial_freq_cycles_per_pixel(H, W, device)  # (H,W)

    # ========================================================================
    # MODE 1: Precompute Mannos-Sakrison CSF weights (if using full CSF)
    # ========================================================================
    if not simple_csf:
        # Convert spatial frequencies to perceptual units (cycles/degree)
        f_near = cycles_per_degree(fr, D_near, pixel_pitch_m)
        f_far = cycles_per_degree(fr, D_far, pixel_pitch_m)
        
        # Apply psychophysical CSF curve
        W_near_fixed = csf_mannos_sakrison(f_near)
        W_far_fixed = csf_mannos_sakrison(f_far)
        
        # Normalize to [0, 1] range
        W_near_fixed = W_near_fixed / (W_near_fixed.max() + 1e-8)
        W_far_fixed = W_far_fixed / (W_far_fixed.max() + 1e-8)

    # ========================================================================
    # Initialize optimization parameters
    # ========================================================================
    
    # Alpha: weight for high-frequency component (always optimized)
    alpha = torch.tensor([1.0], device=device, requires_grad=True)

    # Blur sigma: controls Gaussian blur strength
    if freeze_blur_sigma:
        # MODE 3: Keep blur_sigma fixed (no gradient)
        sigma_fixed = torch.tensor([blur_sigma_fixed], device=device)
    else:
        # MODE 1 & 2: Optimize blur_sigma
        # Use log-space to ensure sigma stays positive: sigma = softplus(log_sigma)
        log_sigma = torch.tensor([math.log(2.0)], device=device, requires_grad=True)

    # Collect parameters to optimize
    params = [alpha]

    if not freeze_blur_sigma:
        params.append(log_sigma)

    # Sigma_p: cutoff frequency for simplified perceptual masks (MODE 2 & 3)
    if simple_csf and learn_sigma_p:
        log_sigma_p = torch.tensor([math.log(0.10)], device=device, requires_grad=True)
        params.append(log_sigma_p)
    else:
        log_sigma_p = None

    opt = torch.optim.Adam(params, lr=lr)

    # ========================================================================
    # Optimization loop
    # ========================================================================
    for t in range(steps):
        # Get current blur_sigma value
        if freeze_blur_sigma:
            sigma = sigma_fixed  # MODE 3: constant, no grad
        else:
            sigma = F.softplus(log_sigma) + 1e-6  # MODE 1 & 2: optimizable

        # Generate hybrid image with current parameters
        Himg = hybrid(I1, I2, sigma, alpha, ksize)

        # Build perceptual frequency weights
        if not simple_csf:
            # ----------------------------------------------------------------
            # MODE 1: Use precomputed Mannos-Sakrison CSF weights
            # ----------------------------------------------------------------
            W_near = W_near_fixed
            W_far = W_far_fixed

        else:
            # ----------------------------------------------------------------
            # MODE 2 & 3: Use simplified complementary Gaussian filters
            # ----------------------------------------------------------------
            # Get sigma_p (cutoff frequency for perceptual masks)
            if learn_sigma_p:
                sigma_p = F.softplus(log_sigma_p) + 1e-6  # MODE 2 & 3: optimizable
            else:
                sigma_p = torch.tensor([0.10], device=device, dtype=fr.dtype)  # fixed

            # Create complementary low-pass and high-pass masks
            W_low = torch.exp(-(fr**2) / (2.0 * sigma_p**2))  # Gaussian low-pass
            W_low = W_low / (W_low.max() + 1e-8)  # normalize

            W_far = W_low  # Far view sees low frequencies
            W_near = 1.0 - W_low  # Near view sees high frequencies (complementary)

        # Compute loss: match I2 at near distance, I1 at far distance
        loss = csf_weighted_spectral_loss(Himg, I2, W_near) + csf_weighted_spectral_loss(Himg, I1, W_far)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if (t % 25) == 0 or t == steps - 1:
            if simple_csf and learn_sigma_p:
                sigma_p_val = (F.softplus(log_sigma_p) + 1e-6).item()
                print(
                    f"step {t:03d} | loss {loss.item():.6f} | blur_sigma {sigma.item():.3f} | "
                    f"alpha {alpha.item():.3f} | sigma_p {sigma_p_val:.4f}"
                )
            else:
                print(f"step {t:03d} | loss {loss.item():.6f} | blur_sigma {sigma.item():.3f} | alpha {alpha.item():.3f}")

    if freeze_blur_sigma:
        sigma = sigma_fixed.detach()
    else:
        sigma = (F.softplus(log_sigma) + 1e-6).detach()

    alpha = alpha.detach()

    if not simple_csf:
        W_near = W_near_fixed.detach()
        W_far = W_far_fixed.detach()
    else:
        if learn_sigma_p:
            sigma_p = (F.softplus(log_sigma_p) + 1e-6).detach()
        else:
            sigma_p = torch.tensor([0.10], device=device, dtype=fr.dtype)
        W_low = torch.exp(-(fr**2) / (2.0 * sigma_p**2))
        W_low = W_low / (W_low.max() + 1e-8)
        W_far = W_low.detach()
        W_near = (1.0 - W_low).detach()

    Himg = hybrid(I1, I2, sigma, alpha, ksize).detach()

    if simple_csf and learn_sigma_p:
        return Himg, sigma, alpha, W_near, W_far, sigma_p
    else:
        return Himg, sigma, alpha, W_near, W_far


# ============================================================================
# VISUALIZATION AND UTILITY FUNCTIONS
# ============================================================================


def simulate_view(image_bchw, W):
    """
    Simulate viewing an image with frequency-dependent sensitivity.
    
    Applies a frequency-domain filter W to simulate what the image looks like
    when viewed under conditions where certain frequencies are more/less visible.
    
    Args:
        image_bchw: Image tensor (B, C, H, W)
        W: Frequency weight mask (H, W)
    
    Returns:
        Filtered image in spatial domain (B, C, H, W)
    """
    Fimg = torch.fft.fft2(image_bchw, norm="ortho")
    Wb = W[None, None, :, :]
    out = torch.real(torch.fft.ifft2(Wb * Fimg, norm="ortho"))
    return out


def center_crop_to_match(I1, I2):
    """
    Center-crop both images to the same size (minimum of both dimensions).
    
    This ensures I1 and I2 have the same spatial dimensions, which is required
    for creating hybrid images.
    """
    H = min(I1.shape[-2], I2.shape[-2])
    W = min(I1.shape[-1], I2.shape[-1])

    def crop(x):
        h0 = (x.shape[-2] - H) // 2
        w0 = (x.shape[-1] - W) // 2
        return x[..., h0 : h0 + H, w0 : w0 + W]

    return crop(I1), crop(I2)


def load_grayscale_image(path, device="cpu"):
    """
    Load an image as grayscale and convert to PyTorch tensor.
    
    Returns:
        Image tensor of shape (1, 1, H, W), float32 in [0, 1]
    """
    img = img_as_float32(io.imread(path, as_gray=True))
    img = torch.from_numpy(img)[None, None, :, :]  # (1,1,H,W)
    return img.to(device)


# ============================================================================
# DEMO FUNCTIONS
# ============================================================================


def run_mannos_sakrison_demo():
    """
    MODE 1 DEMO: Mannos-Sakrison CSF (Full Perceptual Model)
    
    This mode uses the psychophysically-validated CSF curve and optimizes
    blur_sigma and alpha. It works well and gives perceptually-motivated results.
    """
    print("=" * 80)
    print("MODE 1: Mannos-Sakrison CSF Demo")
    print("Optimizing: blur_sigma, alpha")
    print("Status: ✓ WORKS WELL")
    print("=" * 80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    I1 = load_grayscale_image("../data/cat.bmp", device)
    I2 = load_grayscale_image("../data/bird.bmp", device)
    I1, I2 = center_crop_to_match(I1, I2)

    D_near = 0.25  # 25cm viewing distance (close up)
    D_far = 5.0    # 5m viewing distance (far away)

    H, sigma, alpha, W_near, W_far = optimize_sigma_alpha(
        I1,
        I2,
        D_near,
        D_far,
        steps=300,
        lr=0.05,
        device=device,
        simple_csf=False,  # Use Mannos-Sakrison CSF
        learn_sigma_p=False,
        freeze_blur_sigma=False,  # Optimize blur_sigma
    )
    print(f"\nFinal blur_sigma = {sigma.item():.3f}, alpha = {alpha.item():.3f}")

    # Simulate near and far views
    H_near = simulate_view(H, W_near)
    H_far = simulate_view(H, W_far)

    # Visualize
    _plot_hybrid_results(I1, I2, H, H_near, H_far, "MODE 1: Mannos-Sakrison CSF")


def run_simple_csf_cheating_demo():
    """
    MODE 2 DEMO: Simplified CSF with all parameters learned (DEMONSTRATES FAILURE)
    
    This mode optimizes blur_sigma, alpha, AND sigma_p together. The model "cheats"
    by finding a degenerate solution where sigma_p becomes very large, effectively
    showing only one of the input images instead of a true hybrid.
    
    WARNING: This is included to demonstrate an interesting failure mode in
    optimization. Not recommended for actual use!
    """
    print("\n" + "=" * 80)
    print("MODE 2: Simplified CSF (ALL parameters learned) - CHEATING DEMO")
    print("Optimizing: blur_sigma, alpha, sigma_p")
    print("Status: ✗ CHEATS - finds degenerate solution")
    print("=" * 80)
    print("Notice how sigma_p grows very large and blur_sigma shrinks,")
    print("allowing the model to effectively show only one image.\n")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    I1 = load_grayscale_image("../data/cat.bmp", device)
    I2 = load_grayscale_image("../data/bird.bmp", device)
    I1, I2 = center_crop_to_match(I1, I2)

    D_near = 0.25
    D_far = 5.0

    H, sigma, alpha, W_near, W_far, sigma_p = optimize_sigma_alpha(
        I1,
        I2,
        D_near,
        D_far,
        steps=300,
        lr=0.05,
        device=device,
        simple_csf=True,          # Use simplified filters
        learn_sigma_p=True,       # Optimize sigma_p
        freeze_blur_sigma=False,  # Also optimize blur_sigma - THIS CAUSES PROBLEMS!
    )
    print(f"\nFinal blur_sigma = {sigma.item():.3f}, alpha = {alpha.item():.3f}, sigma_p = {sigma_p.item():.4f}")
    print("⚠️  Notice the degenerate parameter values - this is the 'cheating' behavior!")

    H_near = simulate_view(H, W_near)
    H_far = simulate_view(H, W_far)

    _plot_hybrid_results(I1, I2, H, H_near, H_far, "MODE 2: Simplified CSF (Cheats)")


def run_simple_perceptual_demo():
    """
    MODE 3 DEMO: Simplified CSF with frozen blur_sigma (RECOMMENDED ALTERNATIVE)
    
    This mode fixes blur_sigma at a reasonable value and only optimizes alpha and
    sigma_p. By removing the degeneracy between blur_sigma and sigma_p, this gives
    reasonable results similar to the Mannos-Sakrison CSF, but with simpler computation.
    """
    print("\n" + "=" * 80)
    print("MODE 3: Simplified CSF with Frozen Blur Demo")
    print("Optimizing: alpha, sigma_p (blur_sigma = 4.0 fixed)")
    print("Status: ✓ WORKS WELL")
    print("=" * 80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    I1 = load_grayscale_image("../data/cat.bmp", device)
    I2 = load_grayscale_image("../data/bird.bmp", device)
    I1, I2 = center_crop_to_match(I1, I2)

    D_near = 0.25
    D_far = 5.0

    H, sigma, alpha, W_near, W_far, sigma_p = optimize_sigma_alpha(
        I1,
        I2,
        D_near,
        D_far,
        steps=300,
        lr=0.05,
        device=device,
        simple_csf=True,         # Use simplified filters
        learn_sigma_p=True,      # Optimize sigma_p
        freeze_blur_sigma=True,  # BUT keep blur_sigma fixed - this prevents cheating!
        blur_sigma_fixed=4.0,
    )
    print(f"\nFinal blur_sigma = {sigma.item():.3f} (fixed), alpha = {alpha.item():.3f}, sigma_p = {sigma_p.item():.4f}")
    print("✓ Notice reasonable parameter values - no cheating!")

    H_near = simulate_view(H, W_near)
    H_far = simulate_view(H, W_far)

    _plot_hybrid_results(I1, I2, H, H_near, H_far, "MODE 3: Simplified CSF (Frozen Blur)")


def _plot_hybrid_results(I1, I2, H, H_near, H_far, title):
    """
    Helper function to plot hybrid image results.
    """
    def show(img, subtitle):
        plt.imshow(img.squeeze().cpu().numpy(), cmap="gray")
        plt.title(subtitle)
        plt.axis("off")

    plt.figure(figsize=(12, 8))
    plt.suptitle(title, fontsize=14, fontweight='bold')

    plt.subplot(2, 3, 1)
    show(I1, "I1 (low-freq source)")
    plt.subplot(2, 3, 2)
    show(I2, "I2 (high-freq source)")
    plt.subplot(2, 3, 3)
    show(H, "Hybrid Image")

    plt.subplot(2, 3, 5)
    show(H_far, "Simulated Far View\n(should look like I1)")
    plt.subplot(2, 3, 6)
    show(H_near, "Simulated Near View\n(should look like I2)")

    plt.tight_layout()
    plt.show()


def run_all_demos():
    """
    Run all three demo modes in sequence to compare results.
    
    MODE 1: Mannos-Sakrison CSF (works well)
    MODE 2: Simplified CSF, all params (cheats)
    MODE 3: Simplified CSF, frozen blur (works well)
    """
    print("\n" + "=" * 80)
    print("RUNNING ALL THREE OPTIMIZATION MODES FOR COMPARISON")
    print("=" * 80)
    print("\nYou will see three optimization runs:")
    print("  1. MODE 1 - Mannos-Sakrison CSF (recommended)")
    print("  2. MODE 2 - Simple CSF learning everything (demonstrates failure)")
    print("  3. MODE 3 - Simple CSF with frozen blur (recommended alternative)")
    print("\nClose each plot window to proceed to the next mode.\n")
    
    run_mannos_sakrison_demo()
    run_simple_csf_cheating_demo()
    run_simple_perceptual_demo()
    
    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80)
    print("\nKey Observations:")
    print("  - MODE 1 (Mannos-Sakrison): Uses psychophysical model, works well")
    print("  - MODE 2 (Simple, all params): Finds degenerate solution, doesn't work")
    print("  - MODE 3 (Simple, frozen blur): Constrains optimization, works well")
    print("\nConclusion: When optimizing, be careful about parameter interactions!")
    print("Freezing blur_sigma in MODE 3 prevents the model from 'cheating'.\n")


if __name__ == "__main__":
    # By default, run MODE 3 (simple perceptual demo with frozen blur)
    # This is the recommended starting point for students
    
    # To run a specific mode, uncomment one of these:
    # run_mannos_sakrison_demo()          # MODE 1: Full CSF model
    # run_simple_csf_cheating_demo()      # MODE 2: Demonstrates failure case
    run_simple_perceptual_demo()          # MODE 3: Simplified with frozen blur
    
    # To compare all three modes:
    # run_all_demos()

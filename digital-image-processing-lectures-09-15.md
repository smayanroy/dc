# Comprehensive Guide to Digital Image Processing: Lectures 09 to 15

Digital image processing in the spatial and color domains relies on mathematical operations that transform raw pixel arrays into structured, feature-rich representations. This guide presents a rigorous, lecture-by-lecture breakdown of Lectures 09 through 15. Each section details the underlying signal processing theory, mathematical derivations, filter kernel matrices, and runnable side-by-side implementations using OpenCV (`cv2`), Scikit-Image (`skimage`), and NumPy.

---

## 1. Overview & Spatial Domain Fundamentals

Spatial domain image processing operates directly on the intensity values of image pixels. An image is mathematically defined as a two-dimensional discrete function $f(x, y)$, where $x$ and $y$ denote spatial coordinates (row and column indices), and $f(x, y)$ represents the intensity or color value at that coordinate.

Spatial filtering modifies $f(x, y)$ using a local neighborhood operation defined by a kernel (or mask) $w(s, t)$ of size $m \times n$:

$$g(x, y) = (f * w)(x, y) = \sum_{s=-a}^{a} \sum_{t=-b}^{b} w(s, t) f(x - s, y - t)$$

where $a = \frac{m-1}{2}$ and $b = \frac{n-1}{2}$. Linear spatial filtering relies on 2D convolution or cross-correlation, forming the backbone for smoothing, sharpening, derivative analysis, and feature extraction.

---

## 2. Lecture 09: Image Filtering (Laplacian & Sharpening)

### 2.1 Theoretical Foundations & Derivative Behavior
Image sharpening highlights fine details, edges, and intensity discontinuities while suppressing low-frequency, slowly varying background components. While low-pass filters (like Gaussian or box blurs) compute weighted averages, sharpening filters compute spatial derivatives.

To understand derivative behavior across intensity transitions:
- **Step Edge**: A sudden jump in pixel values. The first derivative produces a thick pulse; the second derivative produces a double-spike (positive peak on the dark side, negative peak on the bright side) with a zero-crossing at the midpoint.
- **Ramp Edge**: A gradual transition. The first derivative yields a constant non-zero plateau over the ramp; the second derivative produces two spikes at the start and end of the ramp.
- **Isolated Noise / Line**: A single pixel spike. The first derivative produces two equal and opposite spikes; the second derivative yields a strong positive spike surrounded by negative spikes.

```
Intensity Profile f(x):   [ 0  0  0  1  2  3  4  5  5  5 ]  (Ramp)
1st Derivative f'(x):     [ 0  0  0  1  1  1  1  1  0  0 ]  (Constant Plateau)
2nd Derivative f''(x):    [ 0  0  0  1  0  0  0 -1  0  0 ]  (Zero-Crossing Pair)
```

### 2.2 Mathematical Formulation of 1D and 2D Derivatives
In discrete spatial domain processing, derivatives are approximated via finite differences.

#### 1D Discrete Derivatives
- **Forward Difference**:
  $$\Delta f(x) = f(x + 1) - f(x)$$
- **Backward Difference**:
  $$\nabla f(x) = f(x) - f(x - 1)$$
- **Central Difference**:
  $$\frac{\partial f}{\partial x} \approx \frac{f(x + 1) - f(x - 1)}{2}$$
- **Second Derivative (1D Central Difference)**:
  $$\frac{\partial^2 f}{\partial x^2} = \frac{\partial}{\partial x}\left( \frac{\partial f}{\partial x} \right) \approx f(x + 1) - 2f(x) + f(x - 1)$$

#### 2D Second Derivative: The Laplacian Operator
The Laplacian is an isotropic (rotation-invariant) linear operator defined as the sum of second-order spatial partial derivatives:

$$\nabla^2 f(x, y) = \frac{\partial^2 f}{\partial x^2} + \frac{\partial^2 f}{\partial y^2}$$

Expanding using discrete 1D central second differences:

$$\frac{\partial^2 f}{\partial x^2} \approx f(x + 1, y) - 2f(x, y) + f(x - 1, y)$$

$$\frac{\partial^2 f}{\partial y^2} \approx f(x, y + 1) - 2f(x, y) + f(x, y - 1)$$

Combining both partials yields the standard 4-neighbor discrete Laplacian:

$$\nabla^2 f(x, y) = f(x + 1, y) + f(x - 1, y) + f(x, y + 1) + f(x, y - 1) - 4f(x, y)$$

Including diagonal terms yields the 8-neighbor isotropic extension:

$$\nabla^2_{8} f(x, y) = \sum_{s=-1}^{1} \sum_{t=-1}^{1} f(x+s, y+t) - 9f(x, y)$$

### 2.3 Kernel Matrices
Standard discrete Laplacian kernels:

$$\text{Laplacian (4-neighbor)} = \begin{bmatrix} 0 & 1 & 0 \\ 1 & -4 & 1 \\ 0 & 1 & 0 \end{bmatrix}, \quad \text{Laplacian (8-neighbor)} = \begin{bmatrix} 1 & 1 & 1 \\ 1 & -8 & 1 \\ 1 & 1 & 1 \end{bmatrix}$$

Inverted sign convention matrices (where the center is positive):

$$\text{Laplacian (Positive Center)} = \begin{bmatrix} 0 & -1 & 0 \\ -1 & 4 & -1 \\ 0 & -1 & 0 \end{bmatrix}, \quad \begin{bmatrix} -1 & -1 & -1 \\ -1 & 8 & -1 \\ -1 & -1 & -1 \end{bmatrix}$$

### 2.4 Laplacian Sharpening & Unsharp Masking Formulas

#### Laplacian Image Sharpening
To restore blurred details, the scaled Laplacian response is subtracted from (or added to) the original image depending on the sign convention of the center element:

$$g(x, y) = \begin{cases} f(x, y) - \nabla^2 f(x, y) & \text{if center coefficient is negative (e.g., } -4, -8) \\ f(x, y) + \nabla^2 f(x, y) & \text{if center coefficient is positive (e.g., } +4, +8) \end{cases}$$

#### Unsharp Masking & High-Boost Filtering
Unsharp masking sharpens an image by subtracting a smoothed (blurred) version from the original:

1. **Obtain Blurred Image**: $\bar{f}(x, y) = \text{Smooth}\{f(x, y)\}$
2. **Form Unsharp Mask**: $g_{\text{mask}}(x, y) = f(x, y) - \bar{f}(x, y)$
3. **Add Weighted Mask Back**: 
   $$g(x, y) = f(x, y) + k \cdot g_{\text{mask}}(x, y)$$

where:
- $k = 1$: Standard Unsharp Masking
- $k > 1$: High-Boost Filtering (amplifies high frequencies over low frequencies)

### 2.5 Side-by-Side Code Implementation

```python
import cv2
import numpy as np
from skimage import filters, feature

def apply_laplacian_and_unsharp(image_path):
    # Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_float = img.astype(np.float64)

    # --- OPENCV IMPLEMENTATION ---
    # 1. Discrete Laplacian Filtering (8-neighbor kernel via filter2D)
    kernel_laplacian = np.array([[1,  1, 1],
                                 [1, -8, 1],
                                 [1,  1, 1]], dtype=np.float64)
    laplacian_response = cv2.filter2D(img_float, -1, kernel_laplacian)
    
    # Sharpen image: Subtracting negative-center Laplacian
    img_sharpened_laplacian = np.clip(img_float - laplacian_response, 0, 255).astype(np.uint8)

    # 2. Unsharp Masking & High-Boost Filtering
    blurred = cv2.GaussianBlur(img_float, (5, 5), sigmaX=1.0)
    unsharp_mask = img_float - blurred
    
    k_unsharp = 1.0
    k_highboost = 2.5
    
    img_unsharp = np.clip(img_float + k_unsharp * unsharp_mask, 0, 255).astype(np.uint8)
    img_highboost = np.clip(img_float + k_highboost * unsharp_mask, 0, 255).astype(np.uint8)

    # --- SCIKIT-IMAGE IMPLEMENTATION ---
    # Laplacian using skimage filters
    sk_laplacian = filters.laplace(img / 255.0)
    sk_sharpened = np.clip((img / 255.0) - sk_laplacian, 0.0, 1.0) * 255.0

    return img_sharpened_laplacian, img_unsharp, img_highboost, sk_sharpened.astype(np.uint8)
```

---

## 3. Lecture 10: Edge Detection & Gradient Operators

### 3.1 Edge Model & First Derivative Properties
Edges are significant local intensity changes in an image. They typically signify object boundaries, surface orientation changes, or material variations.

The gradient of a 2D scalar function $f(x, y)$ is a vector pointing in the direction of the maximum rate of change of $f$ at coordinates $(x, y)$:

$$\nabla f(x, y) = \begin{bmatrix} g_x \\ g_y \end{bmatrix} = \begin{bmatrix} \frac{\partial f}{\partial x} \\ \frac{\partial f}{\partial y} \end{bmatrix}$$

- **Gradient Magnitude**:
  $$M(x, y) = \|\nabla f\| = \sqrt{g_x^2 + g_y^2} \approx |g_x| + |g_y|$$
- **Gradient Direction (Orientation)**:
  $$\alpha(x, y) = \arctan\left( \frac{g_y}{g_x} \right)$$
  The edge direction is perpendicular to the gradient direction: $\theta_{\text{edge}} = \alpha(x, y) \pm 90^\circ$.

### 3.2 Gradient Operator Kernels

#### Roberts Cross Gradient Operators ($2 \times 2$)
Approximates diagonal derivative differences:

$$G_x = \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix}, \quad G_y = \begin{bmatrix} 0 & 1 \\ -1 & 0 \end{bmatrix}$$

$$g_x = f(x, y) - f(x+1, y+1), \quad g_y = f(x+1, y) - f(x, y+1)$$

#### Prewitt Operators ($3 \times 3$)
Combines central differences along one axis with uniform smoothing along the orthogonal axis:

$$G_x = \begin{bmatrix} -1 & 0 & 1 \\ -1 & 0 & 1 \\ -1 & 0 & 1 \end{bmatrix}, \quad G_y = \begin{bmatrix} -1 & -1 & -1 \\ 0 & 0 & 0 \\ 1 & 1 & 1 \end{bmatrix}$$

#### Sobel Operators ($3 \times 3$)
Replaces uniform smoothing with Gaussian-weighted smoothing ($1, 2, 1$) to reduce noise sensitivity:

$$G_x = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}, \quad G_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{bmatrix}$$

#### Scharr Operators ($3 \times 3$)
Provides higher rotational symmetry and accurate gradient magnitude estimations for fine edges:

$$G_x = \begin{bmatrix} -3 & 0 & 3 \\ -10 & 0 & 10 \\ -3 & 0 & 3 \end{bmatrix}, \quad G_y = \begin{bmatrix} -3 & -10 & -3 \\ 0 & 0 & 0 \\ 3 & 10 & 3 \end{bmatrix}$$

#### Kirsch & Robinson Compass Masks ($3 \times 3$)
Measures edge responses in eight directional orientations (N, NW, W, SW, S, SE, E, NE). For Kirsch:

$$K_{\text{North}} = \begin{bmatrix} 5 & 5 & 5 \\ -3 & 0 & -3 \\ -3 & -3 & -3 \end{bmatrix}, \quad K_{\text{West}} = \begin{bmatrix} 5 & -3 & -3 \\ 5 & 0 & -3 \\ 5 & -3 & -3 \end{bmatrix}$$

### 3.3 Side-by-Side Code Implementation

```python
import cv2
import numpy as np
from skimage import filters

def apply_gradient_operators(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # --- OPENCV IMPLEMENTATION ---
    # 1. Sobel Gradient Computation
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    
    # Magnitude and Direction
    magnitude_sobel = cv2.magnitude(sobel_x, sobel_y)
    angle_sobel = cv2.phase(sobel_x, sobel_y, angleInDegrees=True)

    # 2. Scharr Operator
    scharr_x = cv2.Scharr(img, cv2.CV_64F, 1, 0)
    scharr_y = cv2.Scharr(img, cv2.CV_64F, 0, 1)
    magnitude_scharr = cv2.magnitude(scharr_x, scharr_y)

    # 3. Prewitt Operator via Custom Kernels
    kernel_prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float64)
    kernel_prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float64)
    prewitt_x = cv2.filter2D(img.astype(np.float64), -1, kernel_prewitt_x)
    prewitt_y = cv2.filter2D(img.astype(np.float64), -1, kernel_prewitt_y)
    magnitude_prewitt = np.hypot(prewitt_x, prewitt_y)

    # --- SCIKIT-IMAGE IMPLEMENTATION ---
    sk_sobel_mag = filters.sobel(img)
    sk_prewitt_mag = filters.prewitt(img)
    sk_roberts_mag = filters.roberts(img)
    sk_scharr_mag = filters.scharr(img)

    return magnitude_sobel, magnitude_prewitt, magnitude_scharr, sk_sobel_mag
```

---

## 4. Lecture 11: Multi-Stage Edge Detection (Canny Pipeline)

### 4.1 Canny Optimization Criteria
John Canny formulated edge detection as an optimal trade-off problem based on three evaluation criteria:
1. **Low Error Rate**: High probability of detecting real edge pixels with low false positive/negative rates.
2. **Localization Precision**: Detected edge pixels must be positioned as close as possible to the true edge center.
3. **Single Response Constraint**: Only one edge response per true edge (elimination of multiple phantom responses across thick edges).

### 4.2 The 4-Stage Canny Edge Detector Pipeline

```
[ Input Image ] ---> (1) Gaussian Filter ---> (2) Sobel Gradient (Mag & Angle)
                                                         |
[ Output Edges ] <--- (4) Hysteresis <--- (3) Non-Maximum Suppression
```

#### Stage 1: Noise Reduction via Gaussian Filtering
Convolve input image $I(x, y)$ with a 2D Gaussian kernel $G_\sigma(x, y)$:

$$G_\sigma(x, y) = \frac{1}{2\pi\sigma^2} \exp\left( -\frac{x^2 + y^2}{2\sigma^2} \right)$$

$$I_{\text{smooth}}(x, y) = I(x, y) * G_\sigma(x, y)$$

#### Stage 2: Gradient Magnitude and Orientation Computation
Apply $3 \times 3$ Sobel operators to $I_{\text{smooth}}$ to derive horizontal $g_x$ and vertical $g_y$ derivatives:

$$M(x, y) = \sqrt{g_x(x, y)^2 + g_y(x, y)^2}, \quad \alpha(x, y) = \arctan\left( \frac{g_y(x, y)}{g_x(x, y)} \right)$$

#### Stage 3: Non-Maximum Suppression (NMS)
NMS thins thick gradient responses into 1-pixel wide edge candidates:
1. Quantize the continuous gradient angle $\alpha(x, y)$ into one of four principal sector directions: $0^\circ$ (Horizontal), $45^\circ$ (Positive Diagonal), $90^\circ$ (Vertical), or $135^\circ$ (Negative Diagonal).
2. Compare pixel magnitude $M(x, y)$ against its two immediate neighbors along the quantized gradient direction vector $\mathbf{d}$.
3. If $M(x, y) < M(x + \mathbf{d})$ or $M(x, y) < M(x - \mathbf{d})$, suppress $M(x, y) = 0$; otherwise, retain $M(x, y)$.

#### Stage 4: Hysteresis Thresholding & Edge Linking
Reduces edge fragmentation using two thresholds ($T_{\text{high}}$ and $T_{\text{low}}$, typically at a ratio of $2:1$ or $3:1$):
- **Strong Edge**: $M(x, y) \ge T_{\text{high}}$ (Immediately preserved).
- **Weak Edge**: $T_{\text{low}} \le M(x, y) < T_{\text{high}}$ (Conditionally preserved).
- **Suppressed Non-Edge**: $M(x, y) < T_{\text{low}}$ (Set to zero).
- **Connectivity Analysis**: A weak edge pixel is retained if and only if it is 8-connected to at least one strong edge pixel; otherwise, it is eliminated.

### 4.3 Side-by-Side Code Implementation

```python
import cv2
import numpy as np
from skimage import feature

def apply_canny_edge_detector(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # --- OPENCV IMPLEMENTATION ---
    # Automatic median-based threshold setting
    median_val = np.median(img)
    lower_thresh = int(max(0, (1.0 - 0.33) * median_val))
    upper_thresh = int(min(255, (1.0 + 0.33) * median_val))

    # Canny execution
    canny_cv2 = cv2.Canny(img, threshold1=lower_thresh, threshold2=upper_thresh, L2gradient=True)

    # --- SCIKIT-IMAGE IMPLEMENTATION ---
    # Canny in skimage accepts sigma parameter for Gaussian pre-smoothing
    canny_skimage = feature.canny(img / 255.0, sigma=1.4, low_threshold=lower_thresh/255.0, high_threshold=upper_thresh/255.0)

    return canny_cv2, (canny_skimage * 255).astype(np.uint8)
```

---

## 5. Lecture 12: Feature Detection & Corner Extraction (Harris Corner Detector)

### 5.1 Local Autocorrelation Theory
Corners are interest points characterized by significant intensity variations in all spatial directions. 

Consider a local window $w(x, y)$ shifted by spatial displacement $(u, v)$. The Weighted Sum of Squared Differences (WSSD) between the original and shifted window is:

$$E(u, v) = \sum_{x, y} w(x, y) \left[ I(x + u, y + v) - I(x, y) \right]^2$$

where $w(x, y)$ is a uniform box or Gaussian weighting function $w(x, y) = \exp\left( -\frac{x^2 + y^2}{2\sigma^2} \right)$.

Using a 1st-order Taylor series expansion for small shifts $(u, v)$:

$$I(x + u, y + v) \approx I(x, y) + u I_x(x, y) + v I_y(x, y)$$

Substituting back into $E(u, v)$:

$$E(u, v) \approx \sum_{x, y} w(x, y) \left[ u I_x(x, y) + v I_y(x, y) \right]^2 = \begin{bmatrix} u & v \end{bmatrix} M \begin{bmatrix} u \\ v \end{bmatrix}$$

### 5.2 Second-Moment Matrix (Structure Tensor)
The Second-Moment Matrix $M$ summarizes the distribution of local gradient directions within the neighborhood $w(x, y)$:

$$M = \sum_{x, y} w(x, y) \begin{bmatrix} I_x^2 & I_x I_y \\ I_x I_y & I_y^2 \end{bmatrix} = \begin{bmatrix} A & C \\ C & B \end{bmatrix}$$

where $A = w * I_x^2$, $B = w * I_y^2$, and $C = w * (I_x I_y)$.

### 5.3 Eigenvalue Analysis & Corner Response Function
Let $\lambda_1$ and $\lambda_2$ be the eigenvalues of $M$:
1. **Flat Region**: $\lambda_1 \approx 0, \lambda_2 \approx 0 \implies E(u, v)$ is nearly constant for all shifts.
2. **Edge Region**: $\lambda_1 \gg 0, \lambda_2 \approx 0$ (or vice versa) $\implies E(u, v)$ changes only perpendicular to the edge direction.
3. **Corner Region**: $\lambda_1 \gg 0, \lambda_2 \gg 0 \implies E(u, v)$ increases sharply for shifts in all directions.

To compute cornerness without explicitly calculating eigenvalues, Chris Harris and Mike Stephens proposed the **Harris Corner Response Function $R$**:

$$R = \det(M) - k \cdot (\text{trace}(M))^2$$

where:
- $\det(M) = \lambda_1 \lambda_2 = AB - C^2$
- $\text{trace}(M) = \lambda_1 + \lambda_2 = A + B$
- $k$: Empirical constant (typically $k \in [0.04, 0.06]$).

```
                        λ2 (Eigenvalue 2)
                           ^
                           |   Edge Region
                           |   (λ2 >> λ1, R < 0)
             Corner Region |
             (λ1, λ2 large)|
             (R > 0)       |
                           +------------------------> λ1 (Eigenvalue 1)
                           |   Flat Region
                           |   (λ1 ≈ 0, λ2 ≈ 0, |R| small)
```

- **Corner**: $R > 0$ and $R > T_{\text{thresh}}$
- **Edge**: $R < 0$
- **Flat**: $|R| \approx 0$

### 5.4 Side-by-Side Code Implementation

```python
import cv2
import numpy as np
from skimage.feature import corner_harris, corner_peaks

def apply_harris_corner_detector(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_float = np.float32(img)

    # --- OPENCV IMPLEMENTATION ---
    # cv2.cornerHarris(src, blockSize, ksize, k)
    # blockSize: Neighborhood size for matrix M, ksize: Sobel aperture, k: Harris detector parameter
    harris_response_cv2 = cv2.cornerHarris(img_float, blockSize=2, ksize=3, k=0.04)

    # Thresholding and Non-Maximum Suppression (Dilate to find local maxima)
    dst_cv2 = cv2.dilate(harris_response_cv2, None)
    img_corners_cv2 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img_corners_cv2[dst_cv2 > 0.01 * dst_cv2.max()] = [0, 0, 255] # Red dots on corners

    # --- SCIKIT-IMAGE IMPLEMENTATION ---
    sk_response = corner_harris(img, method='k', k=0.04, sigma=1.0)
    coords = corner_peaks(sk_response, min_distance=5, threshold_rel=0.01)

    return harris_response_cv2, img_corners_cv2, coords
```

---

## 6. Lecture 13: Morphological Processing (Erosion & Dilation)

### 6.1 Set Theory Foundations
Mathematical morphology analyzes image structures using set theory. Binary images are treated as sets $A \subseteq \mathbb{Z}^2$, where elements represent foreground pixel coordinates $(x, y) \in A$.

Key Set Operations:
- **Translation of $A$ by vector $z = (z_1, z_2)$**:
  $$(A)_z = \{ a + z \mid a \in A \}$$
- **Reflection of Structuring Element (SE) $B$**:
  $$\hat{B} = \{ -b \mid b \in B \}$$
- **Complement of $A$**:
  $$A^c = \{ w \in \mathbb{Z}^2 \mid w \notin A \}$$

### 6.2 Fundamental Operations

#### Erosion ($A \ominus B$)
Erosion shrinks foreground objects by probing set $A$ with Structuring Element $B$:

$$A \ominus B = \{ z \in \mathbb{Z}^2 \mid (B)_z \subseteq A \}$$

Erosion retains pixel $z$ if and only if $B$ translated to $z$ is completely contained within $A$. It removes small boundary protrusions, disconnects thin bridges, and eliminates positive noise artifacts.

#### Dilation ($A \oplus B$)
Dilation expands foreground boundaries by probing set $A$ with reflected SE $\hat{B}$:

$$A \oplus B = \{ z \in \mathbb{Z}^2 \mid (\hat{B})_z \cap A \neq \emptyset \}$$

Dilation retains pixel $z$ if the translated SE overlaps $A$ by at least one pixel. It bridges small gaps, fills inner holes, and thickens object boundaries.

#### Morphological Duality
Erosion and dilation are dual operations under set complementation and SE reflection:

$$(A \ominus B)^c = A^c \oplus \hat{B}, \quad (A \oplus B)^c = A^c \ominus \hat{B}$$

### 6.3 Boundary Extraction Formulations
Morphological operations generate boundaries directly:
- **Internal Boundary $\beta_{\text{int}}(A)$**:
  $$\beta_{\text{int}}(A) = A \setminus (A \ominus B)$$
- **External Boundary $\beta_{\text{ext}}(A)$**:
  $$\beta_{\text{ext}}(A) = (A \oplus B) \setminus A$$
- **Morphological Gradient (Standard Boundary)**:
  $$\beta_{\text{grad}}(A) = (A \oplus B) \setminus (A \ominus B)$$

### 6.4 Side-by-Side Code Implementation

```python
import cv2
import numpy as np
from skimage import morphology

def apply_erosion_dilation(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Structuring Element Creation (3x3 Rectangular/Cross/Ellipse)
    se_cv2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # --- OPENCV IMPLEMENTATION ---
    eroded_cv2 = cv2.erode(binary_img, se_cv2, iterations=1)
    dilated_cv2 = cv2.dilate(binary_img, se_cv2, iterations=1)
    
    # Boundary Extraction
    internal_boundary = cv2.subtract(binary_img, eroded_cv2)
    external_boundary = cv2.subtract(dilated_cv2, binary_img)
    morph_gradient = cv2.subtract(dilated_cv2, eroded_cv2)

    # --- SCIKIT-IMAGE IMPLEMENTATION ---
    se_skimage = morphology.square(3)
    eroded_sk = morphology.erosion(binary_img, se_skimage)
    dilated_sk = morphology.dilation(binary_img, se_skimage)

    return eroded_cv2, dilated_cv2, internal_boundary, morph_gradient
```

---

## 7. Lecture 14: Advanced & Grayscale Morphology

### 7.1 Compound Operations: Opening & Closing

#### Morphological Opening ($A \circ B$)
Opening consists of erosion followed by dilation using the same Structuring Element $B$:

$$A \circ B = (A \ominus B) \oplus B$$

Opening eliminates small object spurs, suppresses thin outward extensions, and breaks narrow isthmuses without changing overall object geometry.

#### Morphological Closing ($A \bullet B$)
Closing consists of dilation followed by erosion using the same Structuring Element $B$:

$$A \bullet B = (A \oplus B) \ominus B$$

Closing fuses narrow gaps, bridges small fractures, and fills inner voids without significantly modifying object area.

#### Algebraic Properties
- **Idempotence**: Repeating opening or closing produces no further changes:
  $$(A \circ B) \circ B = A \circ B, \quad (A \bullet B) \bullet B = A \bullet B$$
- **Anti-Extensivity of Opening & Extensivity of Closing**:
  $$A \circ B \subseteq A \subseteq A \bullet B$$
- **Duality**:
  $$(A \circ B)^c = A^c \bullet \hat{B}, \quad (A \bullet B)^c = A^c \circ \hat{B}$$

### 7.2 Hit-or-Miss Transform
The Hit-or-Miss transform detects specific spatial configurations or shape patterns within binary images.

Given a composite Structuring Element $B = (B_1, B_2)$, where $B_1$ defines the foreground shape requirement ("hit") and $B_2$ defines the surrounding background requirement ("miss") such that $B_1 \cap B_2 = \emptyset$:

$$A \circledast B = (A \ominus B_1) \cap (A^c \ominus B_2)$$

### 7.3 Grayscale Morphology
Grayscale morphology extends binary set operations to 2D continuous/discrete scalar functions $f(x, y)$ using a 2D grayscale structuring function $b(x, y)$:

- **Grayscale Erosion**:
  $$(f \ominus b)(x, y) = \min_{(s, t) \in D_b} \{ f(x + s, y + t) - b(s, t) \}$$
  For a flat structuring element ($b(s, t) = 0$):
  $$(f \ominus b)(x, y) = \min_{(s, t) \in D_b} \{ f(x + s, y + t) \}$$
  *(Darkens image, suppresses bright peaks).*

- **Grayscale Dilation**:
  $$(f \oplus b)(x, y) = \max_{(s, t) \in D_b} \{ f(x - s, y - t) + b(s, t) \}$$
  For a flat structuring element:
  $$(f \oplus b)(x, y) = \max_{(s, t) \in D_b} \{ f(x - s, y - t) \}$$
  *(Brightens image, suppresses dark valleys).*

- **Grayscale Opening & Closing**:
  $$f \circ b = (f \ominus b) \oplus b, \quad f \bullet b = (f \oplus b) \ominus b$$

- **Top-Hat & Black-Hat Transformations**:
  - **White Top-Hat** (Isolates features brighter than their local background):
    $$T_{\text{hat}}(f) = f - (f \circ b)$$
  - **Black Top-Hat** (Isolates features darker than their local background):
    $$B_{\text{hat}}(f) = (f \bullet b) - f$$

### 7.4 Side-by-Side Code Implementation

```python
import cv2
import numpy as np
from skimage import morphology

def apply_advanced_morphology(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    # --- OPENCV IMPLEMENTATION ---
    opening_cv2 = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, se)
    closing_cv2 = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, se)
    
    # Grayscale Top-Hat and Black-Hat
    top_hat_cv2 = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, se)
    black_hat_cv2 = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, se)

    # Hit-or-Miss Transform
    kernel_hitmiss = np.array([[ 0,  1,  0],
                               [ 1, -1,  1],
                               [ 0,  1,  0]], dtype=np.int32)
    hitmiss_cv2 = cv2.morphologyEx(binary_img, cv2.MORPH_HITMISS, kernel_hitmiss)

    # --- SCIKIT-IMAGE IMPLEMENTATION ---
    se_sk = morphology.square(5)
    opening_sk = morphology.opening(binary_img, se_sk)
    closing_sk = morphology.closing(binary_img, se_sk)
    white_tophat_sk = morphology.white_tophat(img, se_sk)

    return opening_cv2, closing_cv2, top_hat_cv2, hitmiss_cv2
```

---

## 8. Lecture 15: Color Image Processing

### 8.1 Human Color Perception & Physiology
Human color perception is governed by two types of photoreceptor cells located on the retina:
- **Rods**: Highly sensitive to low illumination (scotopic vision), providing monochromatic perception.
- **Cones**: Responsible for daytime color perception (photopic vision). Cones fall into three primary spectral response categories:
  - **S-Cones (Short / Blue)**: Peak sensitivity $\approx 420\text{ nm}$
  - **M-Cones (Medium / Green)**: Peak sensitivity $\approx 534\text{ nm}$
  - **L-Cones (Long / Red)**: Peak sensitivity $\approx 564\text{ nm}$

Tristimulus theory posits that any visible color is perceived by mixing specific proportions of Primary Red ($R$), Green ($G$), and Blue ($B$) wavelengths.

### 8.2 Color Spaces & Mathematical Conversion Models

#### 1. RGB Color Model (Additive Primaries)
Represents colors in a normalized 3D Cartesian unit cube $[0, 1]^3$. Used in monitors, digital sensors, and displays.

#### 2. CMY / CMYK Color Model (Subtractive Primaries)
Used in printing devices. Cyan ($C$), Magenta ($M$), and Yellow ($Y$) represent subtractive primaries:

$$\begin{bmatrix} C \\ M \\ Y \end{bmatrix} = \begin{bmatrix} 1 \\ 1 \\ 1 \end{bmatrix} - \begin{bmatrix} R \\ G \\ B \end{bmatrix}$$

CMYK conversion extracts pure Black ($K$) to conserve ink:

$$K = \min(C, M, Y), \quad C' = \frac{C - K}{1 - K}, \quad M' = \frac{M - K}{1 - K}, \quad Y' = \frac{Y - K}{1 - K}$$

#### 3. HSV / HSI Color Spaces (Perceptual Models)
Decouples chromatic information (Hue and Saturation) from luminance/intensity (Value or Intensity):

- **Intensity / Value**:
  $$I = \frac{R + G + B}{3}, \quad V = \max(R, G, B)$$
- **Saturation**:
  $$S_{\text{HSI}} = 1 - \frac{3}{R + G + B} \min(R, G, B), \quad S_{\text{HSV}} = \begin{cases} 0 & \text{if } V = 0 \\ 1 - \frac{\min(R, G, B)}{V} & \text{otherwise} \end{cases}$$
- **Hue ($H$)**:
  $$\theta = \arccos\left( \frac{\frac{1}{2}\left[ (R - G) + (R - B) \right]}{\sqrt{(R - G)^2 + (R - B)(G - B)}} \right)$$

  $$H = \begin{cases} \theta & \text{if } B \le G \\ 360^\circ - \theta & \text{if } B > G \end{cases}$$

#### 4. YCbCr Color Space (Luminance / Chrominance)
Separates Luminance ($Y$) from Blue-difference ($Cb$) and Red-difference ($Cr$) chrominance components (ITU-R BT.601 standard):

$$Y = 0.299 R + 0.587 G + 0.114 B$$

$$Cb = -0.1687 R - 0.3313 G + 0.5 B + 128$$

$$Cr = 0.5 R - 0.4187 G - 0.0813 B + 128$$

### 8.3 Vector vs. Channel-Wise Processing
Color processing can be executed via two paradigms:
1. **Channel-Wise (Per-Channel) Processing**: Apply scalar transformations $T$ to each channel independently:
   $$g(x, y) = \left[ T\{R(x, y)\}, T\{G(x, y)\}, T\{B(x, y)\} \right]^T$$
   *(Can introduce color shifts if channels interact non-linearly).*
2. **Vector Processing**: Treat each pixel as a 3D vector $\mathbf{c}(x, y) = [R(x, y), G(x, y), B(x, y)]^T$. Vector spatial distance between colors $\mathbf{c}_1$ and $\mathbf{c}_2$:
   $$D(\mathbf{c}_1, \mathbf{c}_2) = \|\mathbf{c}_1 - \mathbf{c}_2\| = \sqrt{(R_1 - R_2)^2 + (G_1 - G_2)^2 + (B_1 - B_2)^2}$$

### 8.4 Side-by-Side Code Implementation

```python
import cv2
import numpy as np
from skimage import color

def apply_color_processing(image_path):
    # Load image in BGR (OpenCV default)
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # --- OPENCV COLOR CONVERSIONS ---
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    img_ycbcr = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    
    # HSV Color Segmentation (Isolating specific color ranges e.g., Red)
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    color_mask = cv2.inRange(img_hsv, lower_red, upper_red)
    segmented_red = cv2.bitwise_and(img_bgr, img_bgr, mask=color_mask)

    # Per-Channel Equalization in HSV (Equalizing V channel only)
    h, s, v = cv2.split(img_hsv)
    v_eq = cv2.equalizeHist(v)
    img_hsv_eq = cv2.merge([h, s, v_eq])
    img_bgr_equalized = cv2.cvtColor(img_hsv_eq, cv2.COLOR_HSV2BGR)

    # --- SCIKIT-IMAGE COLOR CONVERSIONS ---
    sk_hsv = color.rgb2hsv(img_rgb)
    sk_lab = color.rgb2lab(img_rgb)
    sk_ycbcr = color.rgb2ycbcr(img_rgb)

    return img_hsv, img_ycbcr, segmented_red, img_bgr_equalized
```

---

## 9. Comparative Summary & Filter Selection Matrix

| Lecture & Filter Type | Primary Mathematical Operator | Input / Output Space | Typical Application | Key Characteristics |
| :--- | :--- | :--- | :--- | :--- |
| **L09: Laplacian** | $\nabla^2 f = \frac{\partial^2 f}{\partial x^2} + \frac{\partial^2 f}{\partial y^2}$ | Grayscale $\to$ Grayscale | High-frequency detail sharpening | Isotropic 2nd derivative; sensitive to noise. |
| **L09: Unsharp Masking** | $g = f + k(f - \bar{f})$ | Grayscale / Color $\to$ Same | Detail enhancement | Subtracts blurred image; amplifies edges. |
| **L10: Sobel / Prewitt** | $g_x, g_y$ 1st-order differences | Grayscale $\to$ Vector Magnitude & Angle | Edge magnitude & direction estimation | Combines smoothing with derivative filters. |
| **L11: Canny Detector** | 4-Stage Optimization Pipeline | Grayscale $\to$ Binary Edge Map | Optimal boundary extraction | NMS + Hysteresis thresholding; single response. |
| **L12: Harris Corner** | $R = \det(M) - k \cdot (\text{trace}(M))^2$ | Grayscale $\to$ Interest Point Map | Feature detection & image matching | Evaluates local matrix $M$ eigenvalues $\lambda_1, \lambda_2$. |
| **L13: Erosion / Dilation** | $A \ominus B, A \oplus B$ | Binary / Gray $\to$ Binary / Gray | Shrinking/expanding object features | Probes geometry using Structuring Element $B$. |
| **L14: Opening / Closing** | $(A \ominus B) \oplus B, (A \oplus B) \ominus B$ | Binary / Gray $\to$ Binary / Gray | Noise suppression & structural restoration | Idempotent filters; preserves global scale. |
| **L14: Top-Hat Filter** | $f - (f \circ b)$ | Grayscale $\to$ Grayscale | Uneven illumination correction | Isolates features brighter than local background. |
| **L15: Color Transforms** | Color Space Coordinate Mappings | RGB $\leftrightarrow$ HSV / YCbCr | Perceptual segmentation & compression | Decouples luminance $Y/V$ from chrominance $H/S/Cb/Cr$. |

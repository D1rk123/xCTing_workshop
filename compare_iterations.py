import numpy as np
import torch
import tomosipo as ts
import ts_algorithms as tsa
import matplotlib.pyplot as plt

class TrackNagLsResidualCb(tsa.TrackMetricCb):
        def __init__(self, A, y_reference, l2_regularization, keep_best_x=False, early_stopping_iterations=None):
            super().__init__(keep_best_x=keep_best_x, early_stopping_iterations=early_stopping_iterations)
            self._y_reference = y_reference
            self._A = A
            self._l2_regularization = l2_regularization
            
        def calc_metric(self, x, iteration):
            squared_residual_error = (self._A(x) - self._y_reference) ** 2
            return torch.sum(squared_residual_error) + self._l2_regularization * torch.sum(x**2)

def add_poisson_noise(img, photon_count, attenuation_factor=1):
    img = img * attenuation_factor
    opt = dict(dtype=np.float32)
    img = np.exp(-img, **opt)
    img *= photon_count
    # Add poisson noise and retain scale by dividing by photon_count
    img = np.random.poisson(img)
    img = img / photon_count
    img[img == 0] = 1
    # Redo log transform and scale img to range [0, img_max] +- some noise.
    img = -np.log(img, **opt)
    return img / attenuation_factor

if __name__ == "__main__":
    # Setup 2D volume and parallel projection geometry
    vg = ts.volume(shape=(1, 256, 256))
    pg = ts.parallel(angles=128, shape=(1, 384))

    # Create an operator from the geometries
    A = ts.operator(vg, pg)

    # Create hollow cube phantom
    x = torch.zeros(A.domain_shape, dtype=torch.float32)
    x[:, 10:-10, 10:-10] = 1.0
    x[:, 30:-30, 30:-30] = 0.0
    x[:, 50:-50, 50:-50] = 0.25
    x[:, 70:-70, 70:-70] = 0.5
    x[:, 90:-90, 90:-90] = 0.75
    x[:, 110:-110, 110:-110] = 1

    # Project the volume data to obtain the projection data and add Poisson noise
    y = A(x)
    y = torch.from_numpy(add_poisson_noise(y.numpy(), 10000, 0.01))
    
    nag_mse_cb = tsa.TrackMseCb(x)
    sirt_mse_cb = tsa.TrackMseCb(x)
    
    r_fbp = tsa.fbp(A, y)
    fbp_mse = torch.mean((r_fbp-x)**2)
    
    r_sirt = tsa.sirt(A, y, 100)
    r_nag10 = tsa.nag_ls(A, y, 10)
    r_nag29 = tsa.nag_ls(A, y, 29)
    r_nag2000 = tsa.nag_ls(A, y, 2000, callbacks=(nag_mse_cb, ))
    r_sirt2000 = tsa.sirt(A, y, 2000, callbacks=(sirt_mse_cb, ))
    
    plt.figure(figsize=(12, 7))
    plt.subplot(231); plt.imshow(x[0, ...].numpy(), vmin=-0.05, vmax=1.05); plt.title("original image"); plt.colorbar()
    plt.subplot(232); plt.imshow(y[0, ...].numpy()); plt.title("noisy projection data"); plt.colorbar()
    plt.subplot(233); plt.imshow(r_fbp[0, ...].numpy(), vmin=-0.05, vmax=1.05); plt.title("fbp reconstruction"); plt.colorbar()
    plt.subplot(234); plt.imshow(r_nag10[0, ...].numpy(), vmin=-0.05, vmax=1.05); plt.title("nag_ls reconstruction(10 iters)"); plt.colorbar()
    plt.subplot(235); plt.imshow(r_nag29[0, ...].numpy(), vmin=-0.05, vmax=1.05); plt.title("nag_ls reconstruction(29 iters)"); plt.colorbar()
    plt.subplot(236); plt.imshow(r_nag2000[0, ...].numpy(), vmin=-0.05, vmax=1.05); plt.title("nag_ls reconstruction(2000 iters)"); plt.colorbar()
    plt.tight_layout()
    plt.show()
    
    print(f"fbp_mse = {fbp_mse}")
    print(f"best sirt_mse = {sirt_mse_cb.best_score} at iteration {sirt_mse_cb.best_iteration}")
    print(f"best nag_mse = {nag_mse_cb.best_score} at iteration {nag_mse_cb.best_iteration}")
    
    plt.figure(figsize=(8, 6))
    plt.plot(np.arange(1,2001), sirt_mse_cb.metric_log, label="sirt")
    plt.plot(np.arange(1,2001), nag_mse_cb.metric_log, label="nag_ls")
    plt.plot((1, 2000), (fbp_mse, fbp_mse), label="fbp")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of iterations")
    plt.ylabel("MSE between reconstructed and original image")
    plt.legend()
    plt.title("Reconstruction error over number of iterations")
    plt.tight_layout()
    plt.show()
    
    nag0_mse_cb = tsa.TrackMseCb(x)
    nag10_mse_cb = tsa.TrackMseCb(x)
    nag100_mse_cb = tsa.TrackMseCb(x)
    nag1000_mse_cb = tsa.TrackMseCb(x)
    
    residual0_cb = TrackNagLsResidualCb(A, y, 0)
    residual10_cb = TrackNagLsResidualCb(A, y, 10)
    residual100_cb = TrackNagLsResidualCb(A, y, 100)
    residual1000_cb = TrackNagLsResidualCb(A, y, 1000)
    
    r_nag_l2_0 = tsa.nag_ls(A, y, 500, l2_regularization=0, callbacks=(nag0_mse_cb, residual0_cb))
    r_nag_l2_10 = tsa.nag_ls(A, y, 500, l2_regularization=10, callbacks=(nag10_mse_cb, residual10_cb))
    r_nag_l2_100 = tsa.nag_ls(A, y, 500, l2_regularization=100, callbacks=(nag100_mse_cb, residual100_cb))
    r_nag_l2_1000 = tsa.nag_ls(A, y, 500, l2_regularization=1000, callbacks=(nag1000_mse_cb, residual1000_cb))
    
    plt.figure(figsize=(12, 7))
    plt.subplot(231); plt.imshow(x[0, ...].numpy(), vmin=-0.05, vmax=1.05); plt.title("original image"); plt.colorbar()
    plt.subplot(232); plt.imshow(y[0, ...].numpy()); plt.title("noisy projection data"); plt.colorbar()
    plt.subplot(233); plt.imshow(r_fbp[0, ...].numpy(), vmin=-0.05, vmax=1.05); plt.title("fbp reconstruction"); plt.colorbar()
    plt.subplot(234); plt.imshow(r_nag_l2_10[0, ...].numpy(), vmin=-0.05, vmax=1.05); plt.title("nag_ls reconstruction($\lambda=10$)"); plt.colorbar()
    plt.subplot(235); plt.imshow(r_nag_l2_100[0, ...].numpy(), vmin=-0.05, vmax=1.05); plt.title("nag_ls reconstruction($\lambda=100$)"); plt.colorbar()
    plt.subplot(236); plt.imshow(r_nag_l2_1000[0, ...].numpy(), vmin=-0.05, vmax=1.05); plt.title("nag_ls reconstruction($\lambda=1000$)"); plt.colorbar()
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(8, 6))
    plt.plot(np.arange(1,501), nag0_mse_cb.metric_log, label="nag_ls ($\lambda=0$)")
    plt.plot(np.arange(1,501), nag10_mse_cb.metric_log, label="nag_ls ($\lambda=10$)")
    plt.plot(np.arange(1,501), nag100_mse_cb.metric_log, label="nag_ls ($\lambda=100$)")
    plt.plot(np.arange(1,501), nag1000_mse_cb.metric_log, label="nag_ls ($\lambda=1000$)")
    plt.plot((1, 501), (fbp_mse, fbp_mse), label="fbp")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of iterations")
    plt.ylabel("MSE between reconstructed and original image")
    plt.legend()
    plt.title("Reconstruction error with different l2 regularization levels")
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(8, 6))
    plt.plot(np.arange(1,501), residual0_cb.metric_log, label="nag_ls ($\lambda=0$)")
    plt.plot(np.arange(1,501), residual10_cb.metric_log, label="nag_ls ($\lambda=10$)")
    plt.plot(np.arange(1,501), residual100_cb.metric_log, label="nag_ls ($\lambda=100$)")
    plt.plot(np.arange(1,501), residual1000_cb.metric_log, label="nag_ls ($\lambda=1000$)")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of iterations")
    plt.ylabel("residual")
    plt.legend()
    plt.title("Residual with different l2 regularization levels")
    plt.tight_layout()
    plt.show()
    
    best_nag_mse_cb = tsa.TrackMseCb(x, keep_best_x=True)
    best_nag_min_mse_cb = tsa.TrackMseCb(x, keep_best_x=True)
    best_nag_min_max_mse_cb = tsa.TrackMseCb(x, keep_best_x=True)
    
    tsa.nag_ls(A, y, 500, callbacks=(best_nag_mse_cb, ))
    tsa.nag_ls(A, y, 500, min_constraint=0, callbacks=(best_nag_min_mse_cb, ))
    tsa.nag_ls(A, y, 500, min_constraint=0, max_constraint=1, callbacks=(best_nag_min_max_mse_cb, ))

    plt.figure(figsize=(12, 7))
    plt.subplot(231); plt.imshow(x[0, ...].numpy(), vmin=-0.05, vmax=1.05); plt.title("original image"); plt.colorbar()
    plt.subplot(232); plt.imshow(y[0, ...].numpy()); plt.title("noisy projection data"); plt.colorbar()
    plt.subplot(233); plt.imshow(r_fbp[0, ...].numpy(), vmin=-0.05, vmax=1.05); plt.title("fbp reconstruction"); plt.colorbar()
    plt.subplot(234); plt.imshow(best_nag_mse_cb.best_x[0, ...].numpy(), vmin=-0.05, vmax=1.05); plt.title("nag_ls reconstruction"); plt.colorbar()
    plt.subplot(235); plt.imshow(best_nag_min_mse_cb.best_x[0, ...].numpy(), vmin=-0.05, vmax=1.05); plt.title("nag_ls (with min constraint)"); plt.colorbar()
    plt.subplot(236); plt.imshow(best_nag_min_max_mse_cb.best_x[0, ...].numpy(), vmin=-0.05, vmax=1.05); plt.title("nag_ls (with min & max constraint)"); plt.colorbar()
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(8, 6))
    plt.plot(np.arange(1,501), best_nag_mse_cb.metric_log, label="nag_ls")
    plt.plot(np.arange(1,501), best_nag_min_mse_cb.metric_log, label="nag_ls (with min constraint)")
    plt.plot(np.arange(1,501), best_nag_min_max_mse_cb.metric_log, label="nag_ls (with min & max constraint)")
    plt.plot((1, 501), (fbp_mse, fbp_mse), label="fbp")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of iterations")
    plt.ylabel("MSE between reconstructed and original image")
    plt.legend()
    plt.title("Reconstruction error with different constraints")
    plt.tight_layout()
    plt.show()
    
    best_tv_min_mse_cb_0005 = tsa.TrackMseCb(x, keep_best_x=True)
    best_tv_min_mse_cb_001 = tsa.TrackMseCb(x, keep_best_x=True)
    best_tv_min_mse_cb_01 = tsa.TrackMseCb(x, keep_best_x=True)
    
    tsa.tv_min2d(A, y, 0.0005, 5000, non_negativity=True, callbacks=(best_tv_min_mse_cb_0005, ))
    tsa.tv_min2d(A, y, 0.01, 5000, non_negativity=True, callbacks=(best_tv_min_mse_cb_001, ))
    tsa.tv_min2d(A, y, 0.1, 5000, non_negativity=True, callbacks=(best_tv_min_mse_cb_01, ))

    plt.figure(figsize=(12, 7))
    plt.subplot(231); plt.imshow(x[0, ...].numpy(), vmin=-0.05, vmax=1.05); plt.title("original image"); plt.colorbar()
    plt.subplot(232); plt.imshow(y[0, ...].numpy()); plt.title("noisy projection data"); plt.colorbar()
    plt.subplot(233); plt.imshow(r_fbp[0, ...].numpy(), vmin=-0.05, vmax=1.05); plt.title("fbp reconstruction"); plt.colorbar()
    plt.subplot(234); plt.imshow(best_tv_min_mse_cb_0005.best_x[0, ...].numpy(), vmin=-0.05, vmax=1.05); plt.title("tv_min ($\lambda=0.005$)"); plt.colorbar()
    plt.subplot(235); plt.imshow(best_tv_min_mse_cb_001.best_x[0, ...].numpy(), vmin=-0.05, vmax=1.05); plt.title("tv_min ($\lambda=0.01$)"); plt.colorbar()
    plt.subplot(236); plt.imshow(best_tv_min_mse_cb_01.best_x[0, ...].numpy(), vmin=-0.05, vmax=1.05); plt.title("tv_min ($\lambda=0.1$)"); plt.colorbar()
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(8, 6))
    plt.plot(np.arange(1,5001), best_tv_min_mse_cb_0005.metric_log, label="tv_min ($\lambda=0.005$)")
    plt.plot(np.arange(1,5001), best_tv_min_mse_cb_001.metric_log, label="tv_min ($\lambda=0.01$)")
    plt.plot(np.arange(1,5001), best_tv_min_mse_cb_01.metric_log, label="tv_min ($\lambda=0.1$)")
    plt.plot((1, 5001), (fbp_mse, fbp_mse), label="fbp")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of iterations")
    plt.ylabel("MSE between reconstructed and original image")
    plt.legend()
    plt.title("Reconstruction error with different constraints")
    plt.tight_layout()
    plt.show()


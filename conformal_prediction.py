# %%
import os
import torch
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

#%%
from neural_lam.models.graph_lam import GraphLAM
from neural_lam.models.hi_lam import HiLAM

from neural_lam.weather_dataset import WeatherDataset
from neural_lam import constants, utils

# %%
plt.rcParams['text.usetex'] = True

# %%
# Paths to saved models
CKPT_DIR = "/proj/berzelius-2022-164/weather/ws_res_models"
HI_LAM_WMSE_PATH = os.path.join(CKPT_DIR, "Hi-LAM.ckpt")
HI_LAM_NLL_PATH = os.path.join(CKPT_DIR, "hi_lam_nll.ckpt")
#GC_LAM_PATH = os.path.join(CKPT_DIR, "GC-LAM.ckpt")

# %%
# Remove boundary area of 10 x 2 grid cells
INTERIOR_SHAPE = tuple(dim - 2*10 for dim in constants.GRID_SHAPE)

# Actual config for CP
MODEL = "Hi-LAM" # Hi-LAM or GC-LAM
# must match sub-directory in data
DS_NAME = "conf_data_sep" # sep 2021 + 2022
#DS_NAME = "meps_example"
BATCH_SIZE = 10
PLOT_DIR = "cp_plots"
PRED_PLOT_DIR = "cp_pred_plots"

#%%
@torch.no_grad()
def predict_on_batch(model, batch):
    """
    Performs prediction for batch using model, returns both prediction and target.
    Extracts the inner forecasting area, removing the boundary where prediction=target
    Also reshapes grid nodes to 2d spatial dimensions

    Returns:
    Prediction, (B, T, N_y, N_x, d_X)
    Target, (B, T, N_y, N_x, d_X)

    Shapes:
    B = BATCH_SIZE (see above)
    T = 19 (time steps)
    N_y = 248 (y-resolution of grid cells)
    N_x = 218 (x-resolution of grid cells)
    d_X = 17 (Number of weather variables being forecast in each grid cell)
    """
    # Predict
    prediction, target, pred_std = model.common_step(batch) # Both (B, T, N, d_X)

    # Remove boundary
    interior_mask = (model.interior_mask[:,0] == 1.) # boolean, (N,)
    new_shape = target.shape[:2] + INTERIOR_SHAPE + target.shape[3:]

    interior_prediction = prediction[:,:,interior_mask].reshape(new_shape)
    interior_target = target[:,:,interior_mask].reshape(new_shape)

    if len(pred_std.shape) > 1:
        # Actually outputs interior std
        pred_std = pred_std[:,:,interior_mask].reshape(new_shape)

    return (
        interior_prediction,
        interior_target,
        pred_std,
    )

@torch.no_grad()
def predict_on_all():
    # Load data
    # "val" and "test" here correspond to sub-directories in `data/DS_NAME/samples`,
    # so one could for example structure things be having one directory "calibration"
    val_loader = torch.utils.data.DataLoader(WeatherDataset(DS_NAME, split="val"),
            BATCH_SIZE, shuffle=False, num_workers=8)
    test_loader = torch.utils.data.DataLoader(WeatherDataset(DS_NAME, split="test"),
            BATCH_SIZE, shuffle=False, num_workers=8)

    # Instantiate model
    if MODEL == "Hi-LAM":
        model_class = HiLAM
        graph_name = "hierarchical"
    else:
        model_class = GraphLAM
        graph_name = "multiscale"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    for model_name, ckpt_path, save_std in (
        ("hi_lam_nll", HI_LAM_NLL_PATH, True),
        ("hi_lam_wmse", HI_LAM_WMSE_PATH, False),
    ):
        # Need to adjust a few parameters in the model arguments to load correctly
        model_args = torch.load(ckpt_path,
                map_location="cpu")["hyper_parameters"]["args"]
        model_args.dataset = DS_NAME
        model_args.graph = graph_name
        model_args.output_std = int(save_std) # Need to add on for older model

        model_save_dir = os.path.join("saved_data", model_name)
        os.makedirs(model_save_dir, exist_ok=True)

        model = model_class.load_from_checkpoint(ckpt_path, args=model_args).to(device)

        for subset_name, loader in (
            ("val", val_loader),
            ("test", test_loader),
        ):
            print(f"Computing predictions: Model = {model_name}, Set = {subset_name}")
            # Perform forward pass using model
            pred = []
            targ = []
            stds = []
            for batch in tqdm(val_loader):
                batch = (t.to(device) for t in batch)
                prediction, target, pred_std = predict_on_batch(model, batch)
                # all of shape (B, T, N_y, N_x, d_X)

                pred.append(prediction)
                targ.append(target)
                stds.append(pred_std)

            pred_np = torch.cat(pred, dim=0).cpu().numpy()
            np.save(os.path.join(model_save_dir, f"{subset_name}_pred.npy"), pred_np)
            del pred_np

            targ_np = torch.cat(targ, dim=0).cpu().numpy()
            np.save(os.path.join(model_save_dir, f"{subset_name}_target.npy"), targ_np)
            del targ_np

            if save_std:
                all_std = torch.cat(stds, dim=0)
                std_np = all_std.cpu().numpy()
                np.save(os.path.join(model_save_dir, f"{subset_name}_std.npy"), std_np)

def non_conformity(pred, target, cal_idx):
    print("Computing non-conformity scores")
    n_samples = len(cal_idx)
    cal_scores_list = []
    # Iterate over samples, only requiring keeping one in memory at a time
    for num_samples, sample_i in enumerate(cal_idx, start=1):
        cal_scores_list.append(np.abs(pred[sample_i] - target[sample_i]))
        if num_samples % 10 == 0:
            print(f"Sample {num_samples}/{n_samples}")

    cal_scores = np.stack(cal_scores_list, axis=0)
    #cal_scores = np.abs(pred-target)
    return cal_scores

def qhat_estimate(cal_scores, alpha, overwrite_input):
    n = len(cal_scores)
    return np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, axis = 0,
            method='higher', overwrite_input=overwrite_input)

def predict_coverage(pred, targets, qhat):
    prediction_sets = [pred - qhat, pred + qhat]
    empirical_coverage = ((targets >= prediction_sets[0]) &
            (targets <= prediction_sets[1])).mean()
    #print(f"The empirical coverage after calibration is: {empirical_coverage}")
    return empirical_coverage, prediction_sets

# %%

def compute_qhats(preds, targets, cal_idx, alpha_levels):
    #cal_scores = non_conformity(preds[cal_idx], targets[cal_idx])
    cal_scores = non_conformity(preds, targets, cal_idx)
    # qhat = qhat_estimate(cal_scores, alpha)
    # coverage, pred_sets = predict_coverage(preds[pred_idx], targets[pred_idx],  qhat)

    # Test coverage across values of alpha
    qhats = qhat_estimate(cal_scores, alpha_levels, overwrite_input=True)
    # Ok to overwrite input, we don't need cal_scores after this
    return qhats # (N_qhats, T, N_y, N_x, dim_var)

def plot_emp_cov(preds, targets, qhats, pred_idx, alpha_levels):
    emp_cov = []
    #for alpha in tqdm(alpha_levels):
        #qhat = qhat_estimate(cal_scores, alpha)
    for qhat in tqdm(qhats):
        # Iterate over samples to save memory (pred and target are memmapped)
        sample_coverages = [predict_coverage(preds[sample_i], targets[sample_i], qhat)[0]
                for sample_i in pred_idx]
        emp_cov.append(np.mean(np.array(sample_coverages)))
        #emp_cov.append(predict_coverage(preds[pred_idx], targets[pred_idx], qhat)[0])

    # Plot empirical coverage
    plt.plot(1-alpha_levels, 1-alpha_levels, label='Ideal', color ='black', alpha=0.75)
    plt.plot(1-alpha_levels, emp_cov, label='Residual' ,ls='-.', color='teal',
            alpha=0.75)
    plt.xlabel(r'1-$\alpha$')
    plt.ylabel('Empirical Coverage')
    plt.title("June")
    plt.legend()
    plt.grid() #Comment out if you dont want grids.
    # plt.savefig("June-Graphcast.svg", format="svg", bbox_inches='tight')
    plt.show()

def plot_slice(preds, targets, pred_idx, qhat, alpha):
    #Slicing along X-Axis

    idx = 10
    var = 10
    y_pos = 20
    time = 2
    x_grid = np.linspace(0, 1, 248)

    pred = preds[pred_idx[idx]]
    target = targets[pred_idx[idx]]
    pred_sets = [pred - qhat, pred + qhat]

    plt.figure()
    plt.plot(x_grid, pred[time, :, y_pos, var], label='Pred.', alpha=0.8,
            color = 'firebrick')
    plt.plot(x_grid, pred_sets[0][time, :, y_pos, var], label='Lower', alpha=0.8,
            color = 'teal', ls='--')
    plt.plot(x_grid, pred_sets[1][time,:, y_pos, var], label='Upper', alpha=0.8,
            color = 'navy', ls='--')
    plt.plot(x_grid, target[time,:, y_pos, var], label='Actual', alpha=0.8,
            color = 'black')
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('var')
    plt.title('Coverage at alpha = ' + str(alpha))
    plt.grid() #Comment out if you dont want grids.
    plt.show()

def plot_qhat(q_hat, alpha, data_std):
    """
    Plot predictions intervals (q_hat)

    q_hat: (T, N_y, N_x, dim_var)
    alpha: scalar
    data_std: (dim_var,)
    """
    time_indices = (4, 10, 18)

    # Recompute extent of plotting area, as boundary is removed
    boundary_size_x = (10/constants.GRID_SHAPE[0])*(
            constants.GRID_LIMITS[1]-constants.GRID_LIMITS[0])
    boundary_size_y = (10/constants.GRID_SHAPE[1])*(
            constants.GRID_LIMITS[3]-constants.GRID_LIMITS[2])
    internal_grid_limits = (
        constants.GRID_LIMITS[0] + boundary_size_x, # min x
        constants.GRID_LIMITS[1] - boundary_size_x, # max x
        constants.GRID_LIMITS[2] + boundary_size_y, # min y
        constants.GRID_LIMITS[3] - boundary_size_y, # max y
    )

    # Make one plot per variable
    for var_i, (var_name, var_unit, var_std)in enumerate(zip(constants.PARAM_NAMES_SHORT,
            constants.PARAM_UNITS, data_std)):
        # Extract q_hat:s to plot, and rescale to original data scale
        q_hat_plot = q_hat[time_indices,:,:,var_i]*var_std # (3, N_y, N_x)

        # Compute range of values (makes most sense to keep vmin=0)
        vmin = 0 # q_hat_plot.min()
        vmax = q_hat_plot.max()

        fig, axes = plt.subplots(1, len(time_indices), figsize=(10,3),
                subplot_kw={"projection": constants.LAMBERT_PROJ})

        for q_hat_field, ax, time_step in zip(q_hat_plot, axes, time_indices):
            ax.coastlines(linewidth=0.3) # Add coastline outlines
            im = ax.imshow(q_hat_field, origin="lower",
                        extent=internal_grid_limits, vmin=vmin, vmax=vmax,
                        cmap="Reds")

            # Add lead time title to subplots
            ax.set_title(f"{3*(time_step+1)} h")

        # Add color bar
        cbar = fig.colorbar(im, ax=axes, orientation="vertical", fraction=.01, aspect=50,
                pad=0.01)
        cbar.ax.set_ylabel("$\hat{q}$" + f" ({var_unit})")

        # Save
        fig.savefig(os.path.join(PLOT_DIR, f"alpha_{alpha}_{var_name}_qhat.pdf"))
    plt.close("all") # Close all figs

def plot_pred(pred, target, data_mean, data_std):
    """
    Plot predictions and targets

    pred: (T, N_y, N_x, dim_var)
    target: (T, N_y, N_x, dim_var)
    data_mean: (dim_var,)
    data_std: (dim_var,)
    """
    time_indices = (4, 10, 18)

    # Recompute extent of plotting area, as boundary is removed
    boundary_size_x = (10/constants.GRID_SHAPE[0])*(
            constants.GRID_LIMITS[1]-constants.GRID_LIMITS[0])
    boundary_size_y = (10/constants.GRID_SHAPE[1])*(
            constants.GRID_LIMITS[3]-constants.GRID_LIMITS[2])
    internal_grid_limits = (
        constants.GRID_LIMITS[0] + boundary_size_x, # min x
        constants.GRID_LIMITS[1] - boundary_size_x, # max x
        constants.GRID_LIMITS[2] + boundary_size_y, # min y
        constants.GRID_LIMITS[3] - boundary_size_y, # max y
    )

    # Make one plot per variable
    for var_i, (var_name, var_unit, var_mean, var_std)in enumerate(zip(constants.PARAM_NAMES_SHORT,
            constants.PARAM_UNITS, data_mean, data_std)):
        # Extract q_hat:s to plot, and rescale to original data scale
        pred_plot = pred[time_indices,:,:,var_i]*var_std + var_mean # (3, N_y, N_x)
        target_plot = target[time_indices,:,:,var_i]*var_std + var_mean # (3, N_y, N_x)

        # Compute range of values
        vmin = min(pred_plot.min(), target_plot.min())
        vmax = min(pred_plot.max(), target_plot.max())

        fig, axes = plt.subplots(2, len(time_indices), figsize=(8,6),
                subplot_kw={"projection": constants.LAMBERT_PROJ})

        for fields, axes_row in zip((pred_plot, target_plot), axes):
            for field, ax, time_step in zip(fields, axes_row, time_indices):
                ax.coastlines(linewidth=0.3) # Add coastline outlines
                im = ax.imshow(field, origin="lower",
                            extent=internal_grid_limits, vmin=vmin, vmax=vmax,
                            cmap="plasma")

                # Add lead time title to subplots
                ax.set_title(f"{3*(time_step+1)} h")

        # Write out pred and target
        axes[0,0].set_ylabel("Prediction", size="large")
        axes[0,0].set_yticks([])
        axes[1,0].set_ylabel("Ground Truth", size="large")
        axes[1,0].set_yticks([])

        # Add color bar
        cbar = fig.colorbar(im, ax=axes, orientation="vertical", fraction=.01, aspect=50,
                pad=0.01)
        if var_unit != "-":
            cbar.ax.set_ylabel(f"{var_unit}")

        # Save
        fig.savefig(os.path.join(PRED_PLOT_DIR, f"{var_name}.pdf"),
                bbox_inches='tight')
    plt.close("all") # Close all figs

if __name__ == "__main__":

    # Obtaining the data by running the model.
    predict_on_all()
    exit()

    # Set seed
    np.random.seed(42)

    #Loading the prediction and target data. pre-saved.
    preds = np.load(os.getcwd() + '/saved_data/preds_june.npy',
            mmap_mode="r") # (B, T, N_y, N_x, d_X)
    targets = np.load(os.getcwd() + '/saved_data/targets_june.npy',
            mmap_mode="r") # (B, T, N_y, N_x, d_X)
    print("Loaded data!")

    ncal = 200
    nsamples = preds.shape[0]
    alpha = 0.1
    rand_idx = np.random.randint(0, nsamples, nsamples)
    cal_idx = rand_idx[:ncal]
    pred_idx = rand_idx[ncal:]
    pred_idx = np.delete(np.arange(nsamples), cal_idx)
    alpha_levels = np.arange(0.05, 0.95, 0.1)

    # Compute qhat from saved predictions and targets
    #  qhats = compute_qhats(preds, targets, cal_idx, alpha_levels)
    #  np.save(os.getcwd() + '/saved_data/qhats_june.npy', qhats)
    qhats = np.load(os.getcwd() + '/saved_data/qhats_june.npy')

    # Compute and plot empirical coverage
    #plot_emp_cov(preds, targets, qhats, pred_idx, alpha_levels)

    # Plot slice of data
    #  plot_slice(preds, targets, pred_idx, qhats[0], alpha_levels[0])
    #  plot_slice(preds, targets, pred_idx, qhats[-1], alpha_levels[-1])

    # Get data stats
    static_data = utils.load_static_data(DS_NAME)
    data_std = static_data["data_std"].numpy()
    data_mean = static_data["data_mean"].numpy()

    # Plot q-hat spatio-temporally
    #  os.makedirs(PLOT_DIR, exist_ok=True)
    #  plot_qhat(qhats[0], alpha_levels[0], data_std)
    #  plot_qhat(qhats[-1], alpha_levels[-1], data_std)

    # Plot example prediction
    os.makedirs(PRED_PLOT_DIR, exist_ok=True)
    example_pred = preds[pred_idx[0]] # From pred set
    example_target = targets[pred_idx[0]]
    plot_pred(example_pred, example_target, data_mean, data_std)

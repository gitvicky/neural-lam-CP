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
from neural_lam import constants

# %%
plt.rcParams['text.usetex'] = True

# %%
# Paths to saved models
CKPT_DIR = "saved_models"
HI_LAM_PATH = os.path.join(CKPT_DIR, "Hi-LAM.ckpt")
GC_LAM_PATH = os.path.join(CKPT_DIR, "GC-LAM.ckpt")

# %%
# Remove boundary area of 10 x 2 grid cells
INTERIOR_SHAPE = tuple(dim - 2*10 for dim in constants.grid_shape)

# Actual config for CP
MODEL = "Hi-LAM" # Hi-LAM or GC-LAM
# DS_NAME = "meps_example" # must match sub-directory in data
DS_NAME = "validation_june" #270 data points
BATCH_SIZE = 10

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
    prediction, target = model.common_step(batch) # Both (B, T, N, d_X)

    # Remove boundary
    interior_mask = (model.interior_mask[:,0] == 1.) # boolean, (N,)
    interior_prediction = prediction[:,:,interior_mask]
    interior_target = target[:,:,interior_mask]

    new_shape = interior_target.shape[:2] + INTERIOR_SHAPE + interior_target.shape[3:]
    return interior_prediction.reshape(new_shape), interior_target.reshape(new_shape)

def main():
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
        ckpt_path = HI_LAM_PATH
        graph_name = "hierarchical"
    else:
        model_class = GraphLAM
        ckpt_path = GC_LAM_PATH
        graph_name = "multiscale"

    # Need to adjust a few parameters in the model arguments to load correctly
    model_args = torch.load(ckpt_path, map_location="cpu")["hyper_parameters"]["args"]
    model_args.dataset = DS_NAME
    model_args.graph = graph_name

    model = model_class.load_from_checkpoint(ckpt_path, args=model_args)

    # Perform forward pass using model
    pred = []
    targ = []
    for batch in tqdm(val_loader):
        prediction, target = predict_on_batch(model, batch)
        pred.append(prediction)
        targ.append(target)
        # Prediction and target of shape (B, T, N_y, N_x, d_X)
        # See predict_on_batch doc-string

        # Do something cool with prediction and target ...

        # Can for example plot like
        #  plt.imshow(prediction[0,18,:,:,0], origin="lower", cmap="plasma")
        #  plt.show()
        # See also `neural_lam/vis.py` for plotting inspiration
    return pred, targ

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

if __name__ == "__main__":

    # Obtaining the data by running the model.
    # preds, targets = main()
    # preds = torch.vstack(preds).numpy()
    # targets = torch.vstack(targets).numpy()

    # Set seed
    np.random.seed(42)

    #Loading the prediction and target data. pre-saved.
    preds = np.load(os.getcwd() + '/saved_data/preds_june.npy', mmap_mode="r")
    targets = np.load(os.getcwd() + '/saved_data/targets_june.npy', mmap_mode="r")
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
    #qhats = compute_qhats(preds, targets, cal_idx, alpha_levels)
    #np.save(os.getcwd() + '/saved_data/qhats_june.npy', qhats)
    qhats = np.load(os.getcwd() + '/saved_data/qhats_june.npy')

    # Compute and plot empirical coverage
    #plot_emp_cov(preds, targets, qhats, pred_idx, alpha_levels)

    # Plot slice of data
    plot_slice(preds, targets, pred_idx, qhats[0], alpha_levels[0])
    plot_slice(preds, targets, pred_idx, qhats[-1], alpha_levels[-1])

# %%


# %%

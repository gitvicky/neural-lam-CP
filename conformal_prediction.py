# %%
import os
import torch

from neural_lam.models.graph_lam import GraphLAM
from neural_lam.models.hi_lam import HiLAM

from neural_lam.weather_dataset import WeatherDataset
from neural_lam import constants

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
DS_NAME = "meps_example" # must match sub-directory in data
DS_NAME = "validation_june"
BATCH_SIZE = 1

# %% 
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
    for batch in val_loader:
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
    return torch.vstack(pred), torch.vstack(target)

if __name__ == "__main__":
    preds, targets = main()

# %% 
# def calibrate(pred, target, alpha):

#     n = len(pred)
#     cal_scores = np.abs(pred.numpy()-target.numpy())           
#     qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, axis = 0, method='higher')
#     return qhat 

# def predict_coverage(pred, qhat):
#     prediction_sets = [pred - qhat, pred + qhat]
#     empirical_coverage = ((pred.numpy() >= prediction_sets[0].numpy()) & (pred.numpy() <= prediction_sets[1].numpy())).mean()
#     print(f"The empirical coverage after calibration is: {empirical_coverage}")
#     return prediction_sets


# ncal = 150
# alpha = 0.1
# preds, targets = main()
# preds = torch.vstack(preds)
# targets = torch.vstack(targets)

# # %% 
# idx = np.random.randint(270, ncal)
# calibration_data = [preds]
# qhat = calibrate(preds[:ncal], targets[:ncal], 0.5)
# pred_sets = predict_coverage(preds[ncal:], qhat)

# # %% 

# #Slicing along X-Axis

# idx = 10
# var = 2
# y_pos = 20
# time = 10 
# x_grid = np.linspace(0, 1, 248)


# plt.figure()
# plt.plot(x_grid, preds[ncal:][idx, var, :, y_pos, time], label='Pred.', alpha=0.8,  color = 'firebrick')
# plt.plot(x_grid, pred_sets[0][idx, var, :, y_pos, time], label='Lower', alpha=0.8,  color = 'teal', ls='--')
# plt.plot(x_grid, pred_sets[1][idx, var,:, y_pos, time], label='Upper', alpha=0.8,  color = 'navy', ls='--')
# plt.plot(x_grid, targets[ncal:][idx, var,:, y_pos, time], label='Actual', alpha=0.8,  color = 'black')
# plt.legend()
# plt.xlabel('X')
# plt.ylabel('var')
# plt.title('Coverage at alpha = ' + str(alpha))
# plt.grid() #Comment out if you dont want grids.


# # %%
# #Estimating Coverage at all Alpha values 

# for ii in range()
# alpha_levels = np.arange(0.05, 0.95, 0.1)
# emp_cov = []
# for ii in tqdm(range(len(alpha_levels))):
#     emp_cov.append(calibrate(alpha_levels[ii]))
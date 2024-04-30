import pyvista as pv
import streamlit as st
import torch
from stpyvista import stpyvista

from models import FreePointTransformer, HeatRegNet


HEADLESS = True
st.set_page_config(layout="wide")

if HEADLESS:    
    pv.start_xvfb()

st.title("HeatReg")

##load sample data

##load all
data = torch.load('sample_data.pth')
fix = data['fix']
init = data['init']
cpos = data['cpos']
#fpt = data['fpt']

#load models
#create models and load weights
fpt = FreePointTransformer()
fpt.load_state_dict(torch.load('fpt_model.pth'))
fpt.eval()

hrn = HeatRegNet(k=64,stride=4,base=24)
hrn.load_state_dict(torch.load('hrn_model.pth'))
hrn.eval()

def homogenous(kpts):
    B, N, _ = kpts.shape
    device = kpts.device
    return torch.cat([kpts, torch.ones(B, N, 1, device=device)], dim=2)


def transform_points(T, kpts):
    return (T @ homogenous(kpts).permute(0, 2, 1)).permute(0, 2, 1)[:, :, :3].contiguous()


##run models
def predict(model, fix, mov):
    with torch.no_grad():
        disp = model(mov.view(1,-1,3), fix.view(1,-1,3)).view(1, 64, -1, 3)
        pred = mov + disp

        T = torch.eye(4).unsqueeze(0).repeat(64, 1, 1)
        T[:, :3, 3] = (pred.view(64, -1, 3)-mov.view(64, -1, 3)).mean(1)
        return transform_points(T, mov).view(64, 32, 3)
    
               
def build_plotter(fpt_result = None, hrn_result = None):
    plotter = pv.Plotter(shape=(1, 3), window_size=(1200, 400))
    if cpos is not None:
        plotter.camera_position = cpos.numpy()
    plotter.subplot(0, 0)
    plotter.add_points(fix.view(-1,3).numpy(), color='blue', point_size=3)
    plotter.add_points(init.view(-1,3).numpy(), color='red', point_size=3)
    plotter.add_title("Initial")


    
    plotter.subplot(0, 1)
    plotter.add_points(fix.view(-1,3).numpy(), color='red', point_size=3)
    if fpt_result is not None:
        plotter.add_points(fpt_result.view(-1,3).numpy(), color='blue', point_size=3)
    plotter.add_title("FPT")


    plotter.subplot(0, 2)
    plotter.add_points(fix.view(-1,3).numpy(), color='red', point_size=3)
    if hrn_result is not None:
        plotter.add_points(hrn_result.view(-1,3).numpy(), color='blue', point_size=3)
    plotter.add_title("Heat Reg")

    plotter.view_isometric()
    plotter.background_color = 'white'
    plotter.link_views()
    return plotter


fpt_result = None
hrn_result = None

# Display initial plot and button
plot_spot = st.empty()
text_spot = st.empty()

plotter = build_plotter(fpt_result, hrn_result)
button_inference = st.button("Run Inference?")

with plot_spot:
    stpyvista(plotter)
if button_inference:
    text_spot.text("Predciting ...")
    fpt_result = predict(fpt, fix, init)
    hrn_result = predict(hrn, fix, fpt_result)
    text_spot.text("Inference Complete")

    plotter = build_plotter(fpt_result, hrn_result)
    with plot_spot:
        stpyvista(plotter)


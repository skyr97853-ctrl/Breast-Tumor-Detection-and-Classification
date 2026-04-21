from model import S1,S2
import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
S1.main_process(img_folder="../testB/cla",output_folder="./",device=device)
S2.main_process(img_folder="../testB/fea",output_folder="./",device=device)


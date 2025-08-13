import glob
from tqdm import tqdm
import cv2
import numpy as np
import romp
import json
import requests
import time
import os




# Define the pointing pose for the right hand (SMPL-X / MANO hand model)
pointing_rhand_pose = [
    0.2, -0.2, 0.0,     # Global hand rotation 
    0.3, -0.1, 0.0,     # Another global parameter 
    0.1, 0.0, 0.0,      # Thumb base 
    0.0, 0.0, 0.0,      # Thumb mid 
    0.0, 0.0, 0.0,      # Thumb tip 
    0.0, 0.0, 0.0,      # Index base - straightened 
    0.1, 0.0, 0.0,      # Index mid - straightened 
    0.1, 0.0, 0.0,      # Index tip - straightened 
    0.6, 0.0, 0.0,      # Middle base - curled 
    0.6, 0.0, 0.0,      # Middle mid - curled 
    0.6, 0.0, 0.0,      # Middle tip - curled 
    0.6, 0.0, 0.0,      # Ring base - curled 
    0.6, 0.0, 0.0,      # Ring mid - curled 
    0.6, 0.0, 0.0,      # Ring tip - curled 
    0.6, 0.0, 0.0       # Pinky (all joints) - curled
]

pointing_lhand_pose = [
    -0.2, -0.2, 0.0,    # Global hand rotation 
    -0.3, -0.1, 0.0,    # Another global parameter 
    -0.1, 0.0, 0.0,     # Thumb base 
     0.0, 0.0, 0.0,     # Thumb mid 
     0.0, 0.0, 0.0,     # Thumb tip 
     0.0, 0.0, 0.0,     # Index base - straightened 
    -0.1, 0.0, 0.0,     # Index mid - straightened 
    -0.1, 0.0, 0.0,     # Index tip - straightened 
    -0.6, 0.0, 0.0,     # Middle base - curled 
    -0.6, 0.0, 0.0,     # Middle mid - curled 
    -0.6, 0.0, 0.0,     # Middle tip - curled 
    -0.6, 0.0, 0.0,     # Ring base - curled 
    -0.6, 0.0, 0.0,     # Ring mid - curled 
    -0.6, 0.0, 0.0,     # Ring tip - curled 
    -0.6, 0.0, 0.0      # Pinky (all joints) - curled 
]





if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    args = parser.parse_args()

    settings = romp.main.default_settings 
    settings.mode = "video"
    settings.t = True
    settings.sc = 1
    settings.show_largest = True
    #settings.calc_smpl = True
    #settings.render_mesh = True
    #settings.show        = True
    #settings.save_video        = True
    #settings.o = './sam_VR/save.mp4'
    
    romp_model = romp.ROMP(settings)

    results = []
    timestep = 0
    url = "http://jorjinapp.ddns.net:16385/set_pose"


    for p in tqdm(sorted(glob.glob(f"{args.data_dir}/images/*"))):
        #timestep = int(time.time()*1000)
        #print(p)
        img = cv2.imread(p)
        result = romp_model(img)
        if result["body_pose"].shape[0] > 1:
            result = {k: v[0:1] for k, v in result.items()}
        #print(result["smpl_thetas"])
        list = []
        #wait = 0.0
        #for i, item in enumerate(result["smpl_thetas"][0]):
            #print(i)
            #if i % 3 == 0:
                #result["smpl_thetas"][0][i] = item*(-1.0)
        #result["smpl_thetas"][0][0] = result["smpl_thetas"][0][0]*(-1.0)
            
            
        #print(result["smpl_thetas"])
        json_save = {
                "body_pose": np.squeeze(result["smpl_thetas"][:, 3:]).tolist(),
                "lhand_pose": pointing_lhand_pose,
                "rhand_pose": pointing_rhand_pose,
                "root_pose": np.squeeze(result["smpl_thetas"][:, :3]).tolist(),
                "transl": np.squeeze(result["cam_trans"]).tolist(),
        }
        #print(json_save)
        #response = requests.post(url, json=json_save)
        #print(response.status_code)  # 输出响应状态码
        #print(response.json()) 





        output_directory = f"{args.data_dir}/smplx_params"
        os.makedirs(output_directory, exist_ok=True)
        #print(json_save)
        with open(f"{output_directory}/smplx_param_{timestep}.json", "w") as f:
            json.dump(json_save, f)
        #results.append(result)
        
        timestep=timestep+1
        
       

        
        


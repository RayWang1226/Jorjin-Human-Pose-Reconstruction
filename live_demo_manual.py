import glob
from tqdm import tqdm
import cv2
import numpy as np
import romp
import json
import requests
import os
import time
import mediapipe as mp

########################################################################
# consecutive frames modifided in args (default 5)
# the maximum frames to keep a pose modified in code (default 60)
########################################################################

####################2025.06.18 test (tun-chuan adding)#########################
MAX_FRAMES_KEEP_POSE = 60 #original not define
consecutive_frames = 2 # original use args.consecutive_frames in line 305 but there's no args in live_demo.py
####################2025.06.18 test (tun-chuan adding)#########################

mp_hands = mp.solutions.hands

default_lhand_pose = [0.0] * 45  # 15 joints * 3
default_rhand_pose = [0.0] * 45

# Defining the poses of hand pose

# fist pose
fist_rhand_pose = [0.5, 0.2, 0.0,  0.5, 0.0, 0.0,  0.4, 0.0, 0.0,
                   0.6, 0.0, 0.0,  0.7, 0.0, 0.0,  0.6, 0.0, 0.0,
                   0.6, 0.0, 0.0,  0.7, 0.0, 0.0,  0.6, 0.0, 0.0,
                   0.6, 0.0, 0.0,  0.7, 0.0, 0.0,  0.6, 0.0, 0.0,
                   0.6, 0.0, 0.0,  0.7, 0.0, 0.0,  0.6, 0.0, 0.0]

fist_lhand_pose = [0.5, 0.2, 0.0,  0.5, 0.0, 0.0,  0.4, 0.0, 0.0,
                   0.6, 0.0, 0.0,  0.7, 0.0, 0.0,  0.6, 0.0, 0.0,
                   0.6, 0.0, 0.0,  0.7, 0.0, 0.0,  0.6, 0.0, 0.0,
                   0.6, 0.0, 0.0,  0.7, 0.0, 0.0,  0.6, 0.0, 0.0,
                   0.6, 0.0, 0.0,  0.7, 0.0, 0.0,  0.6, 0.0, 0.0]

# thumbs up
thumbs_up_rhand_pose = [-0.2176254391670227, -0.20206162333488464, 0.596537709236145, 0.24576875567436218, 0.008928668685257435, 0.7652382254600525, -0.07576794922351837, 0.18717364966869354, -0.04110009968280792, -0.25619247555732727, -0.053401097655296326, 1.0122601985931396, 0.06561538577079773, -0.018982456997036934, -0.04032716527581215, -0.018952196463942528, 0.007533820811659098, 0.14132338762283325, -0.11527887731790543, 0.34702953696250916, 0.5196197032928467, -0.25747576355934143,
                        0.01998116448521614, 0.09303396940231323, -0.08838392049074173, 0.028444340452551842, 0.22613437473773956, 0.018786050379276276, 0.25518250465393066, 0.8800680637359619, -0.2179374247789383, 0.17202991247177124, -0.2006685435771942, -0.09984763711690903, -0.17857205867767334, 0.13039840757846832, -0.39125752449035645, 0.08660601824522018, -0.35248780250549316, -0.15806075930595398, 0.39410585165023804, 0.465411901473999, -0.07559788227081299, 0.08000562340021133, -0.6501806378364563]

thumbs_up_lhand_pose = [-0.20237988233566284, 0.21614666283130646, -0.5048045516014099, 0.21577148139476776, -0.013473067432641983, -0.8157778382301331, -0.03625032678246498, -0.18653081357479095, -0.09216436743736267, -0.22911332547664642, 0.07206475734710693, -0.9018931984901428, 0.03459455817937851, -0.004378796089440584, -0.09727431833744049, -0.03970647230744362, -0.004228339996188879, -0.259234219789505, -0.07425858825445175, -0.33180299401283264, -0.4755752980709076, -0.36754751205444336, -
                        0.07359879463911057, -0.19419467449188232, -0.13754071295261383, -0.05141821131110191, -0.3109399378299713, 0.022457299754023552, -0.22714029252529144, -0.7782456874847412, -0.2456602305173874, -0.20032848417758942, 0.043681226670742035, -0.11625704914331436, 0.18079331517219543, -0.22519609332084656, -0.45792171359062195, -0.14335766434669495, 0.400264173746109, -0.18607310950756073, -0.43417471647262573, -0.5101174712181091, -0.14430098235607147, -0.10638222843408585, 0.7470539808273315]

# pointing
pointing_rhand_pose = [-0.019536109641194344, 0.08942107111215591, -0.3350767493247986, -0.13421615958213806, -0.1841834932565689, -0.8647715449333191, 0.13019977509975433, -0.011293350718915462, -0.21379904448986053, -0.4562130570411682, -0.27199816703796387, 0.9983204007148743, 0.10484061390161514, -0.0795687660574913, 0.5100291967391968, 0.059318214654922485, 0.08810481429100037, 0.31801915168762207, -0.7394726872444153, 0.0894777923822403, 0.8573942184448242, 0.0004250608617439866,
                       0.09527215361595154, 0.4302603304386139, -0.4869771897792816, 0.0516945943236351, 0.5615432858467102, -0.24495187401771545, 0.1415485441684723, 1.1551563739776611, -0.19214919209480286, 0.13161221146583557, 0.1773107796907425, -0.12060423195362091, -0.008107597008347511, 0.5516828894615173, 0.5622410774230957, 0.000956787436734885, 0.11487039923667908, -0.09988228231668472, 0.23435211181640625, -0.022748630493879318, 0.44819673895835876, -0.641718864440918, 0.4941408932209015]

pointing_lhand_pose = [0.08192285895347595, -0.07774589955806732, 0.3979912996292114, -0.17585274577140808, 0.05465409532189369, 0.6490399837493896, 0.10111227631568909, 0.058073997497558594, 0.18288350105285645, -0.2947438657283783, 0.147257000207901, -0.3982015550136566, 0.07402446120977402, 0.12355317920446396, -0.3760266602039337, 0.045483823865652084, -0.06098753958940506, -0.1114245057106018, -0.675298273563385, -0.11373588442802429, -0.5808815360069275, 0.0688437968492508, -
                       0.020855041220784187, -0.37976792454719543, -0.33500564098358154, 0.009226588532328606, -0.2589498460292816, -0.3392491042613983, -0.04721532389521599, -0.8640543818473816, -0.12315529584884644, -0.05412845313549042, -0.5659985542297363, -0.029332229867577553, -0.08689531683921814, -0.27055466175079346, 0.5847598910331726, 0.27790597081184387, -0.16981084644794464, -0.06191617250442505, 0.1087554395198822, 0.18387795984745026, 0.26813971996307373, 0.2519071400165558, -0.37869328260421753]

# "yeah" pose
yeah_rhand_pose = [0.14370101690292358, 0.13935032486915588, -0.539125919342041, -0.21937736868858337, 0.04736620932817459, -0.6393141746520996, 0.090763621032238, -0.12260670959949493, -0.1367853283882141, -0.0486672967672348, 0.04141095280647278, -0.6072007417678833, 0.13461732864379883, -0.03548883646726608, -0.6686115860939026, 0.01061110757291317, -0.019600635394454002, -0.2826547920703888, -0.5101594924926758, 0.2434517741203308, 0.5309790372848511, -0.08845118433237076, -
                   0.03367839753627777, 0.17720705270767212, 0.03402087464928627, -0.01901707425713539, 0.05467155948281288, -0.3700631856918335, -0.035837121307849884, 0.4202747344970703, 0.015010653994977474, -0.007807051297277212, 0.6514725089073181, 0.041440967470407486, 0.05859791114926338, -0.1214468702673912, 0.5679723024368286, -0.11591850966215134, 0.2956843674182892, 0.011455491185188293, -0.1655249297618866, -0.25311940908432007, 0.3393993675708771, -0.1281975656747818, 0.537899911403656]

yeah_lhand_pose = [0.2126484513282776, -0.1744786500930786, 0.6087465286254883, -0.2561101019382477, -0.054700955748558044, 0.6121962666511536, 0.06156224012374878, 0.0860682874917984, 0.11825046688318253, -0.10162236541509628, 0.03667672351002693, 0.5392575263977051, 0.1703561693429947, 0.08813285082578659, 0.6903200745582581, 0.04668775573372841, 0.004406738094985485, 0.24956662952899933, -0.7736870646476746, -0.1387835144996643, -0.7480520606040955, 0.22430677711963654,
                   0.041760411113500595, -0.17951025068759918, -0.08615124970674515, 0.1248009130358696, 0.0884634405374527, -0.426632285118103, 0.06994043290615082, -0.6475301384925842, 0.046697601675987244, 0.08614356070756912, -0.5241201519966125, 0.06250791251659393, -0.07866089046001434, 0.1457505226135254, 0.7178047299385071, 0.24975372850894928, -0.37111127376556396, -0.08465445041656494, 0.10537915676832199, 0.3854857385158539, 0.3102468252182007, 0.0034858440048992634, -0.5718988180160522]

# "ok" pose
ok_rhand_pose = [-0.09884610772132874, -0.07384993880987167, 0.19680000841617584, 0.14215871691703796, 0.06148999184370041, -0.03411637991666794, -0.08444081246852875, 0.00723862461745739, -0.33615943789482117, 0.04003579914569855, 0.07928559929132462, 0.13087327778339386, 0.07098275423049927, -0.005053278990089893, -0.5805941224098206, 0.01284438744187355, -0.022224219515919685, -0.39410245418548584, 0.6980700492858887, 0.01650737039744854, -0.5984206795692444, -0.03623383864760399, -
                 0.15471768379211426, -0.33799511194229126, 0.26595765352249146, -0.002640832681208849, -0.2116403728723526, 0.2777584195137024, 0.06263039261102676, -0.2517591118812561, 0.1181454285979271, -0.022409051656723022, -0.5900528430938721, 0.03375769034028053, -0.07401784509420395, -0.32074183225631714, 0.01701347902417183, 0.03608861565589905, 0.09535131603479385, 0.12836617231369019, -0.0839342251420021, -0.09191214293241501, 0.137675940990448, -0.03913756459951401, 0.16169384121894836]

ok_lhand_pose = [-0.11209169030189514, 0.08424270898103714, -0.2655673921108246, 0.18535956740379333, -0.0810956284403801, -0.06755317002534866, -0.10054394602775574, -0.03151743486523628, 0.330910325050354, -0.010637257248163223, -0.09496603906154633, -0.21600979566574097, 0.06704366952180862, -0.0029057583305984735, 0.4903869032859802, 0.008822930045425892, 0.019246729090809822, 0.37131941318511963, 0.674208402633667, -0.05102011188864708, 0.5424671769142151, -0.11450748890638351,
                 0.13121992349624634, 0.2761227786540985, 0.285659521818161, -0.014749038964509964, 0.19357308745384216, 0.2705777883529663, -0.10804606229066849, 0.1840253323316574, 0.08143547922372818, 0.021042512729763985, 0.5369747281074524, 0.03060474433004856, 0.0803176686167717, 0.30534204840660095, 0.02446223609149456, 0.009992440231144428, -0.07094962894916534, 0.12478002905845642, 0.10593295842409134, 0.07185636460781097, 0.12531860172748566, 0.04753738269209862, -0.19221249222755432]


def calculate_finger_angles(landmarks):
    """Calculate angles between finger segments to determine pose features"""
    angles = {}

    # Calculate angle between three points
    def calculate_angle(p1, p2, p3):
        v1 = np.array([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z])
        v2 = np.array([p3.x - p2.x, p3.y - p2.y, p3.z - p2.z])

        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)

        if v1_norm == 0 or v2_norm == 0:
            return 0

        dot_product = np.dot(v1, v2)
        angle = np.arccos(
            np.clip(dot_product / (v1_norm * v2_norm), -1.0, 1.0))
        return np.degrees(angle)

    lm = landmarks.landmark

    # Thumb angle
    angles['thumb_fold'] = calculate_angle(lm[2], lm[3], lm[4])

    # Index finger angles
    angles['index_fold'] = calculate_angle(lm[5], lm[6], lm[8])

    # Middle finger angles
    angles['middle_fold'] = calculate_angle(lm[9], lm[10], lm[12])

    # Ring finger angles
    angles['ring_fold'] = calculate_angle(lm[13], lm[14], lm[16])

    # Pinky finger angles
    angles['pinky_fold'] = calculate_angle(lm[17], lm[18], lm[20])

    # Distance between thumb and index tips
    thumb_tip = np.array([lm[4].x, lm[4].y, lm[4].z])
    index_tip = np.array([lm[8].x, lm[8].y, lm[8].z])
    angles['thumb_index_dist'] = np.linalg.norm(thumb_tip - index_tip)

    return angles


def is_thumbs_up(hand_landmarks, handedness):
    """Check if the hand pose is a thumbs up"""
    lm = hand_landmarks.landmark
    angles = calculate_finger_angles(hand_landmarks)

    # Thumb extended, other fingers closed
    thumb_up = lm[4].y < lm[3].y if handedness == "Right" else lm[4].y > lm[3].y

    # Check if other fingers are folded
    fingers_folded = (
        angles['index_fold'] < 120 and
        angles['middle_fold'] < 120 and
        angles['ring_fold'] < 120 and
        angles['pinky_fold'] < 120
    )

    return thumb_up and fingers_folded


def is_pointing(hand_landmarks, handedness):
    """Check if the hand pose is pointing (index finger extended, others folded)"""
    lm = hand_landmarks.landmark
    angles = calculate_finger_angles(hand_landmarks)

    # Index extended
    index_extended = angles['index_fold'] > 160

    # Check if other fingers are folded
    other_fingers_folded = (
        angles['middle_fold'] < 120 and
        angles['ring_fold'] < 120 and
        angles['pinky_fold'] < 120
    )

    # For pointing, we don't care as much about the thumb
    return index_extended and other_fingers_folded


def is_yeah_pose(hand_landmarks, handedness):
    """Check if the hand pose is a 'peace' or 'yeah' sign (index and middle extended)"""
    lm = hand_landmarks.landmark
    angles = calculate_finger_angles(hand_landmarks)

    # Index and middle extended
    index_extended = angles['index_fold'] > 150
    middle_extended = angles['middle_fold'] > 150

    # Check if other fingers are folded
    other_fingers_folded = (
        angles['ring_fold'] < 120 and
        angles['pinky_fold'] < 120
    )

    return index_extended and middle_extended and other_fingers_folded


def is_ok_pose(hand_landmarks, handedness):
    """Check if the hand pose is an 'OK' sign (thumb and index form a circle)"""
    lm = hand_landmarks.landmark
    angles = calculate_finger_angles(hand_landmarks)

    # Thumb and index tips are close to each other
    thumb_index_close = angles['thumb_index_dist'] < 0.1

    # Other fingers mostly extended
    other_fingers_extended = (
        angles['middle_fold'] > 140 and
        angles['ring_fold'] > 140 and
        angles['pinky_fold'] > 140
    )

    return thumb_index_close and other_fingers_extended


def is_fist(hand_landmarks, handedness):
    """Check if the hand is forming a closed fist (all fingers folded)"""
    angles = calculate_finger_angles(hand_landmarks)

    all_fingers_folded = (
        angles['thumb_fold'] < 120 and
        angles['index_fold'] < 120 and
        angles['middle_fold'] < 120 and
        angles['ring_fold'] < 120 and
        angles['pinky_fold'] < 120
    )

    return all_fingers_folded


def detect_hand_pose(hand_landmarks, handedness):
    """Determine which pose the hand is making"""
    if is_thumbs_up(hand_landmarks, handedness):
        return "thumbs_up"
    elif is_pointing(hand_landmarks, handedness):
        return "pointing"
    elif is_yeah_pose(hand_landmarks, handedness):
        return "yeah"
    elif is_ok_pose(hand_landmarks, handedness):
        return "ok"
    elif is_fist(hand_landmarks, handedness):
        return "fist"
    else:
        return "unknown"

if __name__ == "__main__":

    settings = romp.main.default_settings 
    settings.mode = 'webcam'
    settings.t = True
    settings.sc = 1
    settings.show_largest = True
    #settings.calc_smpl = True
    settings.render_mesh = True
    #settings.show        = True
    romp_model = romp.ROMP(settings)

    results = []
    timestep = 0
    
    # Initialize MediaPipe hands
    hands = mp_hands.Hands(
        model_complexity=1,  # 1 is more accurate than 0
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Initialize tracking dictionaries for both hands
    hand_pose_tracking = {
        "Right": {
            "current_pose": "unknown",
            "confirmed_pose": "unknown",
            "consecutive_count": 0,
            "frames_since_detection": 0,  # New field to track absence
            "persistent_pose": "unknown"   # New field for the persistent pose
        },
        "Left": {
            "current_pose": "unknown",
            "confirmed_pose": "unknown",
            "consecutive_count": 0,
            "frames_since_detection": 0,  # New field to track absence
            "persistent_pose": "unknown"   # New field for the persistent pose
        }
    }
    
    # Default poses to start with
    current_rhand_pose = default_rhand_pose.copy()
    current_lhand_pose = default_lhand_pose.copy()
    
    url = "http://jorjinapp.ddns.net:16385/set_pose"
    cap = romp.utils.WebcamVideoStream(2)
    cap.start()
    pose_map = {
        ord('1'): ('fist', fist_lhand_pose, fist_rhand_pose),
        ord('2'): ('thumbs_up', thumbs_up_lhand_pose, thumbs_up_rhand_pose),
        ord('3'): ('pointing', pointing_lhand_pose, pointing_rhand_pose),
        ord('4'): ('yeah', yeah_lhand_pose, yeah_rhand_pose),
        ord('5'): ('ok', ok_lhand_pose, ok_rhand_pose),
        ord('0'): ('default', default_lhand_pose, default_rhand_pose)
    }

    print("Press 1: Fist | 2: Thumbs Up | 3: Pointing | 4: Yeah | 5: OK | 0: Default | q: Quit")

    while True:
        frame = cap.read()
        result = romp_model(frame)
        if result is None:
            continue

        cv2.imshow("ROMP Live Demo", frame)  # ← required for keyboard input
        key = cv2.waitKey(1) & 0xFF

        if key != 255:
            ch = chr(key).lower()
            if ch in pose_map:
                name, lpose, rpose = pose_map[ch]
                current_lhand_pose = lpose.copy()
                current_rhand_pose = rpose.copy()
                print(f"[MANUAL] Hand pose set to '{name}'.")
            elif ch == 'q':
                print("[EXIT] Quitting live demo.")
                break

        if key in pose_map:
            pose_name, current_lhand_pose, current_rhand_pose = pose_map[key]
            print(f"[INFO] Manually set hand pose to: {pose_name}")

        elif key == ord('q'):
            print("[INFO] Exiting manual hand pose control.")
            break

        # Run ROMP again for SMPLX export
        result = romp_model(frame)
        if result["body_pose"].shape[0] > 0:
            if result["body_pose"].shape[0] > 1:
                result = {k: v[0:1] for k, v in result.items()}

            smplx_data = {
                "timestamp": int(time.time() * 1000),
                "pose": {
                    "body_pose": np.squeeze(result["smpl_thetas"][:, 3:]).tolist(),
                    "lhand_pose": current_lhand_pose,
                    "rhand_pose": current_rhand_pose,
                    "root_pose": np.squeeze(result["smpl_thetas"][:, :3]).tolist(),
                    "transl": np.squeeze(result["cam_trans"]).tolist(),
                }
            }

            response = requests.post(url, json=smplx_data)
            print(f"POST status: {response.status_code}")

            output_directory = f"smplx_params"
            os.makedirs(output_directory, exist_ok=True)
            with open(f"{output_directory}/smplx_param_{timestep}.json", "w") as f:
                json.dump(smplx_data, f)
            timestep += 1

    cap.stop()



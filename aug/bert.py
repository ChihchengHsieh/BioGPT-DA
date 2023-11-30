import os
import pandas as pd
import re
from tqdm import tqdm


def get_diagnosis(data, label_cols):
    diagnosis = [k for k, v in dict(data[label_cols] > 0).items() if v > 0]
    if len(diagnosis) == 0:
        return " No lesion found"

    diagnosis_str = ""
    for l in diagnosis:
        diagnosis_str += f" {l},"

    return diagnosis_str[:-2]


def get_prompt_for_mask(
    mimic_eye_path,
    data,
    label_cols,
    feature_to_name,
    desired_clinical_feature,
    report_format=False,
):
    # reflacx_id = data['id']
    patient_id = data["subject_id"]
    study_id = data["study_id"]
    # dicom_id = data['dicom_id']
    report_path = os.path.join(
        mimic_eye_path,
        f"patient_{patient_id}",
        "CXR-DICOM",
        f"s{study_id}.txt",
    )
    with open(report_path) as f:
        report = f.read()

    report = report.strip().replace("FINAL REPORT\n", "").replace("\n", "").strip()

    age = data["age"]
    gender = "Female" if data["gender"] == "F" else "Male"
    if report_format:
        return (
            re.sub(
                "[^0-9A-Za-z.\s\:']",
                "",
                f"{report} LESIONS:{get_diagnosis(data, label_cols)}. AGE: {age}. GENDER: {gender}. {feature_to_name[desired_clinical_feature]}:",
            )
            + " [MASK]"
        )
    else:
        # return f"A {age} years old {gender} patient diagnosed with{get_diagnosis(data, label_cols)}. And, This patient has the radiology report: \n{report}\nThis patients is most likely to have {feature_to_name[desired_clinical_feature]} around"
        # return f"A {age} years old {gender} patient diagnosed with{get_diagnosis(data, label_cols)}. And, This patient has the radiology report: \n{report}\nThe {feature_to_name[desired_clinical_feature]} of this patient is around".replace("_", "")
        return (
            re.sub(
                "[^0-9A-Za-z.\s\:']",
                "",
                f"A {age} years old {gender} patient diagnosed with{get_diagnosis(data, label_cols)}. And, This patient has the radiology report: \n{report}\nThe {feature_to_name[desired_clinical_feature]} of this patient is around",
            )
            + " [MASK]"
        )

def get_generated_value(mask_filler, prompt, min_max_v, top_k):
    outputs = mask_filler(prompt, top_k=top_k)
    for o in outputs: 
        next_word = o["token_str"]
        # next_num_str =  "".join(filter(str.isnumeric, next_word))
        next_num_str = re.sub("[^0-9.]", "", next_word).replace("..", ".")
        # if next_num_str.count(".") <= 1 and len(next_num_str.replace(".", "")) > 0:
        try:
            v = float(next_num_str)
        except:
            continue
            # check if v in the range
        min_v, max_v = min_max_v
        if v > min_v and v < max_v:
                return v
    return None

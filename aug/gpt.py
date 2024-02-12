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

    return diagnosis_str[:-1]


def get_prompt(
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
        return re.sub(
            "[^0-9A-Za-z.\s\:']",
            "",
            f"{report} LESIONS:{get_diagnosis(data, label_cols)}. AGE: {age}. GENDER: {gender}. {feature_to_name[desired_clinical_feature]}:",
        )
    else:
        # return f"A {age} years old {gender} patient diagnosed with{get_diagnosis(data, label_cols)}. And, This patient has the radiology report: \n{report}\nThis patients is most likely to have {feature_to_name[desired_clinical_feature]} around"
        # return f"A {age} years old {gender} patient diagnosed with{get_diagnosis(data, label_cols)}. And, This patient has the radiology report: \n{report}\nThe {feature_to_name[desired_clinical_feature]} of this patient is around".replace("_", "")
        return re.sub(
            "[^0-9A-Za-z.\s\:']",
            "",
            f"A {age} years old {gender} patient diagnosed with{get_diagnosis(data, label_cols)}. And, This patient has the radiology report: \n{report}\nThe {feature_to_name[desired_clinical_feature]} of this patient is around",
        )


def get_generated_value(generator, prompt, min_max_v, num_return_sequences=1):
    # if num_return_sequences == 1:
    #     do_sample = False
    # else:
    do_sample = True

    outputs = generator(
        prompt,
        #  max_length=1024,
        max_new_tokens=5,
        num_return_sequences=num_return_sequences,
        do_sample=do_sample,
    )
    generated = [o["generated_text"] for o in outputs]
    for g in generated:
        next_word = g[len(prompt) + 1 :].split(" ")[0]
        # next_num_str =  "".join(filter(str.isnumeric, next_word))
        next_num_str = re.sub("[^0-9.]", "", next_word).replace("..", ".")
        if next_num_str.count(".") <= 1 and len(next_num_str.replace(".", "")) > 0:
            v = float(next_num_str)
            # check if v in the range
            min_v, max_v = min_max_v
            if v > min_v and v < max_v:
                return v


def progressive_aug(generator, prompt, min_max_v, progress=[1, 5, 25, 50]):
    for p in progress:
        # print(f"Attempting on progress {p}")
        v = get_generated_value(generator, prompt, min_max_v, num_return_sequences=p)
        if not v is None:
            return v


def aug_df(
    mimic_eye_path,
    label_cols,
    features_to_aug,
    feature_to_name_map,
    df,
    generator,
    progress=[1, 5, 25, 50],
    report_format=False,
):
    aug_feature_range = {f: (df[f].min(), df[f].max()) for f in features_to_aug}

    for f in features_to_aug:
        df[f"aug_{f}"] = None

    for f in features_to_aug:
        print(f"Resolving {f}")
        # aug the instance one by one
        for idx, data in tqdm(df.iterrows(), total=df.shape[0]):
            prompt = get_prompt(
                mimic_eye_path,
                data,
                label_cols,
                feature_to_name_map,
                f,
                report_format=report_format,
            )
            v = progressive_aug(
                generator, prompt, aug_feature_range[f], progress=progress
            )
            # if v is None:
            #     print(
            #         f"Couldn't find value for [{idx}] prompt [{progress[-1]} sequences]: {prompt}"
            #     )
            df.at[idx, f"aug_{f}"] = v

    return df

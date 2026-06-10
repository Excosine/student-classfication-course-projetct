"""数据预处理模块：独热编码 + 标准化"""
import os, warnings, numpy as np, pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

warnings.filterwarnings("ignore")

NOMINAL_COLS = [
    "Marital status", "Application mode", "Course", "Nacionality",
    "Mother's occupation", "Father's occupation", "Daytime/evening attendance",
    "Displaced", "Educational special needs", "Debtor", "Tuition fees up to date",
    "Gender", "Scholarship holder", "International",
]
RANDOM_SEED = 42

BINARY_MAP = {0: "No", 1: "Yes"}
MARITAL_MAP = {1: "Single", 2: "Married", 3: "Widower", 4: "Divorced", 5: "Facto Union", 6: "Legally Separated"}
APPLICATION_MODE_MAP = {1: "1st_phase_general", 2: "Ordinance_612_93", 5: "1st_phase_Azores", 7: "Other_higher_courses", 10: "Ordinance_854_B_99", 15: "International_student", 16: "1st_phase_Madeira", 17: "2nd_phase_general", 18: "3rd_phase_general", 26: "Ordinance_533_A_99_b2", 27: "Ordinance_533_A_99_b3", 39: "Over_23_years", 42: "Transfer", 43: "Change_course", 44: "Tech_diploma_holders", 51: "Change_institution", 53: "Short_cycle_diploma", 57: "Change_institution_international"}
COURSE_MAP = {33: "Biofuel_Tech", 171: "Animation_Design", 8014: "Social_Svc_evening", 9003: "Agronomy", 9070: "Communication_Design", 9085: "Veterinary_Nursing", 9119: "Informatics_Eng", 9130: "Equinculture", 9147: "Management", 9238: "Social_Service", 9254: "Tourism", 9500: "Nursing", 9556: "Oral_Hygiene", 9670: "Advertising_Marketing", 9773: "Journalism_Communication", 9853: "Basic_Education", 9991: "Management_evening"}
NACIONALITY_MAP = {1: "Portuguese", 2: "German", 6: "Spanish", 11: "Italian", 13: "Dutch", 14: "English", 17: "Lithuanian", 21: "Angolan", 22: "Cape_Verdean", 24: "Guinean", 25: "Mozambican", 26: "Santomean", 32: "Turkish", 41: "Brazilian", 62: "Romanian", 100: "Moldova", 101: "Mexican", 103: "Ukrainian", 105: "Russian", 108: "Cuban", 109: "Colombian"}
OCCUPATION_MAP = {0: "Student", 1: "Executives_Directors", 2: "Intellectual_Science", 3: "Intermediate_Technicians", 4: "Administrative", 5: "Personal_Svc_Security_Sellers", 6: "Farmers_Fishery_Forestry", 7: "Industry_Construction_Craftsmen", 8: "Machine_Operators_Assembly", 9: "Unskilled_Workers", 10: "Armed_Forces", 90: "Other", 99: "Blank", 101: "Armed_Forces_Officers", 102: "Armed_Forces_Sergeants", 103: "Other_Armed_Forces", 112: "Admin_Commercial_Directors", 114: "Hotel_Catering_Trade_Directors", 121: "Physical_Science_Math_Eng", 122: "Health_professionals", 123: "Teachers", 124: "Finance_Accounting_Admin", 125: "ICT_specialists", 131: "Science_Eng_Technicians", 132: "Health_Technicians", 134: "Legal_Social_Sports_Cultural_Technicians", 135: "ICT_Technicians", 141: "Office_Secretaries_Data", 143: "Data_Accounting_Financial_Registry", 144: "Other_Admin_Support", 151: "Personal_Service_Workers", 152: "Sellers", 153: "Personal_Care_Workers", 154: "Protection_Security_Svc", 161: "Market_Farmers_Animal_Prod", 163: "Subsistence_Farmers_Fishermen", 171: "Construction_Workers", 172: "Metallurgy_Metalworking", 173: "Printing_Precision_Jewelers_Artisans", 174: "Electricity_Electronics_Workers", 175: "Food_Wood_Clothing_Crafts", 181: "Fixed_Plant_Machine_Operators", 182: "Assembly_Workers", 183: "Vehicle_Drivers_Mobile_Equipment", 191: "Cleaning_Workers", 192: "Unskilled_Agriculture_Fishery_Forestry", 193: "Unskilled_Extractive_Construction_Manufacturing", 194: "Meal_Preparation_Assistants", 195: "Street_Vendors_Svc"}
QUALIFICATION_MAP = {1: "Secondary_Edu", 2: "Bachelor", 3: "Degree", 4: "Master", 5: "Doctorate", 6: "Freq_Higher_Edu", 9: "12th_Not_Completed", 10: "11th_Not_Completed", 11: "7th_Year_Old", 12: "Other_11th", 13: "2nd_Year_Complementary", 14: "10th_Year", 18: "General_Commerce", 19: "Basic_Edu_3rd_Cycle", 20: "Complementary_High_School", 22: "Tech_Professional_Course", 25: "Complementary_High_Not_Completed", 26: "7th_Year_Schooling", 27: "2nd_Cycle_General_High", 29: "9th_Not_Completed", 30: "8th_Year", 31: "General_Admin_Commerce", 33: "Supplementary_Accounting", 34: "Unknown", 35: "Cannot_Read_Write", 36: "Read_No_4th_Year", 37: "Basic_Edu_1st_Cycle", 38: "Basic_Edu_2nd_Cycle", 39: "Tech_Specialization", 40: "Higher_Edu_Degree_1st", 41: "Specialized_Higher", 42: "Professional_Higher_Tech", 43: "Higher_Edu_Master_2nd", 44: "Higher_Edu_Doctorate_3rd"}
COLUMN_MAP_DICT = {"Marital status": MARITAL_MAP, "Application mode": APPLICATION_MODE_MAP, "Course": COURSE_MAP, "Nacionality": NACIONALITY_MAP, "Mother's occupation": OCCUPATION_MAP, "Father's occupation": OCCUPATION_MAP, "Mother's qualification": QUALIFICATION_MAP, "Father's qualification": QUALIFICATION_MAP, "Previous qualification": QUALIFICATION_MAP}
BINARY_COLS = ["Daytime/evening attendance", "Displaced", "Educational special needs", "Debtor", "Tuition fees up to date", "Gender", "Scholarship holder", "International"]


def _rename_oh_columns(columns, original_col, value_dict):
    new_cols = []
    for col in columns:
        parts = col.rsplit("_", 1)
        if len(parts) == 2 and parts[0] == original_col and parts[1].isdigit():
            code = int(parts[1])
            label = value_dict.get(code, f"code{code}")
            new_cols.append(f"{original_col}={label}")
        else:
            new_cols.append(col)
    return new_cols


def _rename_all_columns(columns):
    renamed = list(columns)
    for bcol in BINARY_COLS:
        renamed = _rename_oh_columns(renamed, bcol, BINARY_MAP)
    for orig_col, mapping in COLUMN_MAP_DICT.items():
        renamed = _rename_oh_columns(renamed, orig_col, mapping)
    return renamed


def load_and_preprocess(csv_path=None):
    if csv_path is None:
        for candidate in ["data.csv", "../data.csv"]:
            candidate = os.path.join(os.path.dirname(__file__), candidate)
            if os.path.exists(candidate):
                csv_path = candidate
                break
    if csv_path is None or not os.path.exists(csv_path):
        raise FileNotFoundError("找不到 data.csv")

    print(f"[预处理] 读取: {csv_path}")
    df = pd.read_csv(csv_path, sep=";")
    df.columns = df.columns.str.strip().str.replace('"', "")

    target_col = next((c for c in df.columns if c.lower() in ("target", "status")), None)
    if target_col is None:
        raise KeyError(f"未找到目标列: {list(df.columns)}")
    print(f"[预处理] 目标列: '{target_col}'")

    df = df[df[target_col] != "Enrolled"]
    print(f"[预处理] 剔除 Enrolled 后样本数: {len(df)}")

    le = LabelEncoder()
    y = le.fit_transform(df[target_col])
    print(f"[预处理] 目标编码: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    X = df.drop(columns=[target_col])
    nominal_present = [c for c in NOMINAL_COLS if c in X.columns]
    print(f"[预处理] 独热编码前特征: {X.shape[1]}")
    X = pd.get_dummies(X, columns=nominal_present, drop_first=True)
    X.columns = _rename_all_columns(list(X.columns))
    print(f"[预处理] 独热编码后特征: {X.shape[1]}")

    if X.isnull().sum().sum() > 0:
        for col in X.columns:
            if X[col].isnull().sum() > 0:
                X[col].fillna(X[col].median() if X[col].dtype in [np.float64, np.int64, bool] else X[col].mode()[0], inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)
    print(f"[预处理] 训练集: {X_train.shape[0]}, 测试集: {X_test.shape[0]}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    save_dir = os.path.dirname(__file__) or "."
    np.save(os.path.join(save_dir, "X_train.npy"), X_train_scaled)
    np.save(os.path.join(save_dir, "X_test.npy"), X_test_scaled)
    np.save(os.path.join(save_dir, "y_train.npy"), y_train)
    np.save(os.path.join(save_dir, "y_test.npy"), y_test)
    joblib.dump(scaler, os.path.join(save_dir, "scaler.pkl"))
    joblib.dump(list(X.columns), os.path.join(save_dir, "feature_names.pkl"))

    eda_data = {"X_raw": df.drop(columns=[target_col]),
                "y_counts": df[target_col].value_counts().to_dict(),
                "y_labels": list(le.classes_)}
    return X_train_scaled, X_test_scaled, y_train, y_test, list(X.columns), eda_data


if __name__ == "__main__":
    load_and_preprocess()

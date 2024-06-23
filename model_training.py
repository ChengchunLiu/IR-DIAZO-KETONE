from rdkit import Chem
from rdkit.Chem import AllChem
from joblib import dump, load
from catboost import CatBoostRegressor
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor, VotingRegressor
from xgboost import XGBRegressor
import lightgbm as lgb
from sklearn.linear model import BayesianRidge
import numpy as np
import math
from sklearn.pipeline import make_pipeline


excel_file = r'data.xlsx'
df = pd.read_excel(excel_file)

def process_smiles(smiles_str):
    try:
        mol = Chem.MolFromSmiles(smiles_str)
        mol = Chem.AddHs(mol)

        # Find the positively charged nitrogen atom (N+)
        nplus_atoms = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetFormalCharge() == 1]
        if not nplus_atoms:
            return None, None, None, None
        nplus = mol.GetAtomWithIdx(nplus_atoms[0])

        # Find the carbon atom connected to N+
        carbon_neighbors = [neighbor for neighbor in nplus.GetNeighbors() if neighbor.GetSymbol() == "C"]
        if not carbon_neighbors:
            return "NoCarbonFound", None, None, None
        carbon = carbon_neighbors[0]

        # Find other atoms connected to the carbon, excluding N+
        connected_atoms = [neighbor for neighbor in carbon.GetNeighbors() if neighbor.GetIdx() != nplus.GetIdx()]

        # Priority order for connected atoms
        priority_order = [
            ('Cl', 'SINGLE'), ('S', 'AROMATIC'), ('S', 'SINGLE'), ('F', 'SINGLE'),
            ('O', 'AROMATIC'), ('O', 'DOUBLE'), ('O', 'SINGLE'),
            ('N', 'TRIPLE'), ('N', 'AROMATIC'), ('N', 'DOUBLE'), ('N', 'SINGLE'),
            ('C', 'TRIPLE'), ('C', 'AROMATIC'), ('C', 'DOUBLE'), ('C', 'SINGLE'),
            ('H', 'SINGLE')
        ]

        # Create a list to store atom atomic number and connections
        connections_list = []
        for atom in connected_atoms:
            atomic_num = atom.GetAtomicNum()
            connected_atom_symbol = atom.GetSymbol()
            neighbors_info = []
            for neighbor in atom.GetNeighbors():
                if neighbor.GetIdx() != carbon.GetIdx():
                    neighbor_symbol = neighbor.GetSymbol()
                    bond = mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
                    bond_type_str = str(bond.GetBondType())
                    neighbors_info.append((neighbor_symbol, bond_type_str))

            connections_dict = {'atomic_num': atomic_num, 'connections': {connected_atom_symbol: neighbors_info}}
            connections_list.append(connections_dict)

        # Determine R1 and R2 based on atomic number and connection priority
        connections_list.sort(key=lambda x: x['atomic_num'], reverse=True)
        if len(connections_list) > 1 and connections_list[0]['atomic_num'] == connections_list[1]['atomic_num']:
            # If atomic numbers are the same, sort by connection priority
            connections_list.sort(key=lambda x: min(priority_order.index(y) if y in priority_order else len(priority_order)
                                                    for y in [item for sublist in x['connections'].values() for item in sublist]))

        R1, R2 = connections_list[0], connections_list[1] if len(connections_list) > 1 else None
        atomic_number_R1 = R1['atomic_num']
        atomic_number_R2 = R2['atomic_num'] if R2 else None

        return R1, R2, atomic_number_R1, atomic_number_R2

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, None, None

def calculate_morgan_fingerprint(smiles, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    return list(fp)
def Feature_Engineering():

    possible_columns = [
        'Cl_SINGLE_R1', 'Cl_SINGLE_R2', 'S_AROMATIC_R1', 'S_AROMATIC_R2', 'S_SINGLE_R1', 'S_SINGLE_R2',
         'F_SINGLE_R1',  'F_SINGLE_R2', 'O_AROMATIC_R1', 'O_AROMATIC_R2', 'O_DOUBLE_R1', 'O_DOUBLE_R2', 'O_SINGLE_R1', 'O_SINGLE_R2',
        'N_TRIPLE_R2', 'N_TRIPLE_R1', 'N_AROMATIC_R1', 'N_AROMATIC_R2', 'N_DOUBLE_R1', 'N_DOUBLE_R2', 'N_SINGLE_R1', 'N_SINGLE_R2',
         'C_TRIPLE_R1', 'C_TRIPLE_R2', 'C_AROMATIC_R2', 'C_AROMATIC_R1',  'C_DOUBLE_R1',  'C_DOUBLE_R2',  'C_SINGLE_R1', 'C_SINGLE_R2',
          'H_SINGLE_R1', 'H_SINGLE_R2'

    ]

    results = []

    for index, row in df.iterrows():
        smiles_str = row['SMILES']
        R1, R2, atomic_number_R1, atomic_number_R2 = process_smiles(smiles_str)
        # 初始化计数字典
        counts = {'R1' : R1, 'R2' : R2, 'atomic_number_R1': atomic_number_R1, 'atomic_number_R2': atomic_number_R2, 'SMILES': smiles_str, 'N2_IR_Characteristic_Peak': row['N2_IR_Characteristic_Peak'], 'File Name' : row['File Name'], 'name': row['name'], 'Corresponding Author': row['Corresponding Author']}
        counts.update({col: 0 for col in possible_columns})


        if R1 == "NoCarbonFound":
            continue

        fingerprint = calculate_morgan_fingerprint(smiles_str)
        for i, bit in enumerate(fingerprint):
            counts[f'Fingerprint_{i}'] = bit

        for suffix, connections in [('R1', R1), ('R2', R2)]:
            if connections:
                for connection in connections['connections'].values():
                    for bond in connection:
                        bond_type_str = f'{bond[0]}_{bond[1]}'
                        counts[f'{bond_type_str}_{suffix}'] = counts.get(f'{bond_type_str}_{suffix}', 0) + 1

        results.append(counts)


    results_df = pd.DataFrame(results)
    results_df_filled = results_df.fillna(0)
    electronegativity_dict = {
        1: 2.20, 5: 2.04, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98, 14: 1.90, 15: 2.19, 16: 2.58, 17: 3.16, 35: 2.96, 53: 2.66
    }

    # Define the covalent radius dictionary
    covalent_radius_dict = {
        1: 37, 6: 77, 7: 75, 8: 73, 9: 71, 14: 111, 15: 106, 16: 102, 17: 99, 35: 114, 53: 133
    }

    # Add electronegativity and covalent radius columns
    results_df_filled['electronegativity_R1'] = results_df_filled['atomic_number_R1'].map(electronegativity_dict)
    results_df_filled['electronegativity_R2'] = results_df_filled['atomic_number_R2'].map(electronegativity_dict)
    results_df_filled['covalent_radius_R1'] = results_df_filled['atomic_number_R1'].map(covalent_radius_dict)
    results_df_filled['covalent_radius_R2'] = results_df_filled['atomic_number_R2'].map(covalent_radius_dict)

    # Replace NaN values if necessary (optional)
    results_df_filled['electronegativity_R1'].fillna(0, inplace=True)
    results_df_filled['electronegativity_R2'].fillna(0, inplace=True)
    results_df_filled['covalent_radius_R1'].fillna(0, inplace=True)
    results_df_filled['covalent_radius_R2'].fillna(0, inplace=True)

    print("FE Processing complete.")
    return results_df_filled

def train_evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)

    # Predict on training and test sets
    y_train_pred = model.predict(X_train).ravel()
    y_test_pred = model.predict(X_test).ravel()

    # Compute performance metrics
    rmse_train = mean_squared_error(y_train, y_train_pred, squared=False)
    r2_train = r2_score(y_train, y_train_pred)
    rmse_test = mean_squared_error(y_test, y_test_pred, squared=False)
    r2_test = r2_score(y_test, y_test_pred)

    return model, y_train_pred, rmse_train, r2_train, y_test_pred, rmse_test, r2_test



def evaluate_model(y_true, y_pred):
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2
def calculate_similarity(train_smiles, test_smiles):
    """计算训练集和测试集之间的Tanimoto相似度"""
    train_fps = [AllChem.GetMorganFingerprint(Chem.MolFromSmiles(smile), 2) for smile in train_smiles]
    test_fps = [AllChem.GetMorganFingerprint(Chem.MolFromSmiles(smile), 2) for smile in test_smiles]

    similarity_scores = []
    for test_fp in test_fps:
        max_similarity = max(AllChem.DataStructs.TanimotoSimilarity(test_fp, train_fp) for train_fp in train_fps)
        similarity_scores.append(max_similarity)
    return similarity_scores


def add_noise(data, noise_level):
    noise = np.random.normal(0, noise_level * np.std(data), data.shape)
    return data + noise


def Model_construction():
    df = Feature_Engineering()
    y = df['N2_IR_Characteristic_Peak'].values - 2104

    feature = ['electronegativity_R1', 'electronegativity_R2', 'covalent_radius_R1', 'covalent_radius_R2',
               'Cl_SINGLE_R1', 'Cl_SINGLE_R2',
               'S_AROMATIC_R1', 'S_AROMATIC_R2', 'S_SINGLE_R1', 'S_SINGLE_R2',
               'F_SINGLE_R1', 'F_SINGLE_R2', 'O_AROMATIC_R1', 'O_AROMATIC_R2', 'O_DOUBLE_R1', 'O_DOUBLE_R2',
               'O_SINGLE_R1', 'O_SINGLE_R2',
               'N_TRIPLE_R2', 'N_TRIPLE_R1', 'N_AROMATIC_R1', 'N_AROMATIC_R2', 'N_DOUBLE_R1', 'N_DOUBLE_R2',
               'N_SINGLE_R1', 'N_SINGLE_R2',
               'C_TRIPLE_R1', 'C_TRIPLE_R2', 'C_AROMATIC_R2', 'C_AROMATIC_R1', 'C_DOUBLE_R1', 'C_DOUBLE_R2',
               'C_SINGLE_R1', 'C_SINGLE_R2',
               'H_SINGLE_R1', 'H_SINGLE_R2'
               ]
    Morgan_features = [f'Fingerprint_{i}' for i in range(2048)]

    all_features = feature + Morgan_features
    X = df[all_features].values

    # 分割训练集和测试集
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    scaler = StandardScaler()
    X_train_val = scaler.fit_transform(X_train_val)
    X_test = scaler.transform(X_test)

    base_models = [
        ('rf', RandomForestRegressor(random_state=42)),
        ('gb', GradientBoostingRegressor(random_state=42)),
        ('xgb', XGBRegressor(random_state=42)),
        ('lgbm', lgb.LGBMRegressor(random_state=42)),
        ('CatBoost', CatBoostRegressor(random_state=42))
    ]

    # Define stacked regressor
    stacked_model = StackingRegressor(
        estimators=base_models,
        final_estimator=BayesianRidge()
    )

    models = {
        "Random Forest": RandomForestRegressor(random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        'Bayesian Ridge Regression': BayesianRidge(),
        "XGBoost": XGBRegressor(random_state=42),
        "LightGBM": lgb.LGBMRegressor(random_state=42),
        "CatBoost": CatBoostRegressor(random_state=42),
        "stacked_model": stacked_model
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for model_name, model in models.items():
        print(f'\nTraining {model_name} with 5-Fold Cross Validation...')
        rmse_scores = []
        r2_scores = []

        for train_index, val_index in kf.split(X_train_val):
            X_train, X_val = X_train_val[train_index], X_train_val[val_index]
            y_train, y_val = y_train_val[train_index], y_train_val[val_index]

            model.fit(X_train, y_train)
            y_val_pred = model.predict(X_val)

            rmse = mean_squared_error(y_val, y_val_pred, squared=False)
            r2 = r2_score(y_val, y_val_pred)

            rmse_scores.append(rmse)
            r2_scores.append(r2)

        print(f'{model_name} - Mean RMSE: {np.mean(rmse_scores)}, Mean R2: {np.mean(r2_scores)}')

        # 最终测试集评估
        model.fit(X_train_val, y_train_val)
        y_test_pred = model.predict(X_test)
        rmse_test = mean_squared_error(y_test, y_test_pred, squared=False)
        r2_test = r2_score(y_test, y_test_pred)
        print(f'{model_name} - Test RMSE: {rmse_test}, Test R2: {r2_test}')


#print('Model saving')
    #dump(stacked_model, 'stacked_model.joblib')
    #dump(scaler, 'scaler.joblib')


if __name__ == '__main__':
    Model_construction()


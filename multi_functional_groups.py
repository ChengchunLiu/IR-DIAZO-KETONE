from rdkit import Chem
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from rdkit.Chem import AllChem
import joblib
import numpy as np

def mulit_diazo_process_smiles(smiles_str):
    try:
        mol = Chem.MolFromSmiles(smiles_str)
        mol = Chem.AddHs(mol)

        # Find the custom group atoms (e.g., positively charged nitrogen atom, N+)
        custom_group_atoms = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetFormalCharge() == 1]
        if not custom_group_atoms:
            return []

        results = []

        for atom_idx in custom_group_atoms:
            custom_group = mol.GetAtomWithIdx(atom_idx)

            # Find the carbon atom connected to the custom group
            carbon_neighbors = [neighbor for neighbor in custom_group.GetNeighbors() if neighbor.GetSymbol() == "C"]
            if not carbon_neighbors:
                continue  # Skip if no connected carbon atom is found
            carbon = carbon_neighbors[0]

            # Find other atoms connected to the carbon, excluding the custom group
            connected_atoms = [neighbor for neighbor in carbon.GetNeighbors() if neighbor.GetIdx() != custom_group.GetIdx()]

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

            results.append({'R1': R1, 'R2': R2, 'atomic_number_R1': atomic_number_R1, 'atomic_number_R2': atomic_number_R2})

        return results

    except Exception as e:
        print(f"An error occurred: {e}")
        return []


def mutli_carbonyl_process_smiles(smiles_str):
    try:
        mol = Chem.MolFromSmiles(smiles_str)
        mol = Chem.AddHs(mol)

        # Find the carbonyl group atoms (carbon double-bonded to oxygen)
        carbonyl_atoms = [atom.GetIdx() for atom in mol.GetAtoms()
                          if atom.GetSymbol() == "C" and
                          any(neighbor.GetSymbol() == "O" and mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondType() == Chem.BondType.DOUBLE
                              for neighbor in atom.GetNeighbors())]
        if not carbonyl_atoms:
            print("No carbonyl atoms found.")
            return []

        results = []

        for atom_idx in carbonyl_atoms:
            carbonyl_carbon = mol.GetAtomWithIdx(atom_idx)

            # Find the oxygen atom connected to the carbonyl carbon
            oxygen_neighbors = [neighbor for neighbor in carbonyl_carbon.GetNeighbors()
                                if neighbor.GetSymbol() == "O" and mol.GetBondBetweenAtoms(carbonyl_carbon.GetIdx(), neighbor.GetIdx()).GetBondType() == Chem.BondType.DOUBLE]
            if not oxygen_neighbors:
                continue  # Skip if no connected oxygen atom is found
            oxygen = oxygen_neighbors[0]

            # Find other atoms connected to the carbonyl carbon, excluding the oxygen atom
            connected_atoms = [neighbor for neighbor in carbonyl_carbon.GetNeighbors() if neighbor.GetIdx() != oxygen.GetIdx()]

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
                    if neighbor.GetIdx() != carbonyl_carbon.GetIdx():
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

            results.append({'R1': R1, 'R2': R2, 'atomic_number_R1': atomic_number_R1, 'atomic_number_R2': atomic_number_R2})

        return results

    except Exception as e:
        print(f"An error occurred: {e}")
        return []



def calculate_morgan_fingerprint(smiles, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    return list(fp)

def Feature_Engineering(df):
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
        try:
            custom_groups = mutli_carbonyl_process_smiles(smiles_str)
            if not custom_groups:  # Skip if no custom groups were found
                continue

            for group in custom_groups:
                R1, R2 = group['R1'], group['R2']
                atomic_number_R1, atomic_number_R2 = group['atomic_number_R1'], group['atomic_number_R2']

                # Initialize counts dictionary
                counts = {'R1': R1, 'R2': R2, 'atomic_number_R1': atomic_number_R1, 'atomic_number_R2': atomic_number_R2,
                          'SMILES': smiles_str
                counts.update({col: 0 for col in possible_columns})

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
        except ValueError as e:
            print(f"Error occurred while processing SMILES: {smiles_str}. Skipping this data point.")
            continue

    results_df = pd.DataFrame(results)
    results_df_filled = results_df.fillna(0)
    electronegativity_dict = {
        1: 2.20, 5: 2.04, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98, 14: 1.90, 15: 2.19, 16: 2.58, 17: 3.16, 35: 2.96, 53: 2.66
    }

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


def predict_new_data(model_path, scaler_path, new_data_path, features, target_column):
    try:
        new_data = Feature_Engineering(df)
        print("Total number of rows in the data:", len(new_data))
    except FileNotFoundError:
        print(f"无法找到文件 {new_data_path}")
        return

    # Check if all required features are present
    if not all(feature in new_data.columns for feature in features):
        print("新数据中缺少所需特征。")
        return

    # Check if the target column is present
    if target_column not in new_data.columns:
        print("新数据中缺少目标列。")
        return

    # Adjusting target values
    y_true = new_data[target_column]

    # Selecting features
    X_new = new_data[features]

    try:
        # Load the model
        model = joblib.load(model_path)
        # Load the scaler
        scaler = joblib.load(scaler_path)
    except FileNotFoundError as e:
        print(f"无法找到文件 {e.filename}")
        return

    # Transform the features using the scaler
    X_new_scaled = scaler.transform(X_new)

    # Predict with the model
    predictions = model.predict(X_new_scaled)

    # Add predictions and performance metrics to the original dataframe
    new_data['predictions'] = predictions

    # Ensure the columns exist before selecting them
    output_columns = ['SMILES', 'predictions']
    output_columns = [col for col in output_columns if col in new_data.columns]

    output_data = new_data[output_columns]

    return output_data
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
target_column = 'N2_IR_Characteristic_Peak'

data = {'SMILES': ['CC(C(C)=O)=[N+]=[N-]']}
df = pd.DataFrame(data)
print(df)

model_path = r'Train.joblib'
scaler_path = r'scaler.joblib'


df = predict_new_data(model_path, scaler_path, df, all_features, target_column)
print(df)

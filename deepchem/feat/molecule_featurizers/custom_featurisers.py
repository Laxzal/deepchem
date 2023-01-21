import enum
from rdkit import Chem
from deepchem.feat import WeaveFeaturizer, MolecularFeaturizer, graph_data
from deepchem.feat.mol_graphs import WeaveMol
import numpy as np
from deepchem.feat.graph_data import GraphData
from deepchem.feat.molecule_featurizers.dmpnn_featurizer import DMPNNFeaturizer



class CustomWeaveMol(WeaveMol):
    def __init__(self, nodes, pairs, pair_edges, additional_info):
        super(CustomWeaveMol, self).__init__(nodes, pairs, pair_edges)
        self.additional_info = additional_info
        
    def get_additional_info(self):
        return self.additional_info

class CustomWeaveFeaturizer(WeaveFeaturizer):

    def __init__(self, additional_info):
        super(CustomWeaveFeaturizer, self).__init__()
        self.additional_info = additional_info

    def featurize(self, smi_list):
        mol_list = [Chem.MolFromSmiles(smi) for smi in smi_list]
        
        features = super(CustomWeaveFeaturizer, self).featurize(mol_list)
        for i, mol in enumerate(mol_list):
            smiles = Chem.MolToSmiles(mol,canonical=False)
            
            #I have had to change self.addtional_info[smiles] to self.additional_info[i] because when Chem.MolToSmiles(mol) will reorder integer values 
            #For example
            # smile = c1ccc2c(c1)ccc1ccccc12.[H].[H][H].[c]1cc2ccccc2c2ccccc12
            # mol = Chem.MolFromSmiles(smile)
            # revertSmiles = Chem.MolToSmiles(mol)
            # revertSmiles -->  c1ccc2c(c1)ccc1ccccc21.[H].[H][H].[c]1cc2ccccc2c2ccccc12
            features[i] = CustomWeaveMol(features[i].nodes, features[i].pairs, features[i].pair_edges, self.additional_info[i])
        return features
    

class CustomDMPNNFeaturizer(DMPNNFeaturizer):
    def __init__(self, smi_list, additional_info):
        super(CustomDMPNNFeaturizer, self).__init__()
        self.additional_info = additional_info
        

    def featurize(self, smi_list):
        mol_list = [Chem.MolFromSmiles(smi) for smi in smi_list]

        graph_data = super(CustomDMPNNFeaturizer,self).featurize(mol_list)
        additional_features = np.array([self.additional_info[smi] for smi in smi_list])
        for i, mol in enumerate(graph_data):
            graph_data[i] = GraphData(node_features=mol.node_features,
                                      edge_index= mol.edge_index,
                                      edge_features= mol.edge_features,
                                      global_features = np.append(mol.global_features, additional_features[i]))
        return graph_data
        

 
    
    
additional_info = {'C': [100], 'CC': [200]}
molecules=['C', 'CC']
# custom_DMPNN_featurizer = CustomDMPNNFeaturizer(molecules, 
#                                                 additional_info, 
#                                                 features_generators = ["rdkit_desc_normalized"])


# molecular_features = custom_DMPNN_featurizer.featurize(molecules)
import random
import tensorflow as tf
random.seed(42)
np.random.seed(42)
#tf.set_random_seed(42)
tf.random.set_seed(
    42
)


import pandas as pd


df = pd.read_csv('/home/calvin/Code/DMPNN/Rxn_Smiles.csv')
temp_dict = df['temp'].to_dict()
cusotm_Weave_featurizer = CustomWeaveFeaturizer(additional_info=temp_dict)
molecular_features = cusotm_Weave_featurizer.featurize(df['reaction_smiles'])
print(molecular_features)
import deepchem as dc
dataset = dc.data.DiskDataset.from_numpy(X=molecular_features, y = df['y'].to_numpy())




from deepchem.models import WeaveModel
from deepchem.models.custom_models import EnhancedWeaveModel
batch_size = 38
model = EnhancedWeaveModel(
      1,
      batch_size=batch_size,
      mode='regression',
      fully_connected_layer_sizes=[2000, 1000],
      batch_normalize=False,
      batch_normalize_kwargs={
          "fused": False,
          "trainable": True,
          "renorm": True
      },
      learning_rate=0.0005)
print(model.fit(dataset))

featuriser = WeaveFeaturizer()
mol_feats = featuriser.featurize(df['reaction_smiles'])
dataset_no_t= dc.data.DiskDataset.from_numpy(X=mol_feats, y = df['y'].to_numpy())

batch_size = 38
model_new = WeaveModel(
      1,
      batch_size=batch_size,
      mode='regression',
      fully_connected_layer_sizes=[2000, 1000],
      batch_normalize=False,
      batch_normalize_kwargs={
          "fused": False,
          "trainable": True,
          "renorm": True
      },
      learning_rate=0.0005)
print(model_new.fit(dataset_no_t))
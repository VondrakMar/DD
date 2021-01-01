import matplotlib.pyplot as plt
import pandas as pd
from rdkit import Chem
import numpy as np
from rdkit.Chem import Descriptors
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
# 0.24 -0.78 LogP -0.0059 MW -0.0122 RB -0.43 AP



def AromaticAtoms(m):
  aromatic_atoms = [m.GetAtomWithIdx(i).GetIsAromatic() for i in range(m.GetNumAtoms())]
  aa_count = []
  for i in aromatic_atoms:
    if i==True:
      aa_count.append(1)
  sum_aa_count = sum(aa_count)
  return sum_aa_count


def solub(LogP, MW, RB, AP):
    return 0.24 - 0.78 * LogP -0.0059 * MW -0.0122 * RB -0.43 * AP



f = open("study.txt", "r")
for line in f:
    smile = line.split()
    m = Chem.MolFromSmiles(smile[1])
    curLogP = Descriptors.MolLogP(m)
    curMW = Descriptors.MolWt(m)
    curRB = Descriptors.NumRotatableBonds(m)
    curAP = AromaticAtoms(m)/Descriptors.HeavyAtomCount(m)
    print(smile[0],": ", solub(curLogP, curMW, curRB, curAP))

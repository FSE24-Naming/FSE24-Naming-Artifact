import pickle

with open('././ptm_vectors/vec_d.pkl', 'rb') as f:
    data = pickle.load(f)

print(data)
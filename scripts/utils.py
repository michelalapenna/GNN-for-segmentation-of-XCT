from imports import *

def raw_to_tensor(name, side):
  raw = np.fromfile(name, dtype='B')
  tensor = torch.tensor(raw).reshape(side,side,side)
  return tensor

def tif_to_tensor(name, side):
  im = tifffile.imread(name)
  tensor = torch.tensor(im).reshape(side,side)
  return tensor

def average_labels_vol(name,side,classes):

  occurrences = [[] for _ in range(classes)]

  for j in range(classes):
      occurrences[j].append(name.reshape(side**3).tolist().count(j))

  return occurrences

def create_edges(k_neigh, cloud):
    
    nbrs = NearestNeighbors(n_neighbors=k_neigh).fit(cloud)
    distances, indices = nbrs.kneighbors(cloud)
    pairs = []

    for i in range(len(indices)):
        for j in range(k_neigh):
            if indices[i][0] != indices[i][j]:
                pairs.append(sorted((indices[i][0], indices[i][j])))
    
    tupled_pairs = set(map(tuple,pairs)) 
    unique_pairs = sorted(list(tupled_pairs))

    edges = [[] for _ in range(2)]

    for i in range(len(unique_pairs)):
        edges[0].append(unique_pairs[i][0])
        edges[1].append(unique_pairs[i][1])

    edges = torch.from_numpy(np.array(edges).reshape(2,-1))

    return edges

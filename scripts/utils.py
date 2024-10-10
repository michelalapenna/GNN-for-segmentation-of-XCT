from imports import *

def raw_to_tensor(name, side):
  
  raw = np.fromfile(name, dtype='B')
  tensor = torch.tensor(raw).reshape(side,side,side)

  return tensor

def tif_to_tensor(name, side):
  
  im = tifffile.imread(name)
  tensor = torch.tensor(im).reshape(side,side)

  return tensor

def df_to_csv(tensor, name):
  
  df = pd.DataFrame(tensor.reshape(-1,))
  df.to_csv(name)

def csv_to_tensor(name,dim):
  
  df = pd.read_csv(name)
  df = df.iloc[: , 1:]
  x = pd.DataFrame.to_numpy(df)
  x = x.reshape(dim)
  x = torch.tensor(x)

  return x

def average_labels_vol(name,side):
  
  occurrences = [[] for _ in range(6)]

  for j in range(6):
      occurrences[j].append(name.reshape(side**3).tolist().count(j))

  return occurrences

def fromfile(name):

    raw = np.fromfile(name, dtype='B')

    return raw

def create_edges(k_neigh, cloud):
    
    nbrs = NearestNeighbors(n_neighbors=k_neigh).fit(cloud)
    _, indices = nbrs.kneighbors(cloud)
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

def extract_overlap_pred(eval_obj, test_loader, classes, steps, new_side, side, stride):
    
  preds = eval_obj.eval_function(test_loader)


  all_preds = preds[:,1:].reshape(classes, steps, steps, steps, new_side, new_side, new_side)


  summed_preds = torch.zeros(classes, side,side,side)


  for l in range(classes):
      for i in range(steps):
          for j in range(steps):
              for k in range(steps):
                  summed_preds[l,(i)*stride:(i)*stride+new_side, 
                  (j)*stride:(j)*stride+new_side, 
                  (k)*stride:(k)*stride+new_side] = summed_preds[l,(i)*stride:(i)*stride+new_side, 
                  (j)*stride:(j)*stride+new_side, 
                  (k)*stride:(k)*stride+new_side] + all_preds[l, i, j, k, :, :, :]


  preds_argmax = torch.argmax(summed_preds, dim=0)

  return preds_argmax

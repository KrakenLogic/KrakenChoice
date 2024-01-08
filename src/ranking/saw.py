
def data_gen():
  rawdata = np.random.randint(low=1, high=15, size=np.random.randint(low=1, high=10, size=(1,2))[0])
  d = np.array(rawdata, dtype=np.longdouble)
  weights = np.random.random(size = (1, d.shape[1]))
  w = (weights/np.sum(weights))[0]

  criterion_type = (np.random.randint(low=0, high=2, size = (1, d.shape[1])))[0]
  c = list(map(lambda x: "max" if x == 1 else "min", criterion_type))
  return d, w, c



# normalization
def normalize_data(data, criterion_type):
  D = data.copy()
  # max
  dmax = D[:, [np.where(np.array(criterion_type) == 'max')[0]]].copy()
  D[:, [np.where(np.array(criterion_type) == 'max')[0]]] = dmax/dmax.max(axis=0)
  # min
  dmin = D[:, [np.where(np.array(criterion_type) == 'min')[0]]].copy()
  D[:, [np.where(np.array(criterion_type) == 'min')[0]]] = dmin.min(axis=0)/dmin

  return D

# calculatopn of saw score
def calculate_saw_scores(normalized_data, weights):
  scores = normalized_data @ weights
  return scores

# ranking
def rank_alternatives(total_scores):
  ranked_indices = np.argsort(total_scores)[::-1]
  # ranked_indices = np.argsort(total_scores)
  return ranked_indices


def main():
  # Step 1:
  data, weights, criterion_type = data_gen()
  # Step 2: Normalize the data
  print(data)
  print(weights)
  print(criterion_type)
  normalized_data = normalize_data(data, criterion_type)
  # Step 3: Calculate total scores
  total_scores = calculate_saw_scores(normalized_data, weights)
  # Step 4: Rank the alternatives
  ranked_indices = rank_alternatives(total_scores)
  # Display the results
  print("\nResults:")
  for i, index in enumerate(ranked_indices):
      print(f"Rank {i + 1}: Alternative {index + 1} - Total Score: {total_scores[index]:.4f}")

if __name__ == "__main__":
    main()



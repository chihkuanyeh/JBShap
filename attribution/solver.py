import numpy as np


class Solver():

  def __init__(self, model, batch_size):
    self.model = model
    self.batch_size = batch_size

  def generate_forward_owen(self, explicand, baseline, players, paths, mask=True, ret_cfs=False, total =False):
    ret = []
    all_cfs = []

    shape = explicand.shape
    batched_shape = list(shape)
    remaining_batch_size = (len(players) + 1) * len(paths)
    batched_shape[0] = min(self.batch_size, remaining_batch_size)

    cfs = np.zeros(batched_shape)
    count = 0
    total_count =0

    for i, path in enumerate(paths):
      cfs[count] = baseline[0]
      prev = cfs[count]
      count += 1
      total_count+=1

      if count == self.batch_size or count == remaining_batch_size:
        if ret_cfs:
          all_cfs.append(cfs)
        ret.append(self.model(cfs).numpy())
        remaining_batch_size -= count
        batched_shape[0] = min(self.batch_size, remaining_batch_size)
        cfs = np.zeros(batched_shape)
        count = 0

      for j, player in enumerate(players):
        if mask:
          cfs[count] = prev * (
              1 - players[paths[i][j]]) + explicand[0] * players[paths[i][j]]
        else:
          x = players[paths[i][j]][0]
          y = players[paths[i][j]][1]
          cfs[count] = prev
          cfs[count, x, y, :] = explicand[0, x, y, :]
        #prev = cfs[count]
        count += 1
        total_count+=1

        if count == self.batch_size or count == remaining_batch_size:
          if ret_cfs:
            all_cfs.append(cfs)
          ret.append(self.modfel(cfs).numpy())
          remaining_batch_size -= count
          batched_shape[0] = min(self.batch_size, remaining_batch_size)
          cfs = np.zeros(batched_shape)
          count = 0
    if total:
      if ret_cfs:
        return np.concatenate(ret), np.concatenate(all_cfs), total_count
      else:
        return np.concatenate(ret), total_count

    if ret_cfs:
        return np.concatenate(ret), np.concatenate(all_cfs)
    return np.concatenate(ret)

  def generate_forward(self, explicand, baseline, players, paths, mask=True, ret_cfs=False, total =False):
    ret = []
    all_cfs = []

    shape = explicand.shape
    batched_shape = list(shape)
    remaining_batch_size = (len(players) + 1) * len(paths)
    batched_shape[0] = min(self.batch_size, remaining_batch_size)

    cfs = np.zeros(batched_shape)
    count = 0
    total_count =0

    for i, path in enumerate(paths):
      cfs[count] = baseline[0]
      prev = cfs[count]
      count += 1
      total_count+=1

      if count == self.batch_size or count == remaining_batch_size:
        if ret_cfs:
          all_cfs.append(cfs)
        ret.append(self.model(cfs).numpy())
        remaining_batch_size -= count
        batched_shape[0] = min(self.batch_size, remaining_batch_size)
        cfs = np.zeros(batched_shape)
        count = 0

      for j, player in enumerate(players):
        if mask:
          cfs[count] = prev * (
              1 - players[paths[i][j]]) + explicand[0] * players[paths[i][j]]
        else:
          x = players[paths[i][j]][0]
          y = players[paths[i][j]][1]
          cfs[count] = prev
          cfs[count, x, y, :] = explicand[0, x, y, :]
        prev = cfs[count]
        count += 1
        total_count+=1

        if count == self.batch_size or count == remaining_batch_size:
          if ret_cfs:
            all_cfs.append(cfs)
          ret.append(self.model(cfs).numpy())
          remaining_batch_size -= count
          batched_shape[0] = min(self.batch_size, remaining_batch_size)
          cfs = np.zeros(batched_shape)
          count = 0
    if total:
      if ret_cfs:
        return np.concatenate(ret), np.concatenate(all_cfs), total_count
      else:
        return np.concatenate(ret), total_count

    if ret_cfs:
        return np.concatenate(ret), np.concatenate(all_cfs)
    return np.concatenate(ret)

  def solve(self, scores, paths, average=False): # output 2d array
    num_paths = len(paths)
    num_players = len(paths[0])
    ret = np.zeros((num_paths, num_players))
    order = np.zeros((num_paths,num_players))
    for i in range(num_paths):
      for j in range(num_players):
        order[i][j] += scores[i*(num_players+1)+j+1] - (
            scores[i*(num_players+1)+j])
        ret[i][paths[i][j]] = scores[i*(num_players+1)+j+1] - (
            scores[i*(num_players+1)+j])
    if average:
      ret = np.mean(ret, axis=0)
    return ret, order

  def solve_owen(self, scores, paths, average=False): # output 2d array
    num_paths = len(paths)
    num_players = len(paths[0])
    ret = np.zeros((num_paths, num_players))
    for i in range(num_paths):
      for j in range(num_players):
        ret[i][paths[i][j]] = scores[i*(num_players+1)+j+1] - (
            scores[i*(num_players+1)])
    if average:
      ret = np.mean(ret, axis=0)
    return ret

import numpy as np
from . import solver
import tensorflow as tf



class SampledShapley():

  def __init__(self, model, batch_size):
    self.solver = solver.Solver(model, batch_size)
    self.model = model
    self.total_forward = 0

  def explain(self, explicand, baseline,
              num_paths, label_index=None):

    shape = explicand.shape
    shapley = np.zeros(explicand.shape)
    members = np.array([[i, j] for i in range(shape[1]) for j in range(shape[2])])
    member_ids = np.arange(len(members))
    member_paths = [np.random.permutation(member_ids) for _ in range(
              num_paths)]
    member_scores_raw, total_f = self.solver.generate_forward(
        explicand,
        baseline,
        members, member_paths, mask=False, total=True)
    member_scores = member_scores_raw[:,label_index]
    self.total_forward += total_f
    members_shapley, shap_array = self.solver.solve(
        member_scores, member_paths, average=True)
    a = tuple(members.T)
    # Evenly split to channels
    b = tuple(np.expand_dims(members_shapley, axis=1) / explicand.shape[-1])
    np.add.at(shapley[0], a, b)
    return shapley, self.total_forward, shap_array

class SampledShapley_topbot():
    
  def __init__(self, model, batch_size):
    self.solver = solver.Solver(model, batch_size)
    self.model = model
    self.total_forward = 0

  def explain(self, explicand, baseline,
              num_paths, label_index=None):

    shape = explicand.shape
    shapley = np.zeros(explicand.shape)
    members = np.array([[i, j] for i in range(shape[1]) for j in range(shape[2])])
    member_ids = np.arange(len(members))
    member_paths = [np.random.permutation(member_ids) for _ in range(
              num_paths)]
    member_scores_raw, total_f = self.solver.generate_forward(
        explicand,
        baseline,
        members, member_paths, mask=False, total=True)
    member_scores = member_scores_raw[:,label_index]
    self.total_forward += total_f
    members_shapley, shap_array = self.solver.solve(
        member_scores, member_paths, average=True)
    a = tuple(members.T)
    # Evenly split to channels
    b = tuple(np.expand_dims(members_shapley, axis=1) / explicand.shape[-1])
    np.add.at(shapley[0], a, b)
    return shapley, self.total_forward, shap_array    

class SampledShapley_variance():
    
  def __init__(self, model, batch_size):
    self.solver = solver.Solver(model, batch_size)
    self.model = model
    self.total_forward = 0

  def explain(self, explicand, baseline,
              num_paths, label_index=None):

    shape = explicand.shape
    shapley = np.zeros(explicand.shape)
    members = np.array([[i, j] for i in range(shape[1]) for j in range(shape[2])])
    member_ids = np.arange(len(members))
    member_paths = [np.random.permutation(member_ids) for _ in range(
              num_paths)]
    member_scores_raw, total_f = self.solver.generate_forward(
        explicand,
        baseline,
        members, member_paths, mask=False, total=True)
    member_scores = member_scores_raw[:,label_index]
    self.total_forward += total_f
    members_shapley, shap_array = self.solver.solve(
        member_scores, member_paths, average=True)
    a = tuple(members.T)
    # Evenly split to channels
    b = tuple(np.expand_dims(members_shapley, axis=1) / explicand.shape[-1])
    np.add.at(shapley[0], a, b)
    return shapley, self.total_forward, shap_array    


class SampledGroupShapley():

  def __init__(self, model, batch_size):
    self.solver = solver.Solver(model, batch_size)
    self.model = model

  def explain(self, explicand, baseline, grouping,
              num_group_paths, label_index=None, top_k_group_paths=3):
    group_ids, counts = np.unique(grouping, return_counts=True)
    num_groups = len(group_ids)
    group_paths = [np.random.permutation(group_ids) for _ in range(
        num_group_paths)]
    groups = [np.expand_dims(grouping == i, axis=-1) for i in range(num_groups)]
    if label_index is None:
      label_index = np.argsort(np.array(self.model(explicand)))[:,-1].item()
    group_scores, group_counterfactuals = self.solver.generate_forward(
        explicand, baseline, groups, group_paths, ret_cfs=True)
    group_scores = group_scores[:, label_index]
    group_shapley = self.solver.solve(group_scores, group_paths, average=True)

    shapley = np.zeros(explicand.shape)

    for i in range(num_groups):
      members = np.argwhere(grouping==i)
      a = tuple(members.T)
      b = group_shapley[i] / (explicand.shape[-1] * len(members))
      np.add.at(shapley[0], a, b)
    return shapley


class SampledShapley_notshareperm():
    


  def __init__(self, model, batch_size):
    self.solver = solver.Solver(model, batch_size)
    self.model = model
    self.total_forward = 0
    self.batch_size = batch_size

  def batch_forward(self,input):
    N = input.shape[0]
    batch_size = self.batch_size
    if N <= batch_size:
      return model(input)
    else:
      output = np.array(self.model(input[:batch_size]))
      size = list(output.shape)
      size[0] = N
      new_output = np.zeros(size)
      new_output[:batch_size] = output
      for i in range(int((N-1)/batch_size)):
        if i==int((N-1)/batch_size):
          new_output[batch_size*(i+1):] = np.array(self.model(input[batch_size*(i+1):]))
        else:
          new_output[batch_size*(i+1):batch_size*(i+2)] = np.array(self.model(input[batch_size*(i+1):batch_size*(i+2)]))
      return new_output

  def explain(self, explicand, baseline,
              num_paths, label_index=None):
    batch_num_paths = 1
    length = 224
    shape = list(explicand.shape)
    print(shape)
    shape[0] = int(num_paths/batch_num_paths)
    
    shapley = np.zeros(shape)
    members = np.array([[i, j] for i in range(shape[1]) for j in range(shape[2])])
    

    member_ids = np.arange(len(members))
    shape = [batch_num_paths,length*length,3]
    explicand = np.reshape(explicand,(1,length*length,3))
    shape_n = shape.copy()
    shape_n.insert(0,len(members))

    for epoch in range(int(num_paths/batch_num_paths)):
      baselines = np.zeros(shape_n)
      baselines_p = np.zeros(shape_n)
      for i in member_ids:
        if i % 5000 ==0:
          print(i)
        member_paths = [np.random.permutation(member_ids) for _ in range(
                batch_num_paths)]
        for j,perm in enumerate(member_paths):
          id = np.where(perm==i)[0][0]
          baselines[i,j,:] = np.reshape(baseline,(length*length,3))
          baselines_p[i,j,:] = np.reshape(baseline,(length*length,3))
          if id >0:
            baselines[i,j,perm[:id],:] = explicand[0,perm[:id],:]
            baselines_p[i,j,perm[:id],:] = explicand[0,perm[:id],:]
          baselines_p[i,j,i,:] = explicand[0,i,:]
      
      
      shape_k = [len(members)*batch_num_paths,length*length,3]

      baselines = np.reshape(baselines,shape_k)
      baselines_p = np.reshape(baselines_p,shape_k)

      shape_l = [len(members)*batch_num_paths,length,length,3]

      baselines = np.reshape(baselines,shape_l)
      baselines_p = np.reshape(baselines_p,shape_l)

      score_0 = self.batch_forward(baselines)[:,label_index]
      score_1 = self.batch_forward(baselines_p)[:,label_index]
      self.total_forward += 2*baselines.shape[0]

      score_avg = np.reshape(score_1-score_0,(len(members),batch_num_paths))
      for i in member_ids:    
        shapley[epoch,members[i][0],members[i][1],:] += np.mean(score_avg[i,:])/3 
      
      del baselines
      del baselines_p

    return shapley, self.total_forward
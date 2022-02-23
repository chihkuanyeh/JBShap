import numpy as np
from . import solver
import time

class SampledOwen():

  def __init__(self, model, batch_size):
    self.solver = solver.Solver(model, batch_size)
    self.model = model
    self.total_forward = 0

  def explain(self, explicand, baseline, grouping,
              num_group_paths, num_member_paths, label_index=None, top_k_group_paths=5):
    group_ids, counts = np.unique(grouping, return_counts=True)
    num_groups = len(group_ids)
    group_paths = [np.random.permutation(group_ids) for _ in range(
        num_group_paths)]
    group_paths_arr = np.array(group_paths)
    groups = [np.expand_dims(grouping == i, axis=-1) for i in range(num_groups)]
    if label_index is None:
      label_index = np.array(np.argsort(self.model(explicand)))[:,-1].item()
    group_scores, group_counterfactuals, total_f = self.solver.generate_forward(
        explicand, baseline, groups, group_paths, ret_cfs=True, total=True)
    self.total_forward += total_f
    group_scores = group_scores[:, label_index]
    group_shapley = self.solver.solve(group_scores, group_paths)
    #print(group_shapley.shape)
    # Check completeness of group shapley
    # print(np.sum(group_shapley) / num_group_paths)

    owen = np.zeros(explicand.shape)
    start = time.time()
    for i in range(num_groups):
      temp_shapley = []
      #print(time.time()-start)
      start = time.time()
      top_path = group_shapley[:,i].argsort()[::-1][:top_k_group_paths]
      #top_path = np.random.permutation(num_group_paths)[:top_k_group_paths]
      total_val = np.sum(group_shapley[:,i])
      top_k_val = np.sum(group_shapley[top_path,i])
      #print(total_val)
      #print(top_k_val)
      for j in range(num_group_paths):
        if j in top_path:
          # TODO(frederickliu): Optimize this for speedup.
          members = np.argwhere(grouping==i)
          member_ids = np.arange(len(members))
          member_paths = [np.random.permutation(member_ids) for _ in range(
              num_member_paths)]
          # Find the beginning
          i0 = np.where(group_paths_arr[j,:]==i)[0][0]
          baseline_idx = j*(num_groups+1)+i0
          member_scores_raw, total_f = self.solver.generate_forward_owen(
              group_counterfactuals[baseline_idx+1:baseline_idx+2],
              group_counterfactuals[baseline_idx:baseline_idx+1],
              members, member_paths, mask=False, total=True)
          member_scores = member_scores_raw[:,label_index]
          self.total_forward += total_f
          members_shapley = self.solver.solve_owen(
              member_scores, member_paths, average=True) / num_group_paths
          a = tuple(members.T)
          #Evenly split to channels
          b = tuple(np.expand_dims(
              members_shapley*1.0*total_val/top_k_val, axis=1) / explicand.shape[-1])
          #b = tuple(np.expand_dims(
          #    members_shapley, axis=1) / explicand.shape[-1])
          np.add.at(owen[0], a, b)
        '''
        else:
          members = np.argwhere(grouping==group_paths[j][i])
          a = tuple(members.T)
          b = group_shapley[j, group_paths[j][i]] / (explicand.shape[-1] * len(members) * num_group_paths)
          np.add.at(owen[0], a, b)
        '''
    return owen, self.total_forward


class SampledOwenLinear():
    
  def __init__(self, model, batch_size):
    self.solver = solver.Solver(model, batch_size)
    self.model = model
    self.total_forward = 0

  def explain(self, explicand, baseline, grouping,
              num_group_paths, num_member_paths, label_index=None, top_k_group_paths=5):
    group_ids, counts = np.unique(grouping, return_counts=True)
    num_groups = len(group_ids)
    group_paths = [np.random.permutation(group_ids) for _ in range(
        num_group_paths)]
    group_paths_arr = np.array(group_paths)
    groups = [np.expand_dims(grouping == i, axis=-1) for i in range(num_groups)]
    if label_index is None:
      label_index = np.argsort(np.array(self.model(explicand)))[:,-1].item()
    group_scores, group_counterfactuals, total_f = self.solver.generate_forward(
        explicand, baseline, groups, group_paths, ret_cfs=True, total=True)
    self.total_forward += total_f
    group_scores = group_scores[:, label_index]
    group_shapley = self.solver.solve(group_scores, group_paths)
    #print(group_shapley.shape)
    # Check completeness of group shapley
    # print(np.sum(group_shapley) / num_group_paths)

    owen = np.zeros(explicand.shape)
    start = time.time()
    for i in range(num_groups):
      temp_shapley = []
      #print(time.time()-start)
      start = time.time()
      top_path = group_shapley[:,i].argsort()[::-1][:top_k_group_paths]
      #top_path = np.random.permutation(num_group_paths)[:top_k_group_paths]
      total_val = np.sum(group_shapley[:,i])
      top_k_val = np.sum(group_shapley[top_path,i])
      #print(total_val)
      #print(top_k_val)
      for j in range(num_group_paths):
        if j in top_path:
          # TODO(frederickliu): Optimize this for speedup.
          members = np.argwhere(grouping==i)
          member_ids = np.arange(len(members))
          member_paths = [np.random.permutation(member_ids) for _ in range(
              num_member_paths)]
          # Find the beginning
          i0 = np.where(group_paths_arr[j,:]==i)[0][0]
          baseline_idx = j*(num_groups+1)+i0
          member_scores_raw, total_f = self.solver.generate_forward(
              group_counterfactuals[baseline_idx+1:baseline_idx+2],
              group_counterfactuals[baseline_idx:baseline_idx+1],
              members, member_paths, mask=False, total=True)
          member_scores = member_scores_raw[:,label_index]
          self.total_forward += total_f
          members_shapley = self.solver.solve(
              member_scores, member_paths, average=True) / num_group_paths
          a = tuple(members.T)
          #Evenly split to channels
          b = tuple(np.expand_dims(
              members_shapley*1.0*total_val/top_k_val, axis=1) / explicand.shape[-1])
          #b = tuple(np.expand_dims(
          #    members_shapley, axis=1) / explicand.shape[-1])
          np.add.at(owen[0], a, b)
        '''
        else:
          members = np.argwhere(grouping==group_paths[j][i])
          a = tuple(members.T)
          b = group_shapley[j, group_paths[j][i]] / (explicand.shape[-1] * len(members) * num_group_paths)
          np.add.at(owen[0], a, b)
        '''
    return owen, self.total_forward

class SampledOwen_notshareperm():

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

  def explain(self, explicand, baseline, grouping,
              num_paths, label_index=None):
    #print(grouping.shape)
    batch_num_paths = 5
    #length = 224
    length = 96

    group_ids, counts = np.unique(grouping, return_counts=True)
    num_groups = len(group_ids)
    group_member = []
    for i in range(num_groups):
      group_member.append(grouping==i)
    group_member = np.array(group_member)

    shape = list(explicand.shape)
    shape[0] = int(num_paths/batch_num_paths)
    
    shapley = np.zeros(shape)
    members = np.array([[i, j] for i in range(shape[1]) for j in range(shape[2])])
    

    member_ids = np.arange(len(members))
    shape = [batch_num_paths,length*length,3]
    explicand = np.reshape(explicand,(1,length*length,3))
    shape_n = shape.copy()
    shape_n.insert(0,len(members))
    
    baseliner = np.reshape(baseline,(length*length,3))

    for epoch in range(int(num_paths/batch_num_paths)):
      if epoch %10 ==0:
        print(epoch)
          
      baselines = np.zeros(shape_n)
      baselines_p = np.zeros(shape_n)

      
      for i in member_ids:
        group_paths = [np.random.permutation(group_ids) for _ in range(
          batch_num_paths)]
        for j,perm in enumerate(group_paths):
          baselines[i,j,:] = baseliner
          baselines_p[i,j,:] = baseliner
          #print(perm)
          #print(members.shape)
          #print(members[i][0])
          #print(grouping[members[i][0], members[i][1]])
          #print(np.where(perm==grouping[members[i][0], members[i][1]]))
          id = np.where(perm==grouping[i])[0][0]
          memberst = np.where(np.sum(group_member[perm[:id],:], axis=0))[0]
          '''
          for k in range(id):
            memberst = np.argwhere(grouping==perm[k])
            if id >0:
              baselines[i,j,memberst,:] = explicand[0,memberst,:]
              baselines_p[i,j,memberst,:] = explicand[0,memberst,:]
          '''
          baselines[i,j,memberst,:] = explicand[0,memberst,:]
          baselines_p[i,j,memberst,:] = explicand[0,memberst,:]
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
        shapley[epoch,members[i][0],members[i][1],:] = np.mean(score_avg[i,:])/3 

      del baselines
      del baselines_p

    return shapley, self.total_forward


class SampledOwen_shareperm():

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

  def explain(self, explicand, baseline, grouping,
              num_paths, label_index=None):
    #print(grouping.shape)
    batch_num_paths = 1
    length = 224

    group_ids, counts = np.unique(grouping, return_counts=True)
    num_groups = len(group_ids)
    group_member = []
    for i in range(num_groups):
      group_member.append(grouping==i)
    group_member = np.array(group_member)

    shape = list(explicand.shape)
    shape[0] = int(num_paths/batch_num_paths)
    
    shapley = np.zeros(shape)
    members = np.array([[i, j] for i in range(shape[1]) for j in range(shape[2])])
    

    member_ids = np.arange(len(members))
    shape = [batch_num_paths,length*length,3]
    explicand = np.reshape(explicand,(1,length*length,3))
    shape_n = shape.copy()
    shape_n.insert(0,len(members))
    shape_ch = shape.copy()
    shape_ch.insert(0,num_groups)
    
    baseliner = np.reshape(baseline,(length*length,3))

    for epoch in range(int(num_paths/batch_num_paths)):
          
      #baselines = np.zeros(shape_n)
      baselines_ch = np.zeros(shape_ch)
      baselines_p = np.zeros(shape_n)
      count = 0
      for g in range(num_groups):
        group_paths = [np.random.permutation(group_ids) for _ in range(
            batch_num_paths)]        
        for j,perm in enumerate(group_paths):
          baselines_ch[g,j,:] = baseliner
          #baselines[group_member[g],j,:] = baseliner
          baselines_p[group_member[g],j,:] = baseliner
          id = np.where(perm==g)[0][0]
          memberst = np.where(np.sum(group_member[perm[:id],:], axis=0))[0]    
          baselines_ch[g,j,memberst,:] = explicand[0,memberst,:]
          #baselines[np.ix_(group_member[g],[j],memberst,np.arange(3))] = explicand[0,memberst,:]
          baselines_p[np.ix_(group_member[g],[j],memberst,np.arange(3))] = explicand[0,memberst,:]
          for i in np.where(group_member[g])[0]:
            if count % 5000 ==0:
              print(count)
            count +=1
            baselines_p[i,j,i,:] = explicand[0,i,:]

      shape_k = [-1,length*length,3]
      #baselines = np.reshape(baselines,shape_k)
      baselines_ch = np.reshape(baselines_ch,shape_k)
      baselines_p = np.reshape(baselines_p,shape_k)

      shape_l = [-1,length,length,3]
      #baselines = np.reshape(baselines,shape_l)
      baselines_ch = np.reshape(baselines_ch,shape_l)
      baselines_p = np.reshape(baselines_p,shape_l)

      #score_0 = self.batch_forward(baselines)[:,label_index]
      score_0_ch = np.reshape(self.batch_forward(baselines_ch)[:,label_index],(num_groups,batch_num_paths))
      score_0 = np.zeros((len(members),batch_num_paths))
      for i in range(score_0_ch.shape[0]):
        score_0[group_member[i],:] = score_0_ch[i,:] 
      score_0 = score_0.flatten()

      score_1 = self.batch_forward(baselines_p)[:,label_index]
      self.total_forward += 2*baselines_p.shape[0]

      score_avg = np.reshape(score_1-score_0,(len(members),batch_num_paths))
      for i in member_ids:    
        shapley[epoch,members[i][0],members[i][1],:] = np.mean(score_avg[i,:])/3 

      del baselines_ch
      del baselines_p

    return shapley, self.total_forward
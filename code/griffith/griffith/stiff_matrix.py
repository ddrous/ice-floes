import numpy as np
from scipy import sparse
import warnings

from .scalar_product import scalar_product
from .finite_elements import Element, Affine_Element, Mid_Element

warnings.filterwarnings('error', category=DeprecationWarning)

def compute_interior_stiff_matrix(field, stiffness_tensor):
  """
  Pretty straight forward.
  We compute all the scalar products.
  """
  row, col, data = [], [], []
  for i, element_i in enumerate(field.interior_elements):
    for node in element_i.base_node.neighboors.union(set([element_i.base_node])):
      for element_j in field.elements_on_node[node]:
        if element_j.type is Element.Type.INTERIOR:
          j = element_j.index
          if j < i:
            ps = scalar_product(stiffness_tensor, element_i, element_j)
            row.extend([i, j]), col.extend([j, i])
            data.extend([ps, ps])
          elif j == i:
            ps = scalar_product(stiffness_tensor, element_i, element_j)
            row.append(j), col.append(j)
            data.append(ps)
  return sparse.csr_matrix((data, (row, col)), shape=(len(field.interior_elements), len(field.interior_elements))), (data, row, col)


def compute_boundary_matrix(field, stiffness_tensor):
  row, col, data = [], [], []
  for i, element_i in enumerate(field.interior_elements):
    for node in element_i.base_node.neighboors.union(set([element_i.base_node])):
      for element_j in field.elements_on_node[node]:
        if element_j.type is Element.Type.BOUNDARY:
          j = element_j.index
          ps = scalar_product(stiffness_tensor, element_i, element_j)
          row.append(i), col.append(j)
          data.append(ps)
  return sparse.csr_matrix((data, (row, col)), shape=(len(field.interior_elements), len(field.boundary_elements)))


def compute_boundary_stiff(field, stiffness_tensor):
  boundary_stiff = np.zeros((len(field.boundary_elements), len(field.boundary_elements)))
  for i, element_i in enumerate(field.boundary_elements) :
    for node in element_i.base_node.neighboors.union(set([element_i.base_node])):
      for element_j in field.elements_on_node[node]:
        if element_j.type is Element.Type.BOUNDARY:
          j = element_j.index
          if boundary_stiff[i, j] == 0:
            boundary_stiff[i,j] = scalar_product(stiffness_tensor, element_i, element_j)
            boundary_stiff[j,i] = boundary_stiff[i, j]
  return boundary_stiff


def modify_stiff_matrix(sparse_data, enriched_field, stiffness_tensor):
  data, row, col = sparse_data
  data, row, col = data.copy(), row.copy(), col.copy()

  i = 0
  modify = enriched_field.removed_interior_indexes + enriched_field.modified_interior_indexes
  for _ in range(len(row)):
    if row[i] in modify or col[i] in modify:
      data.pop(i)
      row.pop(i)
      col.pop(i)
    else:
      row[i] = enriched_field.old_to_new_indexes[row[i]]
      col[i] = enriched_field.old_to_new_indexes[col[i]]
      i += 1
  
  for element_i in enriched_field.modified_interior_elements:
    for node in element_i.base_node.neighboors.union(set([element_i.base_node])):
      for element_j in enriched_field.elements_on_node[node]:
        if element_j.type is Element.Type.INTERIOR:
          if type(element_j) is Affine_Element:
            i, j = element_i.index, element_j.index
            ps = scalar_product(stiffness_tensor, element_i, element_j)
            row.extend([i, j]), col.extend([j, i])
            data.extend([ps, ps])

  for element_i in enriched_field.new_interior_elements:
    for node in element_i.base_node.neighboors.union(set([element_i.base_node])):
      for element_j in enriched_field.elements_on_node[node]:
        if element_j.type is Element.Type.INTERIOR:
          i, j = element_i.index, element_j.index
          if element_j in enriched_field.new_interior_elements:
            if j<i:
              ps = scalar_product(stiffness_tensor, element_i, element_j)
              row.extend([i, j]), col.extend([j, i])
              data.extend([ps, ps])
            elif i == j:
              ps = scalar_product(stiffness_tensor, element_i, element_j)
              row.append(i), col.append(i)
              data.append(ps) 
          else:
            ps = scalar_product(stiffness_tensor, element_i, element_j)
            row.extend([i, j]), col.extend([j, i])
            data.extend([ps, ps])

  return sparse.csr_matrix((data, (row, col)), shape=(len(enriched_field.interior_elements), len(enriched_field.interior_elements)))
